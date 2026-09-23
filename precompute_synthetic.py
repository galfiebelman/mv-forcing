"""
Turn the released SynCamVideo Dataset into the tensors the synthetic training
stages read.

The dataset ships 81-frame 1280x1280 videos and per-scene camera extrinsics. For
each scene this script writes, next to the videos:

    camXX.pth                     {"latents": [16, 21, 60, 104],
                                   "prompt_emb": {"context": [1, 512, 4096]},
                                   "image_emb": {}}
    final_pairs.npy               valid (source, target) view pairs
    final_ode_pair.npy            the pair generate_ode_pairs.py uses
    cutt3r_raymap/SS_TT.pth       {"z_rendered": [17, 21, 60, 104]} per pair

and a metadata.csv at the dataset root listing every video.

Frame handling follows SynCamMaster's data_process step: 81 frames, scaled so
both sides cover the target, centre-cropped to 480x832, normalized to [-1, 1].

Valid pairs follow SynCamMaster's generate_sample_list.py: ordered pairs whose
azimuth differs by less than --az_diff_max and elevation by less than
--el_diff_max. That script reads a rotation_data.json that the public dataset
does not include, so the two angles are recovered here from the extrinsics: about
the point where the camera forward rays intersect.

Usage:
    python precompute_synthetic.py \\
        --dataset_dir data/SynCamVideo-Dataset/train/f24_aperture5 \\
        --captions captions.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

from utils.paths import load_cut3r, set_wan_root
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def load_c2ws(cam_data, view_idx, frame_indices):
    """Camera-to-world matrices in the convention the model is trained with."""
    traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"])
            for idx in frame_indices]
    traj = np.stack(traj).transpose(0, 2, 1)
    c2ws = []
    for c2w in traj:
        c2w = c2w[:, [1, 2, 0, 3]]
        c2w[:3, 1] *= -1.
        c2w[:3, 3] /= 200
        c2ws.append(c2w)
    return c2ws


def scene_angles(cam_data, num_views):
    """Azimuth and elevation of every camera about the point they look at.

    SynCamMaster's generate_sample_list.py reads these from a rotation_data.json
    that the public dataset does not include, so they are recovered here: the
    look-at point is the least-squares intersection of the camera forward rays,
    and each camera's angles are taken about it. Measuring about the world origin
    would be wrong -- the rig sits far from it, so every camera would appear to
    share almost the same angle.
    """
    origins, directions = [], []
    for view in range(1, num_views + 1):
        m = parse_matrix(cam_data["frame0"][f"cam{view:02d}"]).T
        forward = m[:3, 0] / np.linalg.norm(m[:3, 0])
        origins.append(m[:3, 3])
        directions.append(forward)

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for origin, direction in zip(origins, directions):
        projection = np.eye(3) - np.outer(direction, direction)
        A += projection
        b += projection @ origin
    center = np.linalg.lstsq(A, b, rcond=None)[0]

    angles = {}
    for view, origin in enumerate(origins, start=1):
        rel = origin - center
        radius = float(np.linalg.norm(rel))
        angles[view] = (
            float(np.degrees(np.arctan2(rel[1], rel[0]))),
            float(np.degrees(np.arcsin(rel[2] / radius))) if radius > 0 else 0.0,
        )
    return angles


def chain_subset(pairs, limit):
    """A subset that keeps view-sequential unrolling possible.

    Unrolling needs a pair whose source is the previous pair's target, so simply
    truncating the list can leave every pair starting from the same view and no
    continuation. Walk the pairs as a graph instead, so each kept pair extends
    the chain. Immediate backtracking is avoided: the training dataset skips a
    third view that is the pair's own source view, so a 1->2->1 chain is not a
    usable continuation.
    """
    if limit is None or limit >= len(pairs):
        return pairs
    outgoing = {}
    for src, tgt in pairs:
        outgoing.setdefault(src, []).append(tgt)
    kept, used, node, prev = [], set(), pairs[0][0], None
    while len(kept) < limit:
        candidates = [t for t in outgoing.get(node, []) if (node, t) not in used]
        nxt = next((t for t in candidates if t != prev), None)
        if nxt is None:                     # chain exhausted: restart elsewhere
            remaining = [p for p in pairs if tuple(p) not in used]
            if not remaining:
                break
            node, prev = remaining[0][0], None
            continue
        kept.append((node, nxt))
        used.add((node, nxt))
        node, prev = nxt, node

    # The dataset only ever sees the kept list, so a pair whose target has no
    # continuation inside it is unusable. Drop those until a fixed point, which
    # leaves closed cycles over distinct views (this can return fewer than
    # `limit` pairs).
    while True:
        usable = [(a, b) for a, b in kept
                  if any(c == b and d != a for c, d in kept)]
        if len(usable) == len(kept):
            return kept
        kept = usable


def valid_pairs(cam_data, num_views, az_diff_max, el_diff_max):
    angles = scene_angles(cam_data, num_views)
    pairs = []
    for i in range(1, num_views + 1):
        for j in range(1, num_views + 1):
            if i == j:
                continue
            az_i, el_i = angles[i]
            az_j, el_j = angles[j]
            if abs(az_i - az_j) < az_diff_max and abs(el_i - el_j) < el_diff_max:
                pairs.append((i, j))
    return pairs


def load_video_frames(path, num_frames, height, width):
    """Frames as [C, T, H, W] in [-1, 1], following SynCamMaster's preprocessing."""
    import imageio

    reader = imageio.get_reader(path)
    frames = []
    for frame_id in range(num_frames):
        frame = Image.fromarray(reader.get_data(frame_id))
        w, h = frame.size
        scale = max(width / w, height / h)
        frame = frame.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        left = (frame.size[0] - width) // 2
        top = (frame.size[1] - height) // 2
        frame = frame.crop((left, top, left + width, top + height))
        tensor = torch.from_numpy(np.array(frame)).permute(2, 0, 1).float() / 255.0
        frames.append(tensor * 2.0 - 1.0)
    reader.close()
    return rearrange(torch.stack(frames, dim=0), "t c h w -> c t h w")


@torch.no_grad()
def encode_view(video_path, caption, vae, text_encoder, args, device):
    pixels = load_video_frames(video_path, args.num_frames, args.height, args.width)
    latents = vae.encode_to_latent(
        pixels.unsqueeze(0).to(device=device, dtype=torch.bfloat16))
    latents = latents[0].permute(1, 0, 2, 3).cpu()      # [C, F, h, w]
    context = text_encoder(text_prompts=[caption])["prompt_embeds"].cpu()
    return {"latents": latents, "prompt_emb": {"context": context}, "image_emb": {}}


@torch.no_grad()
def render_prior(src_latents, c2w_src, c2w_tgt, vae, cut3r_model, args, device):
    """CUT3R geometric prior of the target view, as [17, F, h, w]."""
    from pipeline.cut3r_prior import (
        _postprocess_rendered_to_z, _render_raymap_frames, _run_cut3r_on_frames,
    )

    latents = src_latents.permute(1, 0, 2, 3).unsqueeze(0).to(
        device=device, dtype=torch.bfloat16)                     # [1, F, C, h, w]
    pixels = vae.decode_to_pixel(latents)
    pixels = (pixels * 0.5 + 0.5).clamp(0, 1)[0]
    vae.model.clear_cache()

    num_prior_frames = min(latents.shape[1], c2w_src.shape[0])
    frames = pixels[::4][:num_prior_frames]
    outputs, state_args, _, H_c, W_c = _run_cut3r_on_frames(
        frames, cut3r_model, args.cut3r_size, device)
    H_out, W_out = frames.shape[2], frames.shape[3]

    rendered, conf = _render_raymap_frames(
        outputs, state_args, c2w_src[:num_prior_frames], c2w_tgt[:num_prior_frames],
        args.focal_ratio, list(range(num_prior_frames)),
        H_c, W_c, H_out, W_out, device, cut3r_model)
    conf_video = torch.stack(conf)
    masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1) for c in conf]
    z_rendered = _postprocess_rendered_to_z(
        rendered, masks, vae, H_out, W_out, device)
    vae.model.clear_cache()
    return z_rendered[0].cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory holding the sceneN folders, "
                             "e.g. data/SynCamVideo-Dataset/train/f24_aperture5")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                        help="Scene folder names; defaults to every scene found")
    parser.add_argument("--captions", type=str, default=None,
                        help="CSV with columns scene,text. Scenes missing from it "
                             "fall back to --default_caption")
    parser.add_argument("--default_caption", type=str,
                        default="A person moving in a realistic environment, "
                                "static camera, fixed viewpoint.")
    parser.add_argument("--pairs_filename", type=str, default="final_pairs.npy")
    parser.add_argument("--max_pairs_per_scene", type=int, default=None,
                        help="Keep only this many valid pairs per scene; priors are "
                             "written for exactly the pairs kept")
    parser.add_argument("--az_diff_max", type=float, default=30.0)
    parser.add_argument("--el_diff_max", type=float, default=15.0)
    parser.add_argument("--num_views", type=int, default=10)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--skip_priors", action="store_true",
                        help="Only write latents and the pair list")

    parser.add_argument("--wan_root", type=str, default=None)
    parser.add_argument("--cut3r_root", type=str, default=None)
    parser.add_argument("--cut3r_path", type=str,
                        default="CUT3R/src/cut3r_512_dpt_4_64.pth")
    parser.add_argument("--cut3r_size", type=int, default=512)
    parser.add_argument("--focal_ratio", type=float, default=24.0 / 23.76)

    args = parser.parse_args()
    set_wan_root(args.wan_root)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    scenes = args.scenes or sorted(
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d)) and d.startswith("scene"))
    print(f"{len(scenes)} scene(s): {scenes}")

    captions = {}
    if args.captions is not None:
        table = pd.read_csv(args.captions)
        captions = dict(zip(table["scene"], table["text"]))

    print("Loading VAE and text encoder...")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)
    text_encoder = WanTextEncoder().to(device=device)
    cut3r_model = None

    frame_indices = list(range(args.num_frames))[::4]
    metadata_rows = []

    for scene in scenes:
        scene_dir = os.path.join(args.dataset_dir, scene)
        caption = captions.get(scene, args.default_caption)
        with open(os.path.join(scene_dir, "cameras", "camera_extrinsics.json")) as f:
            cam_data = json.load(f)

        # 1. Latents and prompt embeddings, one file per view.
        for view in range(1, args.num_views + 1):
            video_path = os.path.join(scene_dir, "videos", f"cam{view:02d}.mp4")
            out_path = video_path.replace(".mp4", ".pth")
            rel = os.path.relpath(video_path, os.path.dirname(args.dataset_dir))
            metadata_rows.append({"file_name": rel, "text": caption})
            if args.skip_existing and os.path.exists(out_path):
                continue
            torch.save(
                encode_view(video_path, caption, vae, text_encoder, args, device),
                out_path)
            print(f"  {scene}/cam{view:02d}: latents written")

        # 2. Valid view pairs.
        pairs = valid_pairs(cam_data, args.num_views, args.az_diff_max, args.el_diff_max)
        if args.max_pairs_per_scene is not None:
            pairs = chain_subset(pairs, args.max_pairs_per_scene)
        np.save(os.path.join(scene_dir, args.pairs_filename), np.array(pairs))
        # generate_ode_pairs.py reads one pair per scene from its own file.
        np.save(os.path.join(scene_dir, "final_ode_pair.npy"), np.array(pairs[0]))
        print(f"  {scene}: {len(pairs)} valid pairs -> {args.pairs_filename} "
              f"(ODE pair {pairs[0]})")

        if args.skip_priors:
            continue

        # 3. CUT3R geometric prior per pair.
        if cut3r_model is None:
            print("Loading CUT3R...")
            cut3r_model = load_cut3r(args.cut3r_path, device, root=args.cut3r_root)
        prior_dir = os.path.join(scene_dir, "cutt3r_raymap")
        os.makedirs(prior_dir, exist_ok=True)
        c2ws = {v: torch.tensor(np.stack(load_c2ws(cam_data, v, frame_indices)),
                                dtype=torch.float32)
                for v in range(1, args.num_views + 1)}
        for src, tgt in pairs:
            out_path = os.path.join(prior_dir, f"{src:02d}_{tgt:02d}.pth")
            if args.skip_existing and os.path.exists(out_path):
                continue
            src_latents = torch.load(
                os.path.join(scene_dir, "videos", f"cam{src:02d}.pth"),
                weights_only=True, map_location="cpu")["latents"]
            z_rendered = render_prior(
                src_latents, c2ws[src], c2ws[tgt], vae, cut3r_model, args, device)
            torch.save({"z_rendered": z_rendered}, out_path)
            print(f"  {scene}: prior {src:02d}->{tgt:02d} written")

    metadata_path = os.path.join(os.path.dirname(args.dataset_dir), "metadata.csv")
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    print(f"metadata.csv written with {len(metadata_rows)} rows -> {metadata_path}")


if __name__ == "__main__":
    main()
