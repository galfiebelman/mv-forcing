import argparse
import os
import sys
import random
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
import lmdb
import json
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# VAE + T5 (from training codebase)
from utils.wan_wrapper import WanVAEWrapper, WanTextEncoder
from utils.paths import add_cut3r_to_path, load_cut3r


# ============================================================================
# LMDB helpers (inlined to avoid import issues)
# ============================================================================

def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        shape_str = txn.get(f"{array_name}_shape".encode()).decode()
        return tuple(map(int, shape_str.split()))


def retrieve_row_from_lmdb(env, array_name, dtype, row_index, shape=None):
    data_key = f'{array_name}_{row_index}_data'.encode()
    with env.begin() as txn:
        row_bytes = txn.get(data_key)
    if dtype == str:
        return row_bytes.decode()
    array = np.frombuffer(row_bytes, dtype=dtype)
    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array


# ============================================================================
# Camera utilities
# ============================================================================

class Camera:
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)


def load_camera_pair(cam_json_path, src_idx, tgt_idx):
    """
    Load c2w matrices for a (source, target) pair from SynCamVideo extrinsics.
    Returns: c2w_src (21, 4, 4), c2w_tgt (21, 4, 4)
    """
    with open(cam_json_path, 'r') as f:
        cam_data = json.load(f)

    cam_idx = list(range(81))[::4]  # [0, 4, 8, ..., 80] → 21 frames
    c2ws = {}
    for view_idx in [src_idx, tgt_idx]:
        traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{int(view_idx):02d}"])
                for idx in cam_idx]
        traj = np.stack(traj).transpose(0, 2, 1)
        view_c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 200
            view_c2ws.append(c2w)
        c2ws[view_idx] = np.stack(view_c2ws)

    c2w_src = torch.tensor(c2ws[src_idx], dtype=torch.float32)
    c2w_tgt = torch.tensor(c2ws[tgt_idx], dtype=torch.float32)
    return c2w_src, c2w_tgt


def get_relative_pose(cam_params):
    abs_w2cs = [c.w2c_mat for c in cam_params]
    abs_c2ws = [c.c2w_mat for c in cam_params]
    target_cam_c2w = np.eye(4, dtype=np.float32)
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w] + [abs2rel @ c for c in abs_c2ws[1:]]
    return np.array(ret_poses, dtype=np.float32)


def compute_camera_embedding(c2w_src, c2w_tgt):
    """
    Compute relative pose embedding (2, 21, 12) matching TensorDataset convention.
    """
    src_cams = [Camera(c2w_src[i].numpy()) for i in range(c2w_src.shape[0])]
    tgt_cams = [Camera(c2w_tgt[i].numpy()) for i in range(c2w_tgt.shape[0])]

    relative_poses = []
    for i in range(len(src_cams)):
        rel = get_relative_pose([src_cams[i], tgt_cams[i]])
        relative_poses.append(torch.as_tensor(rel)[:, :3, :])  # (2, 3, 4)

    pose_embedding = torch.stack(relative_poses, dim=1)  # (2, 21, 3, 4)
    pose_embedding = rearrange(pose_embedding, 'v f c d -> v f (c d)')  # (2, 21, 12)
    return pose_embedding


# ============================================================================
# SynCamVideo scene scanning
# ============================================================================

def scan_synccam_scenes(synccam_path, pairs_filename="final_2_views_2.npy"):
    """Scan SynCamVideo dataset for scenes with valid camera pairs."""
    scenes = []
    for scene_name in sorted(os.listdir(synccam_path)):
        scene_dir = os.path.join(synccam_path, scene_name)
        if not os.path.isdir(scene_dir):
            continue
        pairs_path = os.path.join(scene_dir, pairs_filename)
        cam_path = os.path.join(scene_dir, "cameras", "camera_extrinsics.json")
        if os.path.exists(pairs_path) and os.path.exists(cam_path):
            pairs = np.load(pairs_path)
            if len(pairs) > 0:
                scenes.append({
                    'scene_dir': scene_dir,
                    'pairs': pairs,
                    'cam_path': cam_path,
                })
    print(f"Found {len(scenes)} SynCamVideo scenes with valid pairs")
    return scenes


# ============================================================================
# CUT3R: run + render raymap (adapted from precompute_cutt3r_raymap.py)
# ============================================================================

def run_cut3r(frames_01, cut3r_model, cut3r_size, device):
    """
    Run CUT3R on source frames.
    Args:
        frames_01: (F, 3, H, W) float32 in [0, 1]
    Returns:
        outputs, state_args, H_c, W_c
    """
    H_out, W_out = frames_01.shape[2], frames_01.shape[3]
    scale = cut3r_size / max(H_out, W_out)
    H_c = int(H_out * scale) // 16 * 16
    W_c = int(W_out * scale) // 16 * 16
    frames_resized = F.interpolate(
        frames_01, size=(H_c, W_c), mode='bilinear', align_corners=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(frames_resized.shape[0]):
            frame = (frames_resized[i].float().permute(1, 2, 0).cpu().numpy() * 255
                     ).astype(np.uint8)
            path = os.path.join(tmpdir, f'{i:04d}.png')
            Image.fromarray(frame).save(path)
            paths.append(path)
        from dust3r.utils.image import load_images as cut3r_load_images
        images = cut3r_load_images(paths, size=cut3r_size)

    views = []
    for i, image in enumerate(images):
        views.append({
            'img': image['img'],
            'ray_map': torch.full(
                (image['img'].shape[0], 6,
                 image['img'].shape[-2], image['img'].shape[-1]),
                torch.nan),
            'true_shape': torch.from_numpy(image['true_shape']),
            'idx': i, 'instance': str(i),
            'camera_pose': torch.from_numpy(
                np.eye(4, dtype=np.float32)).unsqueeze(0),
            'img_mask': torch.tensor(True).unsqueeze(0),
            'ray_mask': torch.tensor(False).unsqueeze(0),
            'update': torch.tensor(True).unsqueeze(0),
            'reset': torch.tensor(False).unsqueeze(0),
        })

    from dust3r.inference import inference as cut3r_inference
    outputs, state_args = cut3r_inference(views, cut3r_model, device)
    return outputs, state_args, H_c, W_c


def render_raymap(outputs, state_args, c2w_src, c2w_tgt,
                  focal_ratio, H_c, W_c, H_out, W_out, device, cut3r_model):
    """
    Query CUT3R with target-camera raymaps for all 21 frames.
    Returns: rendered_frames (list of (3, H_c, W_c)), conf_frames (list of (1, H_c, W_c))
    """
    num_frames = c2w_src.shape[0]

    # Intrinsics
    real_focal = focal_ratio * W_out
    focal = real_focal * (W_c / W_out)
    cx, cy = W_c // 2, H_c // 2
    K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
                     device=device, dtype=torch.float32)
    K_inv = torch.inverse(K)

    # Pixel grid
    i, j = torch.meshgrid(
        torch.arange(W_c, device=device, dtype=torch.float32),
        torch.arange(H_c, device=device, dtype=torch.float32), indexing='xy')
    grid = torch.stack([i, j, torch.ones_like(i)], dim=-1)
    grid_flat = K_inv @ grid.reshape(-1, 3).T
    grid_hom = torch.vstack([grid_flat,
                             torch.ones(1, grid_flat.shape[1], device=device)])

    # CUT3R predicted source poses
    cut3r_c2w_src = []
    for t in range(num_frames):
        from dust3r.utils.camera import pose_encoding_to_camera
        pose = pose_encoding_to_camera(
            outputs['pred'][t]['camera_pose'].clone()
        ).squeeze(0).to(device)
        cut3r_c2w_src.append(pose)

    rendered_frames, conf_frames = [], []
    for t in range(num_frames):
        # Relative transform: source_t → target_t
        dataset_c2w_src_t = c2w_src[t].to(device).float()
        dataset_c2w_tgt_t = c2w_tgt[t].to(device).float()
        rel = (torch.inverse(dataset_c2w_src_t) @ dataset_c2w_tgt_t).float()

        # Target pose in CUT3R frame
        cut3r_c2w_tgt_t = (cut3r_c2w_src[t] @ rel).float()

        # Raymap
        ro = cut3r_c2w_tgt_t[:3, 3].expand(H_c, W_c, 3)
        rd = (cut3r_c2w_tgt_t @ grid_hom).T[:, :3].reshape(H_c, W_c, 3).float()
        rd = F.normalize(rd, dim=-1)
        raymap_tensor = torch.cat([ro, rd], dim=-1).unsqueeze(0)

        view_query = {
            'img': torch.full((1, 3, H_c, W_c), torch.nan, device=device),
            'ray_map': raymap_tensor,
            'true_shape': torch.tensor([[H_c, W_c]]),
            'idx': len(outputs['pred']) + t,
            'instance': str(len(outputs['pred']) + t),
            'camera_pose': torch.from_numpy(
                np.eye(4, dtype=np.float32)).unsqueeze(0),
            'img_mask': torch.tensor(False).unsqueeze(0),
            'ray_mask': torch.tensor(True).unsqueeze(0),
            'update': torch.tensor(False).unsqueeze(0),
            'reset': torch.tensor(False).unsqueeze(0),
        }

        from dust3r.inference import inference_step as cut3r_inference_step
        query_output = cut3r_inference_step(
            view_query, state_args[t + 1], cut3r_model, device)
        pred = query_output['pred']
        rgb = (pred['rgb'].squeeze(0) + 1.0) / 2.0
        conf = pred['conf'].squeeze(0)

        rendered_frames.append(rgb.permute(2, 0, 1))   # (3, H_c, W_c)
        conf_frames.append(conf.unsqueeze(0))           # (1, H_c, W_c)

    return rendered_frames, conf_frames


def postprocess_to_z_rendered(rendered_frames, conf_frames, H_out, W_out,
                               vae, device):
    """
    rendered frames + conf → VAE encode → (17, 21, 60, 104) z_rendered tensor.
    """
    rendered_video = torch.stack(rendered_frames).to(device)  # (21, 3, H_c, W_c)
    conf_video = torch.stack(conf_frames).to(device)          # (21, 1, H_c, W_c)

    # Normalize confidence → soft mask
    mask_frames = ((conf_video - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)

    # Spatial resize
    rendered_video = F.interpolate(
        rendered_video, size=(H_out, W_out), mode='bilinear', align_corners=False)
    mask_video = F.interpolate(
        mask_frames, size=(H_out, W_out), mode='nearest')

    # Temporal upsample 21 → 81
    rendered_5d = rearrange(rendered_video, 'f c h w -> 1 c f h w')
    rendered_81 = F.interpolate(rendered_5d, size=(81, H_out, W_out), mode='nearest')

    mask_5d = rearrange(mask_video, 'f c h w -> 1 c f h w')
    mask_81 = F.interpolate(mask_5d, size=(81, H_out, W_out), mode='nearest')

    # VAE encode: [0,1] → [-1,1] → latent
    rendered_81 = rendered_81 * 2.0 - 1.0
    z_rendered = vae.encode_to_latent(
        rendered_81.to(dtype=torch.bfloat16, device=device)
    ).to(device)  # (1, 16, 21, 60, 104)

    # Mask to latent resolution
    mask_latent = F.interpolate(mask_81, size=(21, 60, 104), mode='nearest').to(device)

    mask_latent = rearrange(mask_latent, 'b c f h w -> b f c h w')  # (1, 21, 1, 60, 104)
    z_with_mask = torch.cat([z_rendered, mask_latent], dim=2)        # (1, 21, 17, 60, 104)
    z_with_mask = z_with_mask.permute(0, 2, 1, 3, 4)                # (1, 17, 21, 60, 104)
    return z_with_mask.squeeze(0).to(torch.bfloat16)                 # (17, 21, 60, 104)


# ============================================================================
# Main
# ============================================================================
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Path to CausVid MixKit LMDB")
    parser.add_argument("--synccam_path", type=str, required=True,
                        help="Path to SynCamVideo dataset root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save per-sample .pt files")
    parser.add_argument("--pairs_filename", type=str, default="final_2_views_2.npy")

    # Sharding
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)

    # Model paths
    parser.add_argument("--cut3r_root", type=str, default=None,
                        help="CUT3R checkout (defaults to ./CUT3R or $MVF_CUT3R_ROOT)")
    parser.add_argument("--cut3r_path", type=str,
                        default="CUT3R/src/cut3r_512_dpt_4_64.pth",
                        help="CUT3R checkpoint")
    parser.add_argument("--cut3r_size", type=int, default=512)
    parser.add_argument("--focal_ratio", type=float, default=24.0 / 23.76)

    # Options
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--skip_existing", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Open LMDB ──
    env = lmdb.open(args.lmdb_path, readonly=True,
                    lock=False, readahead=False, meminit=False)
    latents_shape = get_array_shape_from_lmdb(env, 'latents')
    num_total = latents_shape[0]
    print(f"LMDB: {num_total} samples, latent shape per row: {latents_shape[1:]}")

    # ── Compute shard indices ──
    all_indices = list(range(args.shard_idx, num_total, args.num_shards))
    print(f"Shard {args.shard_idx}/{args.num_shards}: {len(all_indices)} samples")

    # ── Scan SynCamVideo scenes ──
    scenes = scan_synccam_scenes(args.synccam_path, args.pairs_filename)
    assert len(scenes) > 0, f"No valid scenes found in {args.synccam_path}"

    # ── Load models ──
    print("Loading CUT3R...")
    cut3r_model = load_cut3r(args.cut3r_path, device, root=args.cut3r_root)

    print("Loading VAE...")
    vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16)

    print("Loading T5 text encoder...")
    text_encoder = WanTextEncoder().to(device=device)

    H_out, W_out = 480, 832

    # ── Process samples ──
    for idx in tqdm(all_indices, desc=f"Shard {args.shard_idx}"):
        out_path = os.path.join(args.output_dir, f"{idx:06d}.pt")
        if args.skip_existing and os.path.exists(out_path):
            continue

        try:
            # 1. Read from LMDB
            latent_np = retrieve_row_from_lmdb(
                env, "latents", np.float16, idx, shape=latents_shape[1:])
            prompt_text = retrieve_row_from_lmdb(env, "prompts", str, idx)

            # latent: (F, C, H, W) = (21, 16, 60, 104)
            source_latent = torch.tensor(latent_np, dtype=torch.bfloat16).unsqueeze(0).to(device)
            # → (1, 21, 16, 60, 104) = (B, F, C, H, W)

            # 2. Decode latent → pixels
            source_pixels = vae.decode_to_pixel(source_latent)  # (1, T_pix, 3, H, W)
            source_pixels = (source_pixels * 0.5 + 0.5).clamp(0, 1)
            vae.model.clear_cache()

            # Subsample to 21 frames for CUT3R (every 4th pixel frame)
            source_frames = source_pixels[0, ::4]  # (21, 3, 480, 832)

            # 3. Encode prompt → T5
            prompt_dict = text_encoder(text_prompts=[prompt_text])
            prompt_emb = prompt_dict["prompt_embeds"][0].cpu()  # (512, 4096)

            # 4. Random scene + camera pair
            scene = random.choice(scenes)
            pair = scene['pairs'][random.randint(0, len(scene['pairs']) - 1)]
            src_cam_idx, tgt_cam_idx = int(pair[0]), int(pair[1])
            c2w_src, c2w_tgt = load_camera_pair(
                scene['cam_path'], src_cam_idx, tgt_cam_idx)

            # 5. Camera embedding (2, 21, 12)
            camera_emb = compute_camera_embedding(c2w_src, c2w_tgt)

            # 6. Run CUT3R
            outputs, state_args, H_c, W_c = run_cut3r(
                source_frames, cut3r_model, args.cut3r_size, device)

            # 7. Render raymap → z_rendered
            rendered_frames, conf_frames = render_raymap(
                outputs, state_args, c2w_src, c2w_tgt,
                args.focal_ratio, H_c, W_c, H_out, W_out,
                device, cut3r_model)

            z_rendered = postprocess_to_z_rendered(
                rendered_frames, conf_frames, H_out, W_out, vae, device)
            vae.model.clear_cache()

            # 8. Save .pt
            torch.save({
                'latents': source_latent.squeeze(0).cpu().half(),  # (21, 16, 60, 104)
                'prompt_emb': prompt_emb.half(),                   # (512, 4096)
                'camera': camera_emb.half(),                       # (2, 21, 12)
                'z_rendered': z_rendered.cpu().half(),             # (17, 21, 60, 104)
                'c2w_cond': c2w_src,                               # (21, 4, 4) float32
                'c2w_tgt': c2w_tgt,                                # (21, 4, 4) float32
                'prompt_text': prompt_text,
                'cam_scene': os.path.basename(scene['scene_dir']),
                'cam_pair': (src_cam_idx, tgt_cam_idx),
            }, out_path)

        except Exception as e:
            print(f"Error on sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Shard {args.shard_idx} done.")


if __name__ == "__main__":
    main()