import argparse
import sys
import time

import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from diffusers.utils import export_to_video

from pipeline import MVForcingInferencePipeline
from utils.dataset_mv import TextCameraDataset
from utils.misc import set_seed, load_mixed_state_dict
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller
import numpy as np
import random
import json
from utils.dataset_mixkit_lmdb import Camera

from utils.paths import load_cut3r, set_wan_root

def save_views(video, num_views, output_folder, method_name, prompt_idx, prompt_text="", camera_config=None, fps=16):
    """
    Save per-view videos and frames + merged side-by-side + metadata.

    video: [num_views, T_pix, 3, H, W] in [0,1]

    Structure:
      output_folder/method_name/prompt_XXX/view_N/video.mp4
      output_folder/method_name/prompt_XXX/view_N/frames/NNNNNN.png
      output_folder/method_name/prompt_XXX/merged.mp4
      output_folder/method_name/prompt_XXX/metadata.json
    """
    import json
    from PIL import Image

    prompt_dir = os.path.join(output_folder, method_name, f"prompt_{prompt_idx:04d}")
    os.makedirs(prompt_dir, exist_ok=True)

    for v in range(num_views):
        view_dir = os.path.join(prompt_dir, f"view_{v}")
        frames_dir = os.path.join(view_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        view_vid = video[v]  # [T_pix, 3, H, W]
        # Save individual frames
        for t in range(view_vid.shape[0]):
            frame_np = (view_vid[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(frame_np).save(os.path.join(frames_dir, f"{t:06d}.png"))

        # Save video
        vid_np = view_vid.permute(0, 2, 3, 1).cpu().numpy()
        export_to_video(vid_np, os.path.join(view_dir, "video.mp4"), fps=fps)

    # Save merged side-by-side
    merged = torch.cat([video[v] for v in range(num_views)], dim=3)  # [T, 3, H, W*num_views]
    merged_np = merged.permute(0, 2, 3, 1).cpu().numpy()
    export_to_video(merged_np, os.path.join(prompt_dir, "merged.mp4"), fps=fps)

    # Save metadata
    meta = {
        "prompt": prompt_text,
        "method": method_name,
        "num_views": num_views,
        "num_pixel_frames": int(video.shape[1]),
        "camera_config": camera_config,
    }
    with open(os.path.join(prompt_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

def parse_matrix(matrix_str):
    rows = matrix_str.strip().split('] [')
    matrix = []
    for row in rows:
        row = row.replace('[', '').replace(']', '')
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

def build_camera_chain(camera_json_path, cam_indices):
    """Build per-transition camera embeddings and c2w from a chain of camera indices."""
    with open(camera_json_path, 'r') as f:
        cam_data = json.load(f)

    cam_frame_indices = list(range(81))[::4]

    def load_trajectory(view_idx):
        traj = [parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"])
                for idx in cam_frame_indices]
        traj = np.stack(traj).transpose(0, 2, 1)
        c2ws = []
        for c2w in traj:
            c2w = c2w[:, [1, 2, 0, 3]]
            c2w[:3, 1] *= -1.
            c2w[:3, 3] /= 200
            c2ws.append(c2w)
        return c2ws

    trajectories = {}
    for idx in set(cam_indices):
        trajectories[idx] = load_trajectory(idx)

    camera_embeddings_list = []
    c2w_matrices = []

    for i in range(len(cam_indices) - 1):
        cond_c2ws = trajectories[cam_indices[i]]
        tgt_c2ws = trajectories[cam_indices[i + 1]]

        cond_cams = [Camera(c) for c in cond_c2ws]
        tgt_cams = [Camera(c) for c in tgt_c2ws]

        relative_poses = []
        for f in range(len(cond_cams)):
            target_cam_c2w = np.eye(4, dtype=np.float32)
            abs2rel = target_cam_c2w @ cond_cams[f].w2c_mat
            ret_poses = [target_cam_c2w, abs2rel @ tgt_cams[f].c2w_mat]
            relative_poses.append(
                torch.as_tensor(np.array(ret_poses, dtype=np.float32))[:, :3, :])

        pose_embedding = torch.stack(relative_poses, dim=1)
        camera_embedding = rearrange(
            pose_embedding, 'v f c d -> v f (c d)').to(torch.bfloat16)

        c2w_cond = torch.tensor(np.stack(cond_c2ws), dtype=torch.float32)
        c2w_tgt = torch.tensor(np.stack(tgt_c2ws), dtype=torch.float32)

        camera_embeddings_list.append(camera_embedding)
        c2w_matrices.append((c2w_cond, c2w_tgt))

    return camera_embeddings_list, c2w_matrices


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True)
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="Path to MV-Forcing student checkpoint")
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--num_output_frames", type=int, default=21)
parser.add_argument("--use_ema", action="store_true")
parser.add_argument("--seed", type=int, default=24)
parser.add_argument("--num_samples", type=int, default=1)

parser.add_argument("--cam_type", type=str, nargs='+', default=['az', 'el', 'dis'],
                    choices=['az', 'el', 'dis'],
                    help="Camera types to evaluate. Defaults to all three.")
parser.add_argument("--long_rolling", action="store_true",
                    help="Memory-efficient rolling generation for very long sequences; "
                         "requires --cam_indices")
parser.add_argument("--cam_indices", type=int, nargs='+', default=None,
                    help="Explicit SynCamMaster camera chain for --long_rolling, "
                         "e.g. --cam_indices 3 1 7 (cam03 -> cam01 -> cam07)")
parser.add_argument("--camera_json_path", type=str,
                    default="SynCamMaster/example_test_data/cameras/camera_extrinsics.json",
                    help="SynCamMaster camera rig JSON")
parser.add_argument("--wan_root", type=str, default=None,
                    help="Directory holding Wan2.1-T2V-1.3B (defaults to ./wan_models)")
parser.add_argument("--cut3r_root", type=str, default=None,
                    help="CUT3R checkout (defaults to ./CUT3R)")

# CUT3R args
parser.add_argument("--cutt3r_method", type=str, default="raymap", choices=[None, "raymap"],
                    help="CUT3R rendering method. None to disable.")
parser.add_argument("--cutt3r_model_path", type=str,
                    default="CUT3R/src/cut3r_512_dpt_4_64.pth",
                    help="Path to CUT3R checkpoint")
parser.add_argument("--cutt3r_accumulate", action="store_true",
                    help="Accumulate v1 blocks into CUT3R state (raymap only)")
parser.add_argument("--cut3r_size", type=int, default=512)
parser.add_argument("--focal_ratio", type=float, default=24.0 / 23.76)
parser.add_argument("--total_frames", type=int, default=21,
                    help="Total latent frames per view (21 = 1 chunk, 42 = 2 chunks, etc.)")
parser.add_argument("--num_views", type=int, default=2)
parser.add_argument("--num_overlap_frames", type=int, default=3)
parser.add_argument("--accumulate_views", action="store_true",
                    help="Accumulate CUT3R state across all previous views (richer geometry for later views)")
parser.add_argument("--method_name", type=str, default="mv_forcing",
                    help="Name for output folder structure")
parser.add_argument("--one_by_one", action="store_true")
parser.add_argument("--prompt_idx", type=int, default=0)
parser.add_argument("--local_attn_size", type=int, default=-1,
                    help="Rolling window size in frames (-1 = no rolling, 21 = training length)")
parser.add_argument("--sink_size", type=int, default=0,
                    help="Number of sink frames to preserve (0 = disabled, 12 = Deep Sink)")


args = parser.parse_args()
set_wan_root(args.wan_root)

# Initialize distributed (optional)
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    set_seed(args.seed)

print(f'Free VRAM {get_cuda_free_memory_gb(gpu)} GB')
low_memory = get_cuda_free_memory_gb(gpu) < 40

torch.set_grad_enabled(False)

# Load config
config = OmegaConf.load(args.config_path)
default_config = OmegaConf.load("configs/base.yaml")
config = OmegaConf.merge(default_config, config)
# Inject CUT3R config into OmegaConf so pipeline.__init__ can read them
config.cutt3r_method = args.cutt3r_method
config.cutt3r_accumulate = args.cutt3r_accumulate
config.cut3r_size = args.cut3r_size
config.focal_ratio = args.focal_ratio
config.local_attn_size = args.local_attn_size
config.sink_size = args.sink_size

# ── Load CUT3R model (if needed) ────────────────────────────────────────
cut3r_model = None
if args.cutt3r_method is not None:
    cut3r_model = load_cut3r(args.cutt3r_model_path, device, root=args.cut3r_root)
    print(f"CUT3R model loaded ({args.cutt3r_method}, accumulate={args.cutt3r_accumulate})")

# Initialize pipeline
pipeline = MVForcingInferencePipeline(config, device=device, cutt3r_method=args.cutt3r_method)

student_sd = torch.load(args.checkpoint_path, map_location="cpu")#["generator_ema"]
key = 'generator_ema' if args.use_ema and 'generator_ema' in student_sd else 'generator'
if key in student_sd:
    pipeline.student.load_state_dict(student_sd[key])
else:
    pipeline.student.load_state_dict(student_sd)
print(f"Student weights loaded (key='{key}').")

# Move to device
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.cut3r_model = cut3r_model

if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
else:
    pipeline.text_encoder.to(device=gpu)
pipeline.student.to(device=gpu)
pipeline.vae.to(device=gpu)


for cam_type in args.cam_type:
    print(f"\n{'='*60}")
    print(f"Running cam_type={cam_type}")
    print(f"{'='*60}\n")

    args_dict = vars(args).copy()
    args_dict["cam_type"] = cam_type
    args_for_dataset = argparse.Namespace(**args_dict)
    dataset = TextCameraDataset(
        args.data_path,
        os.path.join(args.data_path, "metadata.csv"),
        args_for_dataset,
        shuffle=True
    )
    print(f"Number of prompts: {len(dataset)}")

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

    if local_rank == 0:
        os.makedirs(args.output_folder, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()


    for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
        idx = batch_data['idx'].item()
        if args.one_by_one and idx != args.prompt_idx:
            continue
        if isinstance(batch_data, dict):
            batch = batch_data
        else:
            batch = batch_data[0]

        prompt = batch['text'][0]
        camera = batch["camera"].to(device)  # [B, V, F, D]
        B, V, F_cam, D_cam = camera.shape
        camera_bv = rearrange(camera, 'b v ... -> (b v) ...')  # [B*V, F, D]

        prompts = [prompt] * V  # Same prompt for both views

        # c2w matrices for CUT3R (if dataset provides them)
        c2w_source = batch.get('c2w_cond', None)
        c2w_target = batch.get('c2w_tgt', None)
        if c2w_source is not None:
            c2w_source = c2w_source[0].to(device)
        if c2w_target is not None:
            c2w_target = c2w_target[0].to(device)

        # Camera config for metadata
        cam_meta = {
            "cam_type": cam_type,
            "num_views": args.num_views,
        }

        if args.long_rolling:
            # ── Rolling generation for very long sequences ──
            assert args.cam_indices is not None and len(args.cam_indices) == args.num_views, \
                "--long_rolling requires --cam_indices with exactly --num_views entries"
            cam_indices = args.cam_indices
            print(f"Camera chain: {[f'cam{i:02d}' for i in cam_indices]}")

            camera_embeddings_list, c2w_matrices = build_camera_chain(
                args.camera_json_path, cam_indices)

            camera_embeddings_list = [
                emb[:, :1, :].expand(-1, args.total_frames, -1).to(device)
                for emb in camera_embeddings_list
            ]
            c2w_matrices = [
                (s[:1].expand(args.total_frames, -1, -1).to(device),
                 t[:1].expand(args.total_frames, -1, -1).to(device))
                for s, t in c2w_matrices
            ]

            video = pipeline.inference_rolling_chain(
                text_prompts=[prompt] * 2,
                camera_embeddings_list=camera_embeddings_list,
                c2w_matrices=c2w_matrices,
                num_views=args.num_views,
                total_frames=args.total_frames,
                accumulate_views=args.accumulate_views,
            )
        else:
            # ── Long multi-view generation ──
            c2w_matrices = []
            for v in range(1, args.num_views):
                c2w_s = c2w_source[:1].expand(args.total_frames, -1, -1) if c2w_source is not None else None
                c2w_t = c2w_target[:1].expand(args.total_frames, -1, -1) if c2w_target is not None else None
                c2w_matrices.append((c2w_s, c2w_t))
            if args.local_attn_size != -1:
                if camera_bv.shape[1] < args.total_frames:
                    # Camera is [V, 21, 12] — repeat to fill total_frames
                    camera_bv_expanded = camera_bv[:, :1, :].expand(-1, args.total_frames, -1)
                    print(camera_bv_expanded.shape)
                else:
                    camera_bv_expanded = camera_bv

                video = pipeline.inference_long_rolling(
                    text_prompts=[prompt] * args.num_views,
                    camera_embeddings=camera_bv_expanded,
                    num_views=args.num_views,
                    total_frames=args.total_frames,
                    negative_prompt=getattr(config, "negative_prompt", ""),
                    c2w_matrices=c2w_matrices,
                    accumulate_views=args.accumulate_views,
                )

            else:
                video = pipeline.inference_long(
                    text_prompts=[prompt] * args.num_views,
                    camera_embeddings=camera_bv,
                    num_views=args.num_views,
                    total_frames=args.total_frames,
                    chunk_size=args.num_output_frames,
                    num_overlap_frames=args.num_overlap_frames,
                    negative_prompt=getattr(config, "negative_prompt", ""),
                    c2w_matrices=c2w_matrices,
                    accumulate_views=args.accumulate_views,
                )

        method_with_cam = f"{args.method_name}/{cam_type}"

        save_views(video, args.num_views, args.output_folder, method_with_cam, idx,
                   prompt_text=prompt, camera_config=cam_meta)

        pipeline.vae.model.clear_cache()

        if local_rank == 0:
            print(f"Saved: {args.method_name}/prompt_{idx:04d}")
