"""
CUT3R geometric-prior helpers.

Builds the 4D-grounded conditioning signal: run CUT3R over decoded frames of the
previously generated views, query the accumulated state from the target camera's
raymap, and turn the render plus its confidence map into the conditioning tensor
the student consumes.
"""

from typing import List, Optional
import torch
from tqdm import tqdm
import os
import tempfile
import sys
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanMVBidirDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from PIL import Image
from utils.memory import cpu
import time


# ──────────────────────────────────────────────────────────────────────────────
# CUT3R rendering utilities (adapted from inference_syncammaster_cutt3r.py)
# ──────────────────────────────────────────────────────────────────────────────

def _render_raymap_frames(outputs, state_args, c2w_source, c2w_target,
                          focal_ratio, frame_indices, H_c, W_c, H_out, W_out,
                          device, cut3r_model, c2w_offset=0):
    """Render specific target frames via CUT3R raymap queries."""
    from dust3r.inference import inference_step as cut3r_inference_step
    from dust3r.utils.camera import pose_encoding_to_camera

    real_focal = focal_ratio * W_out
    focal = real_focal * (W_c / W_out)
    cx, cy = W_c // 2, H_c // 2
    K = torch.tensor([[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
                     device=device, dtype=torch.float32)
    K_inv = torch.inverse(K)

    i, j = torch.meshgrid(
        torch.arange(W_c, device=device, dtype=torch.float32),
        torch.arange(H_c, device=device, dtype=torch.float32), indexing='xy')
    grid = torch.stack([i, j, torch.ones_like(i)], dim=-1)
    grid_flat = (K_inv @ grid.reshape(-1, 3).T)
    grid_hom = torch.vstack([grid_flat, torch.ones(1, grid_flat.shape[1], device=device)])

    rendered_frames, conf_frames = [], []
    for t in frame_indices:
        cut3r_c2w_v0_t = pose_encoding_to_camera(
            outputs['pred'][t]['camera_pose'].clone()).squeeze(0).to(device)
        rel = (torch.inverse(c2w_source[t - c2w_offset].to(device).float()) @
               c2w_target[t - c2w_offset].to(device).float()).float()
        cut3r_c2w_tgt_t = (cut3r_c2w_v0_t @ rel).float()

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
            'camera_pose': torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            'img_mask': torch.tensor(False).unsqueeze(0),
            'ray_mask': torch.tensor(True).unsqueeze(0),
            'update': torch.tensor(False).unsqueeze(0),
            'reset': torch.tensor(False).unsqueeze(0),
        }
        query_output = cut3r_inference_step(
            view_query, state_args[t + 1], cut3r_model, device)
        pred = query_output['pred']
        rgb = (pred['rgb'].squeeze(0) + 1.0) / 2.0
        rendered_frames.append(rgb.permute(2, 0, 1))
        conf_frames.append(pred['conf_self'].squeeze(0).unsqueeze(0))

    return rendered_frames, conf_frames
def _run_cut3r_on_frames(pixel_frames_01, cut3r_model, cut3r_size, device):
    """
    Run CUT3R on pixel frames to build 3D state.
    Args:
        pixel_frames_01: (N, 3, H, W) in [0,1]
    Returns:
        outputs, state_args, frames_resized, H_c, W_c
    """
    from dust3r.inference import inference as cut3r_inference
    from dust3r.utils.image import load_images as cut3r_load_images


    H_out, W_out = pixel_frames_01.shape[2], pixel_frames_01.shape[3]
    scale = cut3r_size / max(H_out, W_out)
    H_c = int(H_out * scale) // 16 * 16
    W_c = int(W_out * scale) // 16 * 16
    frames_resized = F.interpolate(
        pixel_frames_01, size=(H_c, W_c), mode='bilinear', align_corners=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(frames_resized.shape[0]):
            frame = (frames_resized[i].float().permute(1, 2, 0).cpu().numpy() * 255
                     ).astype(np.uint8)
            path = os.path.join(tmpdir, f'{i:04d}.png')
            Image.fromarray(frame).save(path)
            paths.append(path)
        images = cut3r_load_images(paths, size=cut3r_size)

    views = []
    for i, image in enumerate(images):
        views.append({
            'img': image['img'],
            'ray_map': torch.full(
                (image['img'].shape[0], 6, image['img'].shape[-2], image['img'].shape[-1]),
                torch.nan),
            'true_shape': torch.from_numpy(image['true_shape']),
            'idx': i, 'instance': str(i),
            'camera_pose': torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            'img_mask': torch.tensor(True).unsqueeze(0),
            'ray_mask': torch.tensor(False).unsqueeze(0),
            'update': torch.tensor(True).unsqueeze(0),
            'reset': torch.tensor(False).unsqueeze(0),
        })
    outputs, state_args = cut3r_inference(views, cut3r_model, device)
    return outputs, state_args, frames_resized, H_c, W_c


def _feed_frames_to_cut3r_state(pixel_frames_01, outputs, state_args,
                                cut3r_model, cut3r_size, device):
    """
    Feed new pixel frames into existing CUT3R state (accumulate mode, raymap only).
    Returns updated outputs, state_args.
    """
    from dust3r.inference import inference_step as cut3r_inference_step
    from dust3r.utils.image import load_images as cut3r_load_images

    H_out, W_out = pixel_frames_01.shape[2], pixel_frames_01.shape[3]
    scale = cut3r_size / max(H_out, W_out)
    H_c = int(H_out * scale) // 16 * 16
    W_c = int(W_out * scale) // 16 * 16
    frames_resized = F.interpolate(
        pixel_frames_01, size=(H_c, W_c), mode='bilinear', align_corners=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(frames_resized.shape[0]):
            frame = (frames_resized[i].float().permute(1, 2, 0).cpu().numpy() * 255
                     ).astype(np.uint8)
            path = os.path.join(tmpdir, f'{i:04d}.png')
            Image.fromarray(frame).save(path)
            paths.append(path)
        images = cut3r_load_images(paths, size=cut3r_size)

    current_idx = len(outputs['pred'])
    last_state = state_args[-1]

    for i, image in enumerate(images):
        view = {
            'img': image['img'],
            'ray_map': torch.full(
                (image['img'].shape[0], 6, image['img'].shape[-2], image['img'].shape[-1]),
                torch.nan),
            'true_shape': torch.from_numpy(image['true_shape']),
            'idx': current_idx + i,
            'instance': str(current_idx + i),
            'camera_pose': torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
            'img_mask': torch.tensor(True).unsqueeze(0),
            'ray_mask': torch.tensor(False).unsqueeze(0),
            'update': torch.tensor(True).unsqueeze(0),
            'reset': torch.tensor(False).unsqueeze(0),
        }
        step_output = cut3r_inference_step(view, last_state, cut3r_model, device)
        outputs['pred'].append(step_output['pred'])
        last_state = step_output['state']
        state_args.append(last_state)

    return outputs, state_args


def _postprocess_rendered_to_z(rendered_frames, mask_frames, vae, H_out, W_out, device):
    """
    rendered pixel frames + masks → z_rendered latent [1, F, 17, H_lat, W_lat].
    """
    rendered_video = torch.stack(rendered_frames)  # (F, 3, H_c, W_c)
    mask_video = torch.stack(mask_frames)           # (F, 1, H_c, W_c)

    rendered_video = F.interpolate(
        rendered_video, size=(H_out, W_out), mode='bilinear', align_corners=False)
    mask_video = F.interpolate(mask_video, size=(H_out, W_out), mode='nearest')

    # 21 → 81 pixel frames for VAE temporal compression
    rendered_5d = rearrange(rendered_video, 'f c h w -> 1 c f h w')
    rendered_81 = F.interpolate(rendered_5d, size=(81, H_out, W_out), mode='nearest')
    mask_5d = rearrange(mask_video, 'f c h w -> 1 c f h w')
    mask_81 = F.interpolate(mask_5d, size=(81, H_out, W_out), mode='nearest')

    # VAE encode: [0,1] → [-1,1] → latent
    rendered_81 = rendered_81 * 2.0 - 1.0
    z_rendered = vae.encode_to_latent(
        rendered_81.to(dtype=torch.bfloat16, device=device)
    ).to(device)  # (1, 21, 16, 60, 104) — adjust if your VAE outputs (1, 16, 21, ...)

    # # Ensure (B, F, C, H, W)
    if z_rendered.dim() == 5 and z_rendered.shape[2] == 16 and z_rendered.shape[1] != 21:
        z_rendered = rearrange(z_rendered, 'b c f h w -> b f c h w')

    # Downsample mask to latent spatial resolution
    mask_latent = F.interpolate(mask_81, size=(21, 60, 104), mode='nearest').to(device)
    mask_latent = rearrange(mask_latent, 'b c f h w -> b f c h w')  # (1, 21, 1, 60, 104)

    # Concatenate: (1, 21, 17, 60, 104) — [B, F, C, H, W]
    z_rendered = torch.cat([z_rendered, mask_latent], dim=2).permute(0,2,1,3,4)
    return z_rendered
