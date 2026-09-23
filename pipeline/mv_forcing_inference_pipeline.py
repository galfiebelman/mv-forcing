"""
MV-Forcing inference pipeline.

View 0: the student denoises both view slots jointly from noise and keeps slot 0.
View k: the student generates it autoregressively via few-step temporal
        self-forcing, conditioned on the clean view k-1 and on the CUT3R
        geometric prior rendered from the accumulated 4D state.
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

class MVForcingInferencePipeline(torch.nn.Module):
    """
    Two-stage inference:
      Stage 1: Teacher (bidirectional WanMVModel) generates view 0 in 50 steps.
      Stage 2: Student (CausalWanMVBidirModel) generates view 1 via temporal
               self-forcing, with the preceding view as clean conditioning.
    """

    def __init__(
            self,
            args,
            device,
            student=None,
            text_encoder=None,
            vae=None,
            cutt3r_method=None,
    ):
        super().__init__()

        # Student: CausalWanMVBidirModel (MV-Forcing checkpoint)
        use_cut3r = True if cutt3r_method is not None else False
        local_attn_size = getattr(args, "local_attn_size", -1)
        sink_size = getattr(args, "sink_size", 0)
        self.student = WanMVBidirDiffusionWrapper(
            use_cut3r=use_cut3r,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        ) if student is None else student
        self.local_attn_size = local_attn_size

        # Text encoder & VAE
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.device = device
        self.args = args

        self.cut3r_model = None
        self.cutt3r_method = getattr(args, "cutt3r_method", None)
        self.cutt3r_accumulate = getattr(args, "cutt3r_accumulate", False)
        self.cut3r_size = getattr(args, "cut3r_size", 512)
        self.focal_ratio = getattr(args, "focal_ratio", 24.0 / 23.76)

        self.num_train_timesteps = args.num_train_timestep
        self.guidance_scale = getattr(args, "guidance_scale", 5.0)

        # Student denoising schedule
        self.student_scheduler = self.student.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        if getattr(args, "warp_denoising_step", False):
            timesteps = torch.cat((
                self.student_scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)
            )).to(device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Student temporal params
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.context_noise = getattr(args, "context_noise", 0)

    @torch.no_grad()
    def inference(
            self,
            noise: torch.Tensor,
            text_prompts: List[str],
            camera_embeddings: torch.Tensor,
            negative_prompt: str = "",
            return_latents: bool = False,
            num_output_frames: int = 21,
            c2w_source: Optional[torch.Tensor] = None,
            c2w_target: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            noise: [2, F, C, H, W] — noise for both views
            text_prompts: list of prompts (length = 2 for 2 views, same prompt)
            camera_embeddings: [2, F, 12] — camera poses for both views
            negative_prompt: negative prompt for CFG
        Returns:
            video: [2, T, 3, H_pixel, W_pixel] decoded pixel video
        """
        num_views = noise.shape[0]
        assert num_views == 2, "Currently supports 2 views only"
        num_frames = noise.shape[1]

        # Encode text
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["cam_emb"] = camera_embeddings

        unconditional_dict = self.text_encoder(
            text_prompts=[negative_prompt] * len(text_prompts))
        unconditional_dict["cam_emb"] = camera_embeddings

        # Stage 1: student co-denoises both slots from noise, keep view 0
        print("Stage 1: student generating view 0 (joint denoising)...")
        view0_latent = self._student_generate_view0(
            noise=noise,
            conditional_dict=conditional_dict,
        )[0:1]

        z_rendered_full = None
        cut3r_state = None
        use_cutt3r = (self.cutt3r_method is not None and self.cut3r_model is not None)
        if use_cutt3r:
            assert c2w_source is not None and c2w_target is not None, \
                "c2w_source and c2w_target required for CUT3R"
            print(f"CUT3R rendering ({self.cutt3r_method}, accumulate={self.cutt3r_accumulate})...")

            # Decode view 0 → pixels [0,1]
            view0_pixels = self.vae.decode_to_pixel(view0_latent)
            view0_pixels = (view0_pixels * 0.5 + 0.5).clamp(0, 1)[0]  # [F, 3, H, W]

            # Build CUT3R 3D state from view 0
            outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
                view0_pixels, self.cut3r_model, self.cut3r_size, self.device)
            H_out, W_out = view0_pixels.shape[2], view0_pixels.shape[3]

            if self.cutt3r_accumulate:
                cut3r_state = {
                    "outputs": outputs, "state_args": state_args,
                    "frames_resized": frames_resized,
                    "H_c": H_c, "W_c": W_c, "H_out": H_out, "W_out": W_out,
                    "c2w_source": c2w_source, "c2w_target": c2w_target,
                }
                z_rendered_full = self._render_all_frames(cut3r_state)
            else:
                # One-shot rendering
                all_indices = list(range(num_frames))
                rendered, conf = _render_raymap_frames(
                    outputs, state_args, c2w_source, c2w_target,
                    self.focal_ratio, all_indices,
                    H_c, W_c, H_out, W_out, self.device, self.cut3r_model)
                conf_video = torch.stack(conf)
                masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)
                         for c in conf]
                z_rendered_full = _postprocess_rendered_to_z(
                    rendered, masks, self.vae, H_out, W_out, self.device)

        print("Stage 2: Student generating view 1 (temporal self-forcing)...")
        view1_latent = self._student_generate_view1(
            noise=noise[1:2],  # [1, F, C, H, W]
            view0_latent=view0_latent,  # [1, F, C, H, W]
            conditional_dict={
                "prompt_embeds": conditional_dict["prompt_embeds"],  # [2, L, D]
                "cam_emb": camera_embeddings,  # [2, F, 12]
            },
            z_rendered_full=z_rendered_full.to(dtype=noise.dtype, device=noise.device) if z_rendered_full is not None else None,
            cut3r_state=cut3r_state,
        )  # [1, F, C, H, W]

        all_latents = torch.cat([view0_latent, view1_latent], dim=0)  # [2, F, C, H, W]
        video = self.vae.decode_to_pixel(all_latents)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, all_latents
        return video

    def _student_generate_view0(
            self,
            noise: torch.Tensor,
            conditional_dict: dict,
            z_rendered_full=None,
            start_latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Few-step temporal self-forcing for view 1, conditioned on the preceding view.

        Both views are stacked as batch=2:
          slot 0 = preceding view (clean)
          slot 1 = student view 1 (being denoised)

        Args:
            noise: [1, F, C, H, W] — initial noise for view 1
            view0_latent: [1, F, C, H, W] — clean preceding view
            conditional_dict: must have prompt_embeds [2, L, D] and cam_emb [2, F, 12]
        Returns:
            view1_latent: [1, F, C, H, W]
        """
        num_frames = noise.shape[1]
        num_overlap = start_latents.shape[1] if start_latents is not None else 0
        num_denoising_frames = num_frames - num_overlap
        assert num_denoising_frames % self.num_frame_per_block == 0
        num_blocks = num_denoising_frames // self.num_frame_per_block

        # Initialize KV cache (batch=2: one for each view slot)
        kv_cache_size = num_frames * self.frame_seq_length
        kv_cache = self._init_kv_cache(batch_size=2, kv_cache_size=kv_cache_size,
                                       dtype=noise.dtype, device=noise.device)
        crossattn_cache = self._init_crossattn_cache(batch_size=2,
                                                     dtype=noise.dtype, device=noise.device)

        output = torch.zeros_like(noise)  # [1, F, C, H, W]
        current_start_frame = 0
        camera_embeddings = conditional_dict["cam_emb"]
        z_rendered_both = None
        if start_latents is not None:
            assert num_overlap % self.num_frame_per_block == 0
            num_overlap_blocks = num_overlap // self.num_frame_per_block
            for ob in range(num_overlap_blocks):
                fs = ob * self.num_frame_per_block
                fe = fs + self.num_frame_per_block
                cache_input = start_latents[:, fs:fe]  # [2, Fb, C, H, W]
                output[:, fs:fe] = start_latents[:, fs:fe]
                timestep = torch.zeros(
                    [2, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64)
                conditional_dict["cam_emb"] = camera_embeddings[:, fs:fe]
                conditional_dict["z_rendered"] = None
                self.student(
                    noisy_image_or_video=cache_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        for block_idx in tqdm(range(num_blocks), desc="Student view 1"):
            frame_start = current_start_frame
            frame_end = current_start_frame + self.num_frame_per_block

            # Slot 0: preceding view (clean) for this block
            noisy_view0 = noise[0:1, frame_start:frame_end]  # [1, Fb, C, H, W]

            # Slot 1: noisy view 1 for this block
            noisy_view1 = noise[1:2, frame_start:frame_end]  # [1, Fb, C, H, W]
            conditional_dict["cam_emb"] = camera_embeddings[:,frame_start:frame_end]
            conditional_dict["z_rendered"] = None

            # Denoising loop (all steps, no random exit)
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64
                ) * current_timestep

                # Stack [clean_view0, noisy_view1]
                model_input = torch.cat([noisy_view0, noisy_view1], dim=0)  # [2, Fb, C, H, W]

                _, denoised_pred = self.student(
                    noisy_image_or_video=model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                view0_denoised = denoised_pred[0:1]  # [1, Fb, C, H, W]
                view1_denoised = denoised_pred[1:2]  # [1, Fb, C, H, W]

                # If not last step, re-noise for next step
                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_view0 = self.student_scheduler.add_noise(
                        view0_denoised.flatten(0, 1),
                        torch.randn_like(view0_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

                    noisy_view1 = self.student_scheduler.add_noise(
                        view1_denoised.flatten(0, 1),
                        torch.randn_like(view1_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            # Record view 1 output
            output[0:1, frame_start:frame_end] = view0_denoised
            output[1:2, frame_start:frame_end] = view1_denoised
            # Update KV cache: re-run with context noise for both slots
            ctx_t = torch.ones(
                [2, self.num_frame_per_block],
                device=noise.device, dtype=torch.int64
            ) * self.context_noise

            cache_input = torch.cat([view0_denoised, view1_denoised], dim=0)

            self.student(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += self.num_frame_per_block

        return output



    def _student_generate_view1(
            self,
            noise: torch.Tensor,
            view0_latent: torch.Tensor,
            conditional_dict: dict,
            z_rendered_full=None, cut3r_state=None,
            start_latents: Optional[torch.Tensor] = None,
            kv_cache=None, crossattn_cache=None,
    ) -> torch.Tensor:
        """
        Few-step temporal self-forcing for view 1, conditioned on the preceding view.

        Both views are stacked as batch=2:
          slot 0 = preceding view (clean)
          slot 1 = student view 1 (being denoised)

        Args:
            noise: [1, F, C, H, W] — initial noise for view 1
            view0_latent: [1, F, C, H, W] — clean preceding view
            conditional_dict: must have prompt_embeds [2, L, D] and cam_emb [2, F, 12]
        Returns:
            view1_latent: [1, F, C, H, W]
        """
        num_frames = noise.shape[1]
        num_overlap = start_latents.shape[1] if start_latents is not None else 0
        num_denoising_frames = num_frames - num_overlap
        assert num_denoising_frames % self.num_frame_per_block == 0
        num_blocks = num_denoising_frames // self.num_frame_per_block
        # Initialize KV cache (batch=2: one for each view slot)
        if kv_cache is None:
            kv_cache_size = num_frames * self.frame_seq_length
            kv_cache = self._init_kv_cache(batch_size=2, kv_cache_size=kv_cache_size,
                                           dtype=noise.dtype, device=noise.device)
        if crossattn_cache is None:
            crossattn_cache = self._init_crossattn_cache(batch_size=2,
                                                         dtype=noise.dtype, device=noise.device)

        output_view1 = torch.zeros_like(noise[0:1])  # [1, F, C, H, W]
        current_start_frame = 0
        camera_embeddings = conditional_dict["cam_emb"]
        if z_rendered_full is not None:
            z_rendered_both = torch.cat([
                torch.zeros_like(z_rendered_full), z_rendered_full
            ], dim=0)  # [2, F, 17, H_lat, W_lat]
        else:
            z_rendered_both = None

        # Cache overlap frames at timestep 0 if provided
        if start_latents is not None:
            num_overlap = start_latents.shape[1]
            assert num_overlap % self.num_frame_per_block == 0
            num_overlap_blocks = num_overlap // self.num_frame_per_block
            for ob in range(num_overlap_blocks):
                fs = ob * self.num_frame_per_block
                fe = fs + self.num_frame_per_block
                overlap_v0 = view0_latent[:, fs:fe]
                overlap_v1 = start_latents[:, fs:fe]
                output_view1[:, fs:fe] = start_latents[:, fs:fe]
                cache_input = torch.cat([overlap_v0, overlap_v1], dim=0)
                timestep = torch.zeros(
                    [2, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64)
                conditional_dict["cam_emb"] = camera_embeddings[:, fs:fe]
                if z_rendered_full is not None:
                    conditional_dict["z_rendered"] = z_rendered_both[:, :, fs:fe]
                else:
                    conditional_dict["z_rendered"] = None
                self.student(
                    noisy_image_or_video=cache_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        for block_idx in tqdm(range(num_blocks), desc="Student view 1"):
            frame_start = current_start_frame
            frame_end = current_start_frame + self.num_frame_per_block

            # Slot 0: preceding view (clean) for this block
            gt_view0_block = view0_latent[:, frame_start:frame_end]  # [1, Fb, C, H, W]

            # Slot 1: noisy view 1 for this block
            noisy_view1 = noise[:, frame_start:frame_end]  # [1, Fb, C, H, W]
            conditional_dict["cam_emb"] = camera_embeddings[:,frame_start:frame_end]
            if z_rendered_both is not None:
                conditional_dict["z_rendered"] = z_rendered_both[:, :,frame_start:frame_end]
            else:
                conditional_dict["z_rendered"] = None

            # Denoising loop (all steps, no random exit)
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64
                ) * current_timestep

                # Stack [clean_view0, noisy_view1]
                model_input = torch.cat([gt_view0_block, noisy_view1], dim=0)  # [2, Fb, C, H, W]

                _, denoised_pred = self.student(
                    noisy_image_or_video=model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                view1_denoised = denoised_pred[1:2]  # [1, Fb, C, H, W]

                # If not last step, re-noise for next step
                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_view1 = self.student_scheduler.add_noise(
                        view1_denoised.flatten(0, 1),
                        torch.randn_like(view1_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            # Record view 1 output
            output_view1[:, frame_start:frame_end] = view1_denoised

            # Update KV cache: re-run with context noise for both slots
            ctx_t = torch.ones(
                [2, self.num_frame_per_block],
                device=noise.device, dtype=torch.int64
            ) * self.context_noise

            cache_input = torch.cat([gt_view0_block, view1_denoised], dim=0)

            self.student(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += self.num_frame_per_block

        return output_view1

    def _render_all_frames(self, cut3r_state):
        """Re-render all 21 target frames from current CUT3R state → z_rendered."""
        num_frames = cut3r_state["c2w_source"].shape[0]
        rendered, conf = _render_raymap_frames(
            cut3r_state["outputs"], cut3r_state["state_args"],
            cut3r_state["c2w_source"], cut3r_state["c2w_target"],
            self.focal_ratio, list(range(num_frames)),
            cut3r_state["H_c"], cut3r_state["W_c"],
            cut3r_state["H_out"], cut3r_state["W_out"],
            self.device, self.cut3r_model)
        conf_video = torch.stack(conf)
        masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1) for c in conf]
        return _postprocess_rendered_to_z(
            rendered, masks, self.vae,
            cut3r_state["H_out"], cut3r_state["W_out"], self.device)

    def _generate_long_view0(
            self,
            total_frames: int,
            num_overlap_frames: int,
            noise_shape: tuple,
            conditional_dict: dict,
    ) -> tuple:
        """
        Generate long v0 using _student_generate_view0 (both_noisy) with overlap rollout.

        Args:
            total_frames: total latent frames to generate
            num_overlap_frames: overlap between chunks
            noise_shape: (2, 21, C, H, W) per chunk
            conditional_dict: text embeddings + cam_emb
        Returns:
            all_latents: [1, total_frames, C, H, W]
            all_pixels: [1, total_pixel_frames, 3, H_px, W_px]
        """
        chunk_frames = noise_shape[1]  # 21
        num_gen = chunk_frames - num_overlap_frames
        num_chunks = 1 + (total_frames - chunk_frames) // num_gen
        all_pixel_chunks = []
        all_latent_chunks = []
        start_latents = None  # [2, overlap, C, H, W]
        # VAE temporal compression: 4 pixel frames per latent frame
        pf_per_lf = 4
        camera_embeddings = conditional_dict["cam_emb"]
        for chunk_idx in tqdm(range(num_chunks), desc="Long v0 chunks"):
            noise = torch.randn(noise_shape, device=self.device, dtype=torch.bfloat16)
            conditional_dict["cam_emb"] = camera_embeddings
            output = self._student_generate_view0(
                noise=noise,
                conditional_dict=conditional_dict,
                start_latents=start_latents,
            )  # [2, F, C, H, W]

            v0_latents = output[0:1]  # [1, F, C, H, W]
            v0_pixels = self.vae.decode_to_pixel(v0_latents)
            v0_pixels = (v0_pixels * 0.5 + 0.5).clamp(0, 1)  # [1, T_pix, 3, H, W]

            if chunk_idx < num_chunks - 1:
                keep_latents = v0_latents[:, :num_gen]
                num_drop_pixels = pf_per_lf * (num_overlap_frames - 1) + 1
                keep_pixels = v0_pixels[:, :-num_drop_pixels]
                boundary_pixel = v0_pixels[:, -num_drop_pixels:-num_drop_pixels + 1]  # [1, 1, 3, H, W]
                boundary_latent = self.vae.encode_to_latent(
                    (boundary_pixel.transpose(1, 2) * 2.0 - 1.0).to(torch.bfloat16)
                ).to(torch.bfloat16)  # [1, 1, C, H, W]
                v0_overlap = torch.cat([
                    boundary_latent, v0_latents[:, -(num_overlap_frames - 1):]
                ], dim=1)  # [1, overlap, C, H, W]

                # Same for view1 side (slot 1)
                v1_latents = output[1:2]
                v1_pixels = self.vae.decode_to_pixel(v1_latents)
                v1_pixels = (v1_pixels * 0.5 + 0.5).clamp(0, 1)
                v1_boundary = v1_pixels[:, -num_drop_pixels:-num_drop_pixels + 1]
                v1_boundary_latent = self.vae.encode_to_latent(
                    (v1_boundary.transpose(1, 2) * 2.0 - 1.0).to(torch.bfloat16)
                ).to(torch.bfloat16)
                v1_overlap = torch.cat([
                    v1_boundary_latent, v1_latents[:, -(num_overlap_frames - 1):]
                ], dim=1)

                start_latents = torch.cat([v0_overlap, v1_overlap], dim=0)  # [2, overlap, C, H, W]
            else:
                keep_latents = v0_latents
                keep_pixels = v0_pixels

            all_latent_chunks.append(keep_latents)
            all_pixel_chunks.append(keep_pixels)

        all_latents = torch.cat(all_latent_chunks, dim=1)  # [1, total_frames, C, H, W]
        all_pixels = torch.cat(all_pixel_chunks, dim=1)    # [1, total_pixel_frames, 3, H, W]
        return all_latents, all_pixels

    def _compute_cut3r_per_chunk(
            self,
            prev_view_pixels: torch.Tensor,
            c2w_source: torch.Tensor,
            c2w_target: torch.Tensor,
            latent_chunk_size: int = 21,
            cut3r_state: dict = None,
            view_id=0,
    ) -> tuple:
        """
        Run CUT3R per temporal chunk on the previous view's pixels.

        Args:
            prev_view_pixels: [F, 3, H, W] in [0,1]
            c2w_source, c2w_target: [F, 4, 4]
            chunk_size: frames per chunk (21)
            cut3r_state: if provided, feed frames into existing state (accumulation).
                         If None, run CUT3R fresh (no accumulation).
        Returns:
            z_rendered_chunks: list of z_rendered [1, 17, chunk_size, H_lat, W_lat]
            cut3r_state: updated state (or None if no accumulation)
        """
        pf_per_lf = 4
        total_pixel_frames = prev_view_pixels.shape[0]
        total_latent_frames = c2w_source.shape[0]
        pixel_chunk_size = ((latent_chunk_size - 1) * pf_per_lf) +1 # 81
        num_chunks = (total_latent_frames + latent_chunk_size - 1) // latent_chunk_size
        z_rendered_chunks = []

        if cut3r_state is not None:
            # Accumulation: state already contains the source view frames.
            # Just render from existing state — don't feed again.
            outputs = cut3r_state["outputs"]
            state_args = cut3r_state["state_args"]
            H_c, W_c = cut3r_state["H_c"], cut3r_state["W_c"]
            H_out, W_out = cut3r_state["H_out"], cut3r_state["W_out"]
            # Source view frames are the last total_frames entries in the accumulated outputs
            source_start_idx = len(outputs['pred']) - total_pixel_frames

            for chunk_idx in tqdm(range(num_chunks), desc="CUT3R chunks (accumulated)"):
                pfs = chunk_idx * pixel_chunk_size
                pfe = min(pfs + pixel_chunk_size, total_pixel_frames)
                lfs = chunk_idx * latent_chunk_size
                lfe = min(lfs + latent_chunk_size, total_latent_frames)
                chunk_c2w_s_latent = c2w_source[lfs:lfe]  # [chunk_latent, 4, 4]
                chunk_c2w_t_latent = c2w_target[lfs:lfe]
                chunk_c2w_s = chunk_c2w_s_latent.repeat_interleave(pf_per_lf, dim=0)[:pfe - pfs]
                chunk_c2w_t = chunk_c2w_t_latent.repeat_interleave(pf_per_lf, dim=0)[:pfe - pfs]
                acc_indices = list(range(source_start_idx + pfs, source_start_idx + pfe))

                # Re-index outputs and state_args to 0-based so rendering
                # functions can use t for both outputs['pred'][t] and c2w[t]
                chunk_outputs = {
                    'pred': [outputs['pred'][i] for i in acc_indices],
                    'views': outputs.get('views', None),
                }
                # state_args[i+1] = state after frame i; prepend state before first frame
                chunk_state_args = (
                    [state_args[acc_indices[0]]] +
                    [state_args[i + 1] for i in acc_indices]
                )
                local_indices = list(range(pfe - pfs))

                rendered, conf = _render_raymap_frames(
                    chunk_outputs, chunk_state_args, chunk_c2w_s, chunk_c2w_t,
                    self.focal_ratio, local_indices,
                    H_c, W_c, H_out, W_out, self.device, self.cut3r_model)
                conf_video = torch.stack(conf)
                masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)
                         for c in conf]
                z_rendered = _postprocess_rendered_to_z(
                    rendered, masks, self.vae, H_out, W_out, self.device)
                z_rendered_chunks.append(z_rendered)

        else:
            # No accumulation: fresh CUT3R per chunk
            for chunk_idx in tqdm(range(num_chunks), desc="CUT3R chunks"):
                pfs = chunk_idx * pixel_chunk_size
                pfe = min(pfs + pixel_chunk_size, total_pixel_frames)
                lfs = chunk_idx * latent_chunk_size
                lfe = min(lfs + latent_chunk_size, total_latent_frames)

                chunk_pixels = prev_view_pixels[pfs:pfe]
                chunk_c2w_s_latent = c2w_source[lfs:lfe]
                chunk_c2w_t_latent = c2w_target[lfs:lfe]
                chunk_c2w_s = chunk_c2w_s_latent.repeat_interleave(pf_per_lf, dim=0)[:pfe - pfs]
                chunk_c2w_t = chunk_c2w_t_latent.repeat_interleave(pf_per_lf, dim=0)[:pfe - pfs]

                outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
                    chunk_pixels, self.cut3r_model, self.cut3r_size, self.device)
                H_out, W_out = chunk_pixels.shape[2], chunk_pixels.shape[3]

                all_indices = list(range(pfe - pfs))
                rendered, conf = _render_raymap_frames(
                    outputs, state_args, chunk_c2w_s, chunk_c2w_t,
                    self.focal_ratio, all_indices,
                    H_c, W_c, H_out, W_out, self.device, self.cut3r_model)
                conf_video = torch.stack(conf)
                masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)
                         for c in conf]
                z_rendered = _postprocess_rendered_to_z(
                    rendered, masks, self.vae, H_out, W_out, self.device)
                z_rendered_chunks.append(z_rendered)

        return z_rendered_chunks, cut3r_state

    def _generate_long_view_n(
            self,
            prev_view_latents: torch.Tensor,
            prev_view_pixels: torch.Tensor,
            conditional_dict: dict,
            c2w_source: torch.Tensor,
            c2w_target: torch.Tensor,
            chunk_size: int = 21,
            num_overlap_frames: int = 3,
            cut3r_state: dict = None,
            view_id=0,
    ) -> tuple:
        """
        Generate a long view conditioned on the previous view, chunk by chunk.

        Args:
            prev_view_latents: [1, total_frames, C, H, W] — clean previous view
            prev_view_pixels: [total_frames, 3, H, W] in [0,1]
            conditional_dict: text embeddings + cam_emb
            c2w_source, c2w_target: [total_frames, 4, 4]
            chunk_size: 21
            num_overlap_frames: frames to re-use between chunks
            cut3r_state: accumulated CUT3R state from all previous views (None = no accumulation)
        Returns:
            all_latents: [1, total_frames, C, H, W]
            all_pixels: decoded pixels
            cut3r_state: updated state (or None)
        """
        total_frames = prev_view_latents.shape[1]
        camera_embeddings = conditional_dict["cam_emb"]
        pf_per_lf = 4  # VAE temporal compression

        # Step 1: CUT3R for all chunks
        use_cutt3r = (self.cutt3r_method is not None and self.cut3r_model is not None)
        if use_cutt3r:
            z_rendered_chunks, cut3r_state = self._compute_cut3r_per_chunk(
                prev_view_pixels, c2w_source, c2w_target, latent_chunk_size=chunk_size,
                cut3r_state=cut3r_state, view_id=view_id)
        else:
            z_rendered_chunks = [None] * ((total_frames + chunk_size - 1) // chunk_size)

        # Step 2: Generate chunk by chunk with overlap
        num_gen_frames = chunk_size - num_overlap_frames  # new frames per chunk
        num_chunks = 1 + (total_frames - chunk_size) // num_gen_frames
        num_chunks = num_chunks if len(z_rendered_chunks) == num_chunks else len(z_rendered_chunks)
        all_latent_chunks = []
        all_pixel_chunks = []
        overlap_latents = None

        for chunk_idx in tqdm(range(num_chunks), desc=f"Long view chunks"):
            if chunk_idx == 0:
                fs = 0
                fe = chunk_size
            else:
                fs = chunk_idx * num_gen_frames
                fe = fs + chunk_size

            # Slice previous view for this chunk
            v0_chunk = prev_view_latents[:, fs:fe]
            z_rendered = z_rendered_chunks[chunk_idx] if use_cutt3r else None

            # Noise for this chunk
            noise_chunk = torch.randn(
                [1, chunk_size, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)

            chunk_cond = {
                "prompt_embeds": conditional_dict["prompt_embeds"],
                "cam_emb": camera_embeddings,
            }

            if z_rendered is not None:
                z_rendered_both = torch.cat([
                    torch.zeros_like(z_rendered), z_rendered
                ], dim=0)
            else:
                z_rendered_both = None

            # Generate this chunk
            view1_chunk = self._student_generate_view1(
                noise=noise_chunk,
                view0_latent=v0_chunk,
                conditional_dict=chunk_cond,
                z_rendered_full=z_rendered_both.to(dtype=noise_chunk.dtype, device=self.device) if z_rendered_both is not None else None,
                start_latents=overlap_latents,
            )

            chunk_pixels = self.vae.decode_to_pixel(view1_chunk)
            chunk_pixels = (chunk_pixels * 0.5 + 0.5).clamp(0, 1)  # [1, T_pix, 3, H, W]

            # Keep non-overlap portion
            if chunk_idx < num_chunks - 1:
                keep_latents = view1_chunk[:, :num_gen_frames]
                num_drop_pixels = pf_per_lf * (num_overlap_frames - 1) + 1
                keep_pixels = chunk_pixels[:, :-num_drop_pixels]

                # Re-encode boundary pixel frame
                boundary_pixel = chunk_pixels[:, -num_drop_pixels:-num_drop_pixels + 1]
                boundary_latent = self.vae.encode_to_latent(
                    (boundary_pixel.transpose(1, 2) * 2.0 - 1.0).to(torch.bfloat16)
                ).to(torch.bfloat16)

                overlap_latents = torch.cat([
                    boundary_latent, view1_chunk[:, -(num_overlap_frames - 1):]
                ], dim=1)  # [1, overlap, C, H, W]
            else:
                keep_latents = view1_chunk
                keep_pixels = chunk_pixels

            all_latent_chunks.append(keep_latents)
            all_pixel_chunks.append(keep_pixels)

        all_latents = torch.cat(all_latent_chunks, dim=1)
        all_pixels = torch.cat(all_pixel_chunks, dim=1)
        return all_latents, all_pixels, cut3r_state

    def _roll_kv_cache(self, kv_cache, evict_frames):
        """
        Evict the oldest `evict_frames` frames from the KV cache by shifting left.

        This is the core of approach (c): instead of rebuilding the cache from
        overlap latents, we just drop the oldest entries and keep generating.

        Args:
            kv_cache: list of 30 layer dicts, each with k/v [B, cache_tokens, 12, 128]
            evict_frames: number of frames to evict (typically num_frame_per_block)
        """
        evict_tokens = evict_frames * self.frame_seq_length
        for layer_cache in kv_cache:
            for key in ("k", "v"):
                t = layer_cache[key]  # [B, cache_size, 12, 128]
                t[:, :-evict_tokens] = t[:, evict_tokens:].clone()
                t[:, -evict_tokens:] = 0
            layer_cache["global_end_index"] -= evict_tokens
            layer_cache["local_end_index"] -= evict_tokens

    def inference_long(
            self,
            text_prompts: List[str],
            camera_embeddings: torch.Tensor,
            num_views: int = 2,
            total_frames: int = 42,
            chunk_size: int = 21,
            num_overlap_frames: int = 3,
            negative_prompt: str = "",
            c2w_matrices: Optional[List[tuple]] = None,
            accumulate_views: bool = False,
            rolling_kv: bool = False,
            max_cache_frames: int = 21,

    ):
        """
        Long multi-view generation via chunked overlap.

        Args:
            text_prompts: list of prompts
            camera_embeddings: [2, F, 12] — same pair reused for all view transitions
            num_views: number of views to generate
            total_frames: total latent frames per view
            chunk_size: 21 latent frames per chunk
            num_overlap_frames: overlap between chunks
            c2w_matrices: list of (c2w_source, c2w_target) per view transition
            accumulate_views: if True, CUT3R state accumulates across all previous views.
                              if False, CUT3R runs fresh from only the immediate predecessor.
        Returns:
            all_videos: [num_views, total_pixel_frames, 3, H, W]
        """
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["cam_emb"] = camera_embeddings
        unconditional_dict = self.text_encoder(
            text_prompts=[negative_prompt] * len(text_prompts))
        unconditional_dict["cam_emb"] = camera_embeddings

        # Phase 1: Long v0
        print(f"=== Phase 1: Generating long v0 ({total_frames} frames) ===")
        noise_shape = (2, chunk_size, 16, 60, 104)

        if total_frames > chunk_size:
            v0_latents, v0_pixels = self._generate_long_view0(
                total_frames=total_frames,
                num_overlap_frames=num_overlap_frames,
                noise_shape=noise_shape,
                conditional_dict=conditional_dict,
            )
        else:
            # Single chunk: just run _student_generate_view0 once
            noise = torch.randn(noise_shape, device=self.device, dtype=torch.bfloat16)
            v0_latents = self._student_generate_view0(
                noise=noise,
                conditional_dict=conditional_dict,
            )[0:1]
            v0_pixels = self.vae.decode_to_pixel(v0_latents)
            v0_pixels = (v0_pixels * 0.5 + 0.5).clamp(0, 1)

        all_view_pixels = [v0_pixels[0]]  # [F, 3, H, W]
        prev_latents = v0_latents
        prev_pixels = v0_pixels[0]  # [F, 3, H, W]

        # Initialize accumulated CUT3R state from v0 if accumulating
        use_cutt3r = (self.cutt3r_method is not None and self.cut3r_model is not None)
        accumulated_pixels = [prev_pixels] if (accumulate_views and use_cutt3r) else []

        # if accumulate_views and use_cutt3r:
        #     print("Initializing accumulated CUT3R state from v0...")
        #     outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
        #         prev_pixels, self.cut3r_model, self.cut3r_size, self.device)
        #     H_out, W_out = prev_pixels.shape[2], prev_pixels.shape[3]
        #     cut3r_state = {
        #         "outputs": outputs, "state_args": state_args,
        #         "H_c": H_c, "W_c": W_c, "H_out": H_out, "W_out": W_out,
        #     }
        #
        # Phase 2+: Generate subsequent views
        for view_idx in range(1, num_views):
            print(f"=== Phase {view_idx + 1}: Generating view {view_idx} ===")

            c2w_source, c2w_target = c2w_matrices[view_idx - 1]
            cut3r_state = None
            if accumulate_views and use_cutt3r:
                print(f"Building CUT3R state from {len(accumulated_pixels)} accumulated views...")
                all_pixels_cat = torch.cat(accumulated_pixels, dim=0)  # [total_pix_frames, 3, H, W]
                outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
                    all_pixels_cat, self.cut3r_model, self.cut3r_size, self.device)
                H_out, W_out = all_pixels_cat.shape[2], all_pixels_cat.shape[3]
                cut3r_state = {
                    "outputs": outputs, "state_args": state_args,
                    "H_c": H_c, "W_c": W_c, "H_out": H_out, "W_out": W_out,
                }
            view_latents, view_pixels, cut3r_state = self._generate_long_view_n(
                prev_view_latents=prev_latents,
                prev_view_pixels=prev_pixels,
                conditional_dict={
                    "prompt_embeds": conditional_dict["prompt_embeds"],
                    "cam_emb": camera_embeddings,
                },
                c2w_source=c2w_source,
                c2w_target=c2w_target,
                chunk_size=chunk_size,
                num_overlap_frames=num_overlap_frames,
                cut3r_state=cut3r_state,
                view_id=view_idx,
            )

            all_view_pixels.append(view_pixels[0])
            prev_latents = view_latents
            prev_pixels = view_pixels[0]

            # If accumulating, feed this view's pixels into state for next view
            if accumulate_views and use_cutt3r:
                accumulated_pixels.append(prev_pixels)

        min_frames = 1500
        for arr in all_view_pixels:
            min_frames = min (arr.shape[0], min_frames)
        for i, arr in enumerate(all_view_pixels):
            all_view_pixels[i] = all_view_pixels[i][:min_frames]
        return torch.stack(all_view_pixels, dim=0)  # [num_views, F, 3, H, W]

    def _generate_v0_rolling(
            self,
            total_frames: int,
            conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Generate v0 via continuous rolling Self-Forcing (both_noisy joint denoising).
        No chunks, no overlap, no VAE re-encode. Deep Sink activates automatically
        if model was constructed with sink_size > 0.

        Args:
            total_frames: total latent frames to generate
            conditional_dict: dict with 'prompt_embeds' and 'cam_emb'
        Returns:
            v0_latents: [1, total_frames, C, H, W]
        """
        noise = torch.randn(
            [2, total_frames, 16, 60, 104],
            device=self.device, dtype=torch.bfloat16)

        assert total_frames % self.num_frame_per_block == 0
        num_blocks = total_frames // self.num_frame_per_block

        # KV cache sized to rolling window
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = total_frames * self.frame_seq_length

        kv_cache = self._init_kv_cache(
            batch_size=2, kv_cache_size=kv_cache_size,
            dtype=noise.dtype, device=noise.device)
        crossattn_cache = self._init_crossattn_cache(
            batch_size=2, dtype=noise.dtype, device=noise.device)

        output = torch.zeros_like(noise)  # [2, total_frames, C, H, W]
        camera_embeddings = conditional_dict["cam_emb"]
        current_start_frame = 0

        for block_idx in tqdm(range(num_blocks), desc="Rolling v0"):
            fs = current_start_frame
            fe = fs + self.num_frame_per_block

            noisy_v0 = noise[0:1, fs:fe]
            noisy_v1 = noise[1:2, fs:fe]
            conditional_dict["cam_emb"] = camera_embeddings[:, fs:fe]
            conditional_dict["z_rendered"] = None

            # Multi-step denoising
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=self.device, dtype=torch.int64) * current_timestep

                model_input = torch.cat([noisy_v0, noisy_v1], dim=0)

                _, denoised_pred = self.student(
                    noisy_image_or_video=model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                v0_denoised = denoised_pred[0:1]
                v1_denoised = denoised_pred[1:2]

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_v0 = self.student_scheduler.add_noise(
                        v0_denoised.flatten(0, 1),
                        torch.randn_like(v0_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=self.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

                    noisy_v1 = self.student_scheduler.add_noise(
                        v1_denoised.flatten(0, 1),
                        torch.randn_like(v1_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=self.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            # Record
            output[0:1, fs:fe] = v0_denoised
            output[1:2, fs:fe] = v1_denoised

            # Cache update at context noise
            ctx_t = torch.ones(
                [2, self.num_frame_per_block],
                device=self.device, dtype=torch.int64) * self.context_noise

            cache_input = torch.cat([v0_denoised, v1_denoised], dim=0)
            self.student(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += self.num_frame_per_block

        return output[0:1]  # [1, total_frames, C, H, W]

    def _generate_view_n_rolling(
            self,
            view0_latent: torch.Tensor,
            z_rendered_full: Optional[torch.Tensor],
            conditional_dict: dict,
            total_frames: int,
    ) -> torch.Tensor:
        """
        Generate view N via continuous rolling Self-Forcing, conditioned on
        clean view0 latent + CUT3R geometric prior. No chunks, no overlap.

        Args:
            view0_latent: [1, total_frames, C, H, W] — clean previous view
            z_rendered_full: [1, 17, total_frames, H_lat, W_lat] or None
            conditional_dict: dict with 'prompt_embeds' and 'cam_emb'
            total_frames: total latent frames
        Returns:
            view_latents: [1, total_frames, C, H, W]
        """
        noise = torch.randn(
            [1, total_frames, 16, 60, 104],
            device=self.device, dtype=torch.bfloat16)

        assert total_frames % self.num_frame_per_block == 0
        num_blocks = total_frames // self.num_frame_per_block

        # KV cache sized to rolling window
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = total_frames * self.frame_seq_length

        kv_cache = self._init_kv_cache(
            batch_size=2, kv_cache_size=kv_cache_size,
            dtype=noise.dtype, device=noise.device)
        crossattn_cache = self._init_crossattn_cache(
            batch_size=2, dtype=noise.dtype, device=noise.device)

        # Prepare z_rendered for both slots: [zeros for v0, real for v1]
        if z_rendered_full is not None:
            z_rendered_both = torch.cat([
                torch.zeros_like(z_rendered_full), z_rendered_full
            ], dim=0).to(torch.bfloat16)  # [2, 17, total_frames, H_lat, W_lat]
        else:
            z_rendered_both = None

        output_view = torch.zeros_like(noise)  # [1, total_frames, C, H, W]
        camera_embeddings = conditional_dict["cam_emb"]
        current_start_frame = 0

        for block_idx in tqdm(range(num_blocks), desc="Rolling view N"):
            fs = current_start_frame
            fe = fs + self.num_frame_per_block

            # Slot 0: clean view0 for this block
            gt_view0_block = view0_latent[:, fs:fe]

            # Slot 1: noisy target view
            noisy_view1 = noise[:, fs:fe]

            # Slice conditioning
            conditional_dict["cam_emb"] = camera_embeddings[:, fs:fe]
            if z_rendered_both is not None:
                conditional_dict["z_rendered"] = z_rendered_both[:, :, fs:fe]
            else:
                conditional_dict["z_rendered"] = None

            # Multi-step denoising
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=self.device, dtype=torch.int64) * current_timestep

                model_input = torch.cat([gt_view0_block, noisy_view1], dim=0)

                _, denoised_pred = self.student(
                    noisy_image_or_video=model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                view1_denoised = denoised_pred[1:2]

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_view1 = self.student_scheduler.add_noise(
                        view1_denoised.flatten(0, 1),
                        torch.randn_like(view1_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=self.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            # Record
            output_view[:, fs:fe] = view1_denoised

            # Cache update at context noise
            ctx_t = torch.ones(
                [2, self.num_frame_per_block],
                device=self.device, dtype=torch.int64) * self.context_noise

            cache_input = torch.cat([gt_view0_block, view1_denoised], dim=0)
            self.student(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += self.num_frame_per_block

        return output_view  # [1, total_frames, C, H, W]

    def _generate_view_n_rolling_lazy(
            self,
            view0_latent: torch.Tensor,
            z_rendered_full: Optional[torch.Tensor],
            conditional_dict: dict,
            total_frames: int,
    ) -> torch.Tensor:
        """
        Same as _generate_view_n_rolling but builds z_rendered_both per-block
        instead of pre-allocating the full [2, 17, total_frames, H, W] tensor.
        """
        noise = torch.randn(
            [1, total_frames, 16, 60, 104],
            device=self.device, dtype=torch.bfloat16)

        assert total_frames % self.num_frame_per_block == 0
        num_blocks = total_frames // self.num_frame_per_block

        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = total_frames * self.frame_seq_length

        kv_cache = self._init_kv_cache(
            batch_size=2, kv_cache_size=kv_cache_size,
            dtype=noise.dtype, device=noise.device)
        crossattn_cache = self._init_crossattn_cache(
            batch_size=2, dtype=noise.dtype, device=noise.device)

        output_view = torch.zeros_like(noise)
        camera_embeddings = conditional_dict["cam_emb"]
        current_start_frame = 0

        for block_idx in tqdm(range(num_blocks), desc="Rolling view N (lazy)"):
            fs = current_start_frame
            fe = fs + self.num_frame_per_block

            gt_view0_block = view0_latent[:, fs:fe]
            noisy_view1 = noise[:, fs:fe]

            conditional_dict["cam_emb"] = camera_embeddings[:, fs:fe]
            if z_rendered_full is not None:
                z_block = z_rendered_full[:, :, fs:fe]
                conditional_dict["z_rendered"] = torch.cat([
                    torch.zeros_like(z_block), z_block
                ], dim=0).to(torch.bfloat16)
            else:
                conditional_dict["z_rendered"] = None

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=self.device, dtype=torch.int64) * current_timestep

                model_input = torch.cat([gt_view0_block, noisy_view1], dim=0)

                _, denoised_pred = self.student(
                    noisy_image_or_video=model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                view1_denoised = denoised_pred[1:2]

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_view1 = self.student_scheduler.add_noise(
                        view1_denoised.flatten(0, 1),
                        torch.randn_like(view1_denoised.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=self.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            output_view[:, fs:fe] = view1_denoised

            ctx_t = torch.ones(
                [2, self.num_frame_per_block],
                device=self.device, dtype=torch.int64) * self.context_noise

            cache_input = torch.cat([gt_view0_block, view1_denoised], dim=0)
            self.student(
                noisy_image_or_video=cache_input,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += self.num_frame_per_block

        return output_view
    def _compute_cut3r_full(
            self,
            prev_view_pixels: torch.Tensor,
            c2w_source: torch.Tensor,
            c2w_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run CUT3R on all pixel frames, render all target frames, VAE-encode
        in chunks to produce z_rendered for the full sequence.

        Args:
            prev_view_pixels: [T_pix, 3, H, W] in [0,1]
            c2w_source: [T_latent, 4, 4]
            c2w_target: [T_latent, 4, 4]
        Returns:
            z_rendered: [1, 17, T_latent, H_lat, W_lat]
        """
        pf_per_lf = 4
        total_pixel_frames = prev_view_pixels.shape[0]
        total_latent_frames = c2w_source.shape[0]
        H_out, W_out = prev_view_pixels.shape[2], prev_view_pixels.shape[3]

        # ── Step 1: Build CUT3R 3D state from all pixel frames ────────────
        print(f"  CUT3R: processing {total_pixel_frames} pixel frames...")
        outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
            prev_view_pixels, self.cut3r_model, self.cut3r_size, self.device)

        render_pixel_frames = (total_latent_frames - 1) * pf_per_lf + 1
        c2w_source_pixel = c2w_source.repeat_interleave(pf_per_lf, dim=0)[:render_pixel_frames]
        c2w_target_pixel = c2w_target.repeat_interleave(pf_per_lf, dim=0)[:render_pixel_frames]
        render_start = total_pixel_frames - render_pixel_frames

        # ── Step 3: Render last view's frames only ────────────────────────
        all_indices = list(range(render_start, total_pixel_frames))

        rendered, conf = _render_raymap_frames(
            outputs, state_args, c2w_source_pixel, c2w_target_pixel,
            self.focal_ratio, all_indices,
            H_c, W_c, H_out, W_out, self.device, self.cut3r_model,
            c2w_offset=render_start)
        conf_video = torch.stack(conf)
        masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)
                 for c in conf]
        # ── Step 4: VAE-encode rendered frames in 21-latent-frame chunks ──
        print(f"  CUT3R: encoding z_rendered ({total_latent_frames} latent frames)...")
        latent_chunk_size = 21
        pixel_chunk_size = (latent_chunk_size - 1) * pf_per_lf + 1  # 81

        z_chunks = []
        for lf_start in range(0, total_latent_frames, latent_chunk_size):
            lf_end = min(lf_start + latent_chunk_size, total_latent_frames)
            pf_start = 0 if lf_start == 0 else (lf_start - 1) * pf_per_lf + 1
            pf_end = min((lf_end - 1) * pf_per_lf + 1, total_pixel_frames)

            chunk_rendered = rendered[pf_start:pf_end]
            chunk_masks = masks[pf_start:pf_end]

            z_chunk = _postprocess_rendered_to_z(
                chunk_rendered, chunk_masks, self.vae,
                H_out, W_out, self.device)
            z_chunks.append(z_chunk)
        # Concatenate along temporal dim: [1, 17, T_latent, H_lat, W_lat]
        z_rendered = torch.cat(z_chunks, dim=2)

        print(f"  CUT3R: z_rendered shape {z_rendered.shape}")
        return z_rendered
    @torch.no_grad()
    def inference_long_rolling(
            self,
            text_prompts: List[str],
            camera_embeddings: torch.Tensor,
            num_views: int = 2,
            total_frames: int = 42,
            negative_prompt: str = "",
            c2w_matrices: Optional[List[tuple]] = None,
            accumulate_views: bool = False,
    ):
        """
        Long multi-view generation via continuous rolling KV (Option 1/3).
        No chunk boundaries, no VAE re-encode. Deep Sink activates
        automatically if the model was constructed with sink_size > 0.
        """
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["cam_emb"] = camera_embeddings
        torch.cuda.synchronize()
        start_all = time.time()
        torch.cuda.reset_peak_memory_stats()

        # ── Phase 1: Generate long v0 (continuous rolling) ────────────────
        print(f"=== Phase 1: Rolling v0 ({total_frames} frames) ===")
        v0_latents = self._generate_v0_rolling(
            total_frames=total_frames,
            conditional_dict=conditional_dict,
        )  # [1, total_frames, C, H, W]
        start = time.time()
        v0_pixels = self.vae.decode_to_pixel(v0_latents, use_cache=False)
        v0_pixels = (v0_pixels * 0.5 + 0.5).clamp(0, 1)  # [1, T_pix, 3, H, W]
        self.vae.model.clear_cache()
        end = time.time()
        print(f"Elapsed DEC: {end - start:.2f}s")

        all_view_pixels = [v0_pixels[0]]  # [T_pix, 3, H, W]
        prev_latents = v0_latents
        prev_pixels = v0_pixels[0]

        # ── Phase 2+: Generate subsequent views ───────────────────────────
        use_cutt3r = (self.cutt3r_method is not None and self.cut3r_model is not None)
        accumulated_pixels = [prev_pixels] if (accumulate_views and use_cutt3r) else []

        for view_idx in range(1, num_views):
            print(f"=== Phase {view_idx + 1}: Rolling view {view_idx} ===")

            c2w_source, c2w_target = c2w_matrices[view_idx - 1]

            start = time.time()

            # CUT3R geometric prior (single pass, no chunking)
            z_rendered = None
            if use_cutt3r:
                if accumulate_views and len(accumulated_pixels) > 1:
                    # Feed all previous views to CUT3R for richer geometry
                    all_pixels_cat = torch.cat(accumulated_pixels, dim=0)
                    z_rendered = self._compute_cut3r_full(
                        prev_view_pixels=all_pixels_cat,
                        c2w_source=c2w_source,
                        c2w_target=c2w_target,
                    )
                else:
                    z_rendered = self._compute_cut3r_full(
                        prev_view_pixels=prev_pixels,
                        c2w_source=c2w_source,
                        c2w_target=c2w_target,
                    )
            end = time.time()
            print(f"Elapsed CUtt: {end - start:.2f}s")

            # Generate view continuously
            view_latents = self._generate_view_n_rolling(
                view0_latent=prev_latents,
                z_rendered_full=z_rendered,
                conditional_dict={
                    "prompt_embeds": conditional_dict["prompt_embeds"],
                    "cam_emb": camera_embeddings,
                },
                total_frames=total_frames,
            )
            start = time.time()

            view_pixels = self.vae.decode_to_pixel(view_latents, use_cache=False)
            view_pixels = (view_pixels * 0.5 + 0.5).clamp(0, 1)
            self.vae.model.clear_cache()
            end = time.time()
            print(f"Elapsed DEC: {end - start:.2f}s")

            all_view_pixels.append(view_pixels[0])
            prev_latents = view_latents
            prev_pixels = view_pixels[0]

            if accumulate_views and use_cutt3r:
                accumulated_pixels.append(prev_pixels)

        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        torch.cuda.synchronize()
        end_all = time.time()
        print(f"Elapsed Total: {end_all - start_all:.2f}s")

        # Trim to common length
        min_frames = min(arr.shape[0] for arr in all_view_pixels)
        all_view_pixels = [arr[:min_frames] for arr in all_view_pixels]

        return torch.stack(all_view_pixels, dim=0)  # [num_views, T_pix, 3, H, W]

    @torch.no_grad()
    def inference_rolling_chain(
            self,
            text_prompts: List[str],
            camera_embeddings_list: list,
            c2w_matrices: list,
            num_views: int = 2,
            total_frames: int = 42,
            accumulate_views: bool = False,
    ):
        """
        Extreme-scale multi-view generation. Memory-optimized:
        pixels on CPU, chunked decode, model offloading during CUT3R.

        Args:
            text_prompts: list of prompts (same prompt)
            camera_embeddings_list: list of [2, F, 12], one per view transition
            c2w_matrices: list of (c2w_s, c2w_t), one per view transition
            num_views: total views to generate
            total_frames: latent frames per view
            accumulate_views: accumulate CUT3R state across views
        Returns:
            video: [num_views, T_pix, 3, H, W] on CPU
        """
        if accumulate_views and num_views > 3:
            print("[WARN] Disabling accumulation for >3 views (memory)")
            accumulate_views = False

        # ── Phase 1: V0 ──────────────────────────────────────────────────
        print(f"=== Phase 1: Rolling v0 ({total_frames} frames) ===")
        conditional_dict = self.text_encoder(
            text_prompts=[text_prompts[0], text_prompts[0]])
        conditional_dict["cam_emb"] = camera_embeddings_list[0]
        torch.cuda.synchronize()
        start_all = time.time()
        torch.cuda.reset_peak_memory_stats()

        v0_latents = self._generate_v0_rolling(
            total_frames=total_frames,
            conditional_dict=conditional_dict,
        )

        prev_pixels = self._decode_to_cpu(v0_latents)
        all_view_pixels = [prev_pixels]
        prev_latents = v0_latents

        use_cutt3r = (self.cutt3r_method is not None and self.cut3r_model is not None)
        accumulated_pixels = [prev_pixels] if (accumulate_views and use_cutt3r) else []

        # ── Phase 2+: Subsequent views ────────────────────────────────────
        for view_idx in range(1, num_views):
            print(f"=== Phase {view_idx + 1}: Rolling view {view_idx} ===")

            c2w_source, c2w_target = c2w_matrices[view_idx - 1]

            # ── CUT3R: offload student, run, reload ──────────────────────
            z_rendered = None
            if use_cutt3r:
                self.student.to("cpu")
                torch.cuda.empty_cache()

                if accumulate_views and len(accumulated_pixels) > 1:
                    pixels_for_cut3r = torch.cat(accumulated_pixels, dim=0).to(self.device)
                else:
                    pixels_for_cut3r = prev_pixels.to(self.device)

                z_chunks, _ = self._compute_cut3r_per_chunk(
                    pixels_for_cut3r,
                    c2w_source, c2w_target,
                    latent_chunk_size=21,
                    cut3r_state=None,
                    view_id=view_idx,
                )
                z_rendered = torch.cat(z_chunks, dim=2)

                del pixels_for_cut3r
                torch.cuda.empty_cache()
                self.student.to(self.device)

            # ── Generate view ─────────────────────────────────────────────
            view_cond = {
                "prompt_embeds": conditional_dict["prompt_embeds"],
                "cam_emb": camera_embeddings_list[view_idx - 1],
            }

            view_latents = self._generate_view_n_rolling_lazy(
                view0_latent=prev_latents,
                z_rendered_full=z_rendered,
                conditional_dict=view_cond,
                total_frames=total_frames,
            )

            # Free z_rendered
            del z_rendered
            torch.cuda.empty_cache()

            # ── Decode to CPU ─────────────────────────────────────────────
            view_pixels = self._decode_to_cpu(view_latents)

            all_view_pixels.append(view_pixels)

            # Free old latents
            del prev_latents
            prev_latents = view_latents
            prev_pixels = view_pixels

            if accumulate_views and use_cutt3r:
                accumulated_pixels.append(prev_pixels)

        del prev_latents
        torch.cuda.empty_cache()
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        torch.cuda.synchronize()
        end_all = time.time()
        print(f"Elapsed Total: {end_all - start_all:.2f}s")

        min_frames = min(arr.shape[0] for arr in all_view_pixels)
        all_view_pixels = [arr[:min_frames] for arr in all_view_pixels]

        return torch.stack(all_view_pixels, dim=0)  # on CPU



    def _init_kv_cache(self, batch_size, kv_cache_size, dtype, device):
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        return kv_cache

    def _init_crossattn_cache(self, batch_size, dtype, device):
        cache = []
        for _ in range(self.num_transformer_blocks):
            cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False,
            })
        return cache

    def _decode_to_cpu(self, latents, chunk_size=21):
        """Decode latents to pixels in chunks, move each to CPU immediately."""
        total = latents.shape[1]
        pixel_chunks = []
        for lf in range(0, total, chunk_size):
            chunk = latents[:, lf:min(lf + chunk_size, total)]
            px = self.vae.decode_to_pixel(chunk, use_cache=True)
            px = (px * 0.5 + 0.5).clamp(0, 1)
            pixel_chunks.append(px[0].cpu())
        self.vae.model.clear_cache()
        return torch.cat(pixel_chunks, dim=0)  # [T_pix, 3, H, W] on CPU