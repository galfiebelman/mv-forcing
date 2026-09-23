"""
Text-to-Multi-View Inference Pipeline.

Two V0 paths:
  - Self-Forcing (causal single-view) → V0
  - Student both-noisy → V0

Always: CUT3R on V0 → z_rendered → Student V1

No ReCamMaster, no source latents, no teacher.
"""

from typing import Optional
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.wan_wrapper import (
    WanDiffusionWrapper, WanDSDiffusionWrapper, WanMVBidirDiffusionWrapper,
    WanTextEncoder, WanVAEWrapper,
)
from pipeline.cut3r_prior import (
    _run_cut3r_on_frames, _render_raymap_frames,
    _postprocess_rendered_to_z,_feed_frames_to_cut3r_state,
)


class TextToMVPipeline(nn.Module):

    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.args = args

        # ── Text encoder + VAE ──
        self.text_encoder = WanTextEncoder()
        self.vae = WanVAEWrapper()

        # ── Self-Forcing generator (causal, single-view) ──
        model_kwargs = dict(getattr(args, "model_kwargs", {}) or {})
        local_attn_size = getattr(args, "local_attn_size", -1)
        sink_size = getattr(args, "sink_size", 0)
        if sink_size > 0:
            self.sf_generator = WanDSDiffusionWrapper(
                local_attn_size=local_attn_size, sink_size=sink_size)
        else:
            self.sf_generator = WanDiffusionWrapper(
                **model_kwargs, is_causal=True)
        self.sf_generator.requires_grad_(False)

        self.sf_scheduler = self.sf_generator.get_scheduler()
        self.sf_local_attn_size = self.sf_generator.model.local_attn_size
        self.local_attn_size = local_attn_size

        # ── Student (MV-Forcing, 2-view causal) ──
        self.student = WanMVBidirDiffusionWrapper(
            use_cut3r=True,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )
        self.student.requires_grad_(False)

        self.student_scheduler = self.student.get_scheduler()

        # ── Shared config ──
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        if getattr(args, "warp_denoising_step", False):
            timesteps = torch.cat((
                self.sf_scheduler.timesteps.cpu(),
                torch.tensor([0], dtype=torch.float32)
            )).to(device)
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.context_noise = getattr(args, "context_noise", 0)
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560

        # ── CUT3R (set externally) ──
        self.cut3r_model = None
        self.cut3r_size = getattr(args, "cut3r_size", 512)
        self.focal_ratio = getattr(args, "focal_ratio", 24.0 / 23.76)
        self.cutt3r_method = "raymap"
    # ═════════════════════════════════════════════════════════════════════
    # Cache helpers
    # ═════════════════════════════════════════════════════════════════════

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

    def _init_sf_kv_cache(self, dtype, device):
        """KV cache sized for Self-Forcing (batch=1, respects local_attn_size)."""
        if self.sf_local_attn_size != -1:
            kv_cache_size = self.sf_local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = 32760
        return self._init_kv_cache(1, kv_cache_size, dtype, device)

    def _reset_cache(self, kv_cache, crossattn_cache):
        for block_idx in range(self.num_transformer_blocks):
            kv_cache[block_idx]["global_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=self.device)
            kv_cache[block_idx]["local_end_index"] = torch.tensor(
                [0], dtype=torch.long, device=self.device)
            crossattn_cache[block_idx]["is_init"] = False

    # ═════════════════════════════════════════════════════════════════════
    # V0 Path 1: Self-Forcing (causal, single-view)
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _generate_v0_sf(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ):
        """
        Self-Forcing causal generation for V0 (single view).
        Ported from BaselineInferencePipeline._generate_v0_sf.

        Args:
            noise: [1, F, C, H, W]
            conditional_dict: dict with 'prompt_embeds' [1, L, D]
        Returns:
            latents: [1, F, C, H, W]
        """
        batch_size, num_frames, C, H, W = noise.shape

        if not self.independent_first_frame:
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        kv_cache = self._init_sf_kv_cache(noise.dtype, noise.device)
        crossattn_cache = self._init_crossattn_cache(1, noise.dtype, noise.device)

        output = torch.zeros(
            [batch_size, num_frames, C, H, W],
            device=noise.device, dtype=noise.dtype)

        current_start_frame = 0
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame:
            all_num_frames = [1] + all_num_frames

        for current_num_frames in tqdm(all_num_frames, desc="SF view 0"):
            noisy_input = noise[:, current_start_frame:current_start_frame + current_num_frames]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device, dtype=torch.int64) * current_timestep

                _, denoised_pred = self.sf_generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.sf_scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])

            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Re-run clean to update KV cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            self.sf_generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames

        return output  # [1, F, C, H, W]

    # ═════════════════════════════════════════════════════════════════════
    # V0 Path 2: Student both-noisy
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _generate_v0_student(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
    ):
        """
        Both views from noise (both_noisy path). Returns V0 only.

        Args:
            noise: [2, F, C, H, W]
            conditional_dict: dict with 'prompt_embeds' [2, L, D] and 'cam_emb' [2, F, 12]
        Returns:
            view0_latent: [1, F, C, H, W]
        """
        num_frames = noise.shape[1]
        num_blocks = num_frames // self.num_frame_per_block

        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = num_frames * self.frame_seq_length
        kv_cache = self._init_kv_cache(2, kv_cache_size, noise.dtype, noise.device)
        crossattn_cache = self._init_crossattn_cache(2, noise.dtype, noise.device)

        output = torch.zeros_like(noise)  # [2, F, C, H, W]
        current_start_frame = 0
        camera_embeddings = conditional_dict["cam_emb"]

        for block_idx in tqdm(range(num_blocks), desc="Student V0 (both noisy)"):
            frame_start = current_start_frame
            frame_end = current_start_frame + self.num_frame_per_block

            noisy_v0 = noise[0:1, frame_start:frame_end]
            noisy_v1 = noise[1:2, frame_start:frame_end]

            conditional_dict["cam_emb"] = camera_embeddings[:, frame_start:frame_end]
            conditional_dict["z_rendered"] = None

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64
                ) * current_timestep

                model_input = torch.cat([noisy_v0, noisy_v1], dim=0)

                _, denoised_pred = self.student(
                    noisy_image_or_video=model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

                if index < len(self.denoising_step_list) - 1:
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_v0 = self.student_scheduler.add_noise(
                        denoised_pred[0:1].flatten(0, 1),
                        torch.randn_like(denoised_pred[0:1].flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))
                    noisy_v1 = self.student_scheduler.add_noise(
                        denoised_pred[1:2].flatten(0, 1),
                        torch.randn_like(denoised_pred[1:2].flatten(0, 1)),
                        next_timestep * torch.ones(
                            [self.num_frame_per_block],
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            output[:, frame_start:frame_end] = denoised_pred

            ctx_t = torch.ones(
                [2, self.num_frame_per_block],
                device=noise.device, dtype=torch.int64
            ) * self.context_noise

            self.student(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=ctx_t,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            current_start_frame += self.num_frame_per_block

        return output[0:1]  # [1, F, C, H, W]

    # ═════════════════════════════════════════════════════════════════════
    # V1: Student conditioned on V0 + CUT3R z_rendered
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _generate_v1_student(
        self,
        noise: torch.Tensor,
        view0_latent: torch.Tensor,
        conditional_dict: dict,
        z_rendered_full=None,
    ):
        """
        Few-step student V1, conditioned on clean V0 + z_rendered.

        Args:
            noise: [1, F, C, H, W]
            view0_latent: [1, F, C, H, W]
            conditional_dict: dict with 'prompt_embeds' [2, L, D] and 'cam_emb' [2, F, 12]
            z_rendered_full: [1, 17, F, H_lat, W_lat] or None
        Returns:
            view1_latent: [1, F, C, H, W]
        """
        num_frames = noise.shape[1]
        num_blocks = num_frames // self.num_frame_per_block

        kv_cache_size = num_frames * self.frame_seq_length
        kv_cache = self._init_kv_cache(2, kv_cache_size, noise.dtype, noise.device)
        crossattn_cache = self._init_crossattn_cache(2, noise.dtype, noise.device)

        output_view1 = torch.zeros_like(noise)  # [1, F, C, H, W]
        current_start_frame = 0
        camera_embeddings = conditional_dict["cam_emb"]

        if z_rendered_full is not None:
            z_rendered_both = torch.cat([
                torch.zeros_like(z_rendered_full), z_rendered_full
            ], dim=0)  # [2, 17, F, H_lat, W_lat]
        else:
            z_rendered_both = None

        for block_idx in tqdm(range(num_blocks), desc="Student V1"):
            frame_start = current_start_frame
            frame_end = current_start_frame + self.num_frame_per_block

            gt_view0_block = view0_latent[:, frame_start:frame_end]
            noisy_view1 = noise[:, frame_start:frame_end]

            conditional_dict["cam_emb"] = camera_embeddings[:, frame_start:frame_end]
            if z_rendered_both is not None:
                conditional_dict["z_rendered"] = z_rendered_both[:, :, frame_start:frame_end]
            else:
                conditional_dict["z_rendered"] = None

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [2, self.num_frame_per_block],
                    device=noise.device, dtype=torch.int64
                ) * current_timestep

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
                            device=noise.device, dtype=torch.long)
                    ).unflatten(0, (1, self.num_frame_per_block))

            output_view1[:, frame_start:frame_end] = view1_denoised

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

        return output_view1  # [1, F, C, H, W]

    # ═════════════════════════════════════════════════════════════════════
    # CUT3R: V0 pixels → z_rendered
    # ═════════════════════════════════════════════════════════════════════
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

    @torch.no_grad()
    def _compute_z_rendered(self, v0_latent, c2w_cond, c2w_tgt):
        """
        Decode V0 → pixels → CUT3R → raymap render → z_rendered.

        Args:
            v0_latent: [1, F, C, H, W]
            c2w_cond: [F, 4, 4]
            c2w_tgt: [F, 4, 4]
        Returns:
            z_rendered: [1, 17, F, H_lat, W_lat] bfloat16
        """
        v0_pixels = self.vae.decode_to_pixel(v0_latent)
        v0_pixels = (v0_pixels * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
            v0_pixels[0], self.cut3r_model, self.cut3r_size, self.device)
        H_out, W_out = v0_pixels.shape[3], v0_pixels.shape[4]
        frame_indices = list(range(v0_latent.shape[1]))

        rendered_frames, conf_frames = _render_raymap_frames(
            outputs, state_args, c2w_cond, c2w_tgt,
            self.focal_ratio, frame_indices, H_c, W_c, H_out, W_out,
            self.device, self.cut3r_model)

        z_rendered = _postprocess_rendered_to_z(
            rendered_frames, conf_frames, self.vae, H_out, W_out, self.device)
        self.vae.model.clear_cache()

        return z_rendered.to(dtype=torch.bfloat16, device=self.device)

    # ═════════════════════════════════════════════════════════════════════
    # Main entry point
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def inference(
        self,
        text_prompts: list,
        camera_embeddings: torch.Tensor,
        c2w_cond: torch.Tensor,
        c2w_tgt: torch.Tensor,
        negative_prompt: str = "",
        v0_mode: str = "sf",
        num_frames: int = 21,
    ):
        """
        Text → multi-view video.

        Args:
            text_prompts: list of 2 prompts (same prompt repeated)
            camera_embeddings: [2, F, 12]
            c2w_cond: [F, 4, 4]
            c2w_tgt: [F, 4, 4]
            negative_prompt: str
            v0_mode: "sf" or "student"
            num_frames: latent frames (21 = 81 pixel frames)
        Returns:
            video: [2, T_pix, 3, H, W] in [0,1]
        """
        # ── Text encoding ──
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["cam_emb"] = camera_embeddings

        # ── Generate V0 ──
        if v0_mode == "sf":
            print("=== V0: Self-Forcing ===")
            # SF is single-view: use prompt[0] only
            sf_cond = self.text_encoder(text_prompts=[text_prompts[0]])
            noise_v0 = torch.randn(
                [1, num_frames, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)
            v0_latent = self._generate_v0_sf(
                noise=noise_v0,
                conditional_dict=sf_cond,
            )

        elif v0_mode == "student":
            print("=== V0: Student (both noisy) ===")
            noise_both = torch.randn(
                [2, num_frames, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)
            v0_latent = self._generate_v0_student(
                noise=noise_both,
                conditional_dict=conditional_dict,
            )
        else:
            raise ValueError(f"Unknown v0_mode: {v0_mode}")

        # ── CUT3R: V0 → z_rendered ──
        print("=== CUT3R rendering ===")
        z_rendered = self._compute_z_rendered(v0_latent, c2w_cond, c2w_tgt)

        # ── Generate V1 ──
        print("=== V1: Student (conditioned) ===")
        noise_v1 = torch.randn_like(v0_latent)
        v1_latent = self._generate_v1_student(
            noise=noise_v1,
            view0_latent=v0_latent,
            conditional_dict={
                "prompt_embeds": conditional_dict["prompt_embeds"],
                "cam_emb": camera_embeddings,
            },
            z_rendered_full=z_rendered,
        )

        # ── Decode ──
        v0_video = self.vae.decode_to_pixel(v0_latent)
        v0_video = (v0_video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        v1_video = self.vae.decode_to_pixel(v1_latent)
        v1_video = (v1_video * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        return torch.cat([v0_video, v1_video], dim=0)  # [2, T_pix, 3, H, W]


    @torch.no_grad()
    def _generate_v1_student_long(
        self,
        view0_latent: torch.Tensor,
        z_rendered_full,
        conditional_dict: dict,
        total_frames: int,
    ) -> torch.Tensor:
        """
        Generate view N via continuous rolling Self-Forcing, conditioned on
        clean view0 latent + CUT3R geometric prior. No chunks, no overlap.
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

        kv_cache = self._init_kv_cache(2, kv_cache_size, noise.dtype, noise.device)
        crossattn_cache = self._init_crossattn_cache(2, noise.dtype, noise.device)

        if z_rendered_full is not None:
            z_rendered_both = torch.cat([
                torch.zeros_like(z_rendered_full), z_rendered_full
            ], dim=0).to(torch.bfloat16)
        else:
            z_rendered_both = None

        output_view = torch.zeros_like(noise)
        camera_embeddings = conditional_dict["cam_emb"]
        current_start_frame = 0

        for block_idx in tqdm(range(num_blocks), desc="Rolling view N"):
            fs = current_start_frame
            fe = fs + self.num_frame_per_block

            gt_view0_block = view0_latent[:, fs:fe]
            noisy_view1 = noise[:, fs:fe]

            conditional_dict["cam_emb"] = camera_embeddings[:, fs:fe]
            if z_rendered_both is not None:
                conditional_dict["z_rendered"] = z_rendered_both[:, :, fs:fe]
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
    @torch.no_grad()
    def _generate_v1_student_long_lazy(
        self,
        view0_latent: torch.Tensor,
        z_rendered_full,
        conditional_dict: dict,
        total_frames: int,
    ) -> torch.Tensor:
        """Same as _generate_v1_student_long but builds z_rendered_both per-block."""
        noise = torch.randn(
            [1, total_frames, 16, 60, 104],
            device=self.device, dtype=torch.bfloat16)

        assert total_frames % self.num_frame_per_block == 0
        num_blocks = total_frames // self.num_frame_per_block

        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            kv_cache_size = total_frames * self.frame_seq_length

        kv_cache = self._init_kv_cache(2, kv_cache_size, noise.dtype, noise.device)
        crossattn_cache = self._init_crossattn_cache(2, noise.dtype, noise.device)

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
    # ═════════════════════════════════════════════════════════════════════
    # Long: CUT3R full sequence
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _compute_cut3r_full(
        self,
        prev_view_pixels: torch.Tensor,
        c2w_source: torch.Tensor,
        c2w_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        CUT3R on all pixel frames → render all targets → z_rendered.
        """
        pf_per_lf = 4
        total_pixel_frames = prev_view_pixels.shape[0]
        total_latent_frames = c2w_source.shape[0]
        H_out, W_out = prev_view_pixels.shape[2], prev_view_pixels.shape[3]

        print(f"  CUT3R: processing {total_pixel_frames} pixel frames...")
        outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
            prev_view_pixels, self.cut3r_model, self.cut3r_size, self.device)

        render_pixel_frames = (total_latent_frames - 1) * pf_per_lf + 1
        c2w_source_pixel = c2w_source.repeat_interleave(pf_per_lf, dim=0)[:render_pixel_frames]
        c2w_target_pixel = c2w_target.repeat_interleave(pf_per_lf, dim=0)[:render_pixel_frames]
        render_start = total_pixel_frames - render_pixel_frames

        all_indices = list(range(render_start, total_pixel_frames))
        print(f"  CUT3R: rendering {len(all_indices)} target frames...")

        rendered, conf = _render_raymap_frames(
            outputs, state_args, c2w_source_pixel, c2w_target_pixel,
            self.focal_ratio, all_indices,
            H_c, W_c, H_out, W_out, self.device, self.cut3r_model,
            c2w_offset=render_start)
        conf_video = torch.stack(conf)
        masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)
                 for c in conf]

        print(f"  CUT3R: encoding z_rendered ({total_latent_frames} latent frames)...")
        latent_chunk_size = 21

        z_chunks = []
        for lf_start in range(0, total_latent_frames, latent_chunk_size):
            lf_end = min(lf_start + latent_chunk_size, total_latent_frames)
            pf_start = 0 if lf_start == 0 else (lf_start - 1) * pf_per_lf + 1
            pf_end = min((lf_end - 1) * pf_per_lf + 1, render_pixel_frames)

            chunk_rendered = rendered[pf_start:pf_end]
            chunk_masks = masks[pf_start:pf_end]

            z_chunk = _postprocess_rendered_to_z(
                chunk_rendered, chunk_masks, self.vae,
                H_out, W_out, self.device)
            z_chunks.append(z_chunk)

        z_rendered = torch.cat(z_chunks, dim=2)
        print(f"  CUT3R: z_rendered shape {z_rendered.shape}")
        return z_rendered

    # ═════════════════════════════════════════════════════════════════════
    # Long multi-view entry point
    # ═════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def inference_long(
        self,
        text_prompts: list,
        camera_embeddings: torch.Tensor,
        c2w_matrices: list,
        num_views: int = 2,
        total_frames: int = 42,
        v0_mode: str = "sf",
        accumulate_views: bool = False,
    ):
        """
        Long multi-view generation.

        Args:
            text_prompts: list of prompts (same prompt repeated)
            camera_embeddings: [2, total_frames, 12]
            c2w_matrices: list of (c2w_source, c2w_target) per view transition
            num_views: number of views
            total_frames: total latent frames per view
            v0_mode: "sf" or "student"
            accumulate_views: accumulate CUT3R state across views
        Returns:
            video: [num_views, T_pix, 3, H, W] in [0,1]
        """
        # ── Phase 1: V0 ──────────────────────────────────────────────────
        print(f"=== Phase 1: V0 ({v0_mode}, {total_frames} frames) ===")
        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["cam_emb"] = camera_embeddings
        if v0_mode == "sf":
            sf_cond = self.text_encoder(text_prompts=[text_prompts[0]])
            noise_v0 = torch.randn(
                [1, total_frames, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)
            v0_latent = self._generate_v0_sf(
                noise=noise_v0,
                conditional_dict=sf_cond,
            )
        elif v0_mode == "student":
            conditional_dict = self.text_encoder(text_prompts=text_prompts[:2])
            conditional_dict["cam_emb"] = camera_embeddings
            noise_both = torch.randn(
                [2, total_frames, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)
            v0_latent = self._generate_v0_student(
                noise=noise_both,
                conditional_dict=conditional_dict,
            )

        v0_pixels = self.vae.decode_to_pixel(v0_latent)
        v0_pixels = (v0_pixels * 0.5 + 0.5).clamp(0, 1)
        self.vae.model.clear_cache()

        all_view_pixels = [v0_pixels[0]]
        prev_latents = v0_latent
        prev_pixels = v0_pixels[0]

        # ── Phase 2+: Subsequent views ────────────────────────────────────
        accumulated_pixels = [prev_pixels] if accumulate_views else []

        for view_idx in range(1, num_views):
            print(f"=== Phase {view_idx + 1}: View {view_idx} ===")

            c2w_source, c2w_target = c2w_matrices[view_idx - 1]

            # CUT3R
            z_rendered = None
            if self.cut3r_model is not None:
                if accumulate_views and len(accumulated_pixels) > 1:
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

            # Generate view
            conditional_dict["cam_emb"] = camera_embeddings

            view_latents = self._generate_v1_student_long(
                view0_latent=prev_latents,
                z_rendered_full=z_rendered,
                conditional_dict=conditional_dict,
                total_frames=total_frames,
            )

            view_pixels = self.vae.decode_to_pixel(view_latents)
            view_pixels = (view_pixels * 0.5 + 0.5).clamp(0, 1)
            self.vae.model.clear_cache()

            all_view_pixels.append(view_pixels[0])
            prev_latents = view_latents
            prev_pixels = view_pixels[0]

            if accumulate_views:
                accumulated_pixels.append(prev_pixels)

        min_frames = min(arr.shape[0] for arr in all_view_pixels)
        all_view_pixels = [arr[:min_frames] for arr in all_view_pixels]

        return torch.stack(all_view_pixels, dim=0)

    @torch.no_grad()
    def inference_rolling_chain(
            self,
            text_prompts: list,
            camera_embeddings_list: list,
            c2w_matrices: list,
            num_views: int = 2,
            total_frames: int = 42,
            v0_mode: str = "sf",
            accumulate_views: bool = False,
    ):
        """
        Extreme-scale text-to-multi-view. Memory-optimized:
        pixels on CPU, chunked decode, model offloading during CUT3R.
        """
        if accumulate_views and num_views > 3:
            print("[WARN] Disabling accumulation for >3 views (memory)")
            accumulate_views = False

        # ── Phase 1: V0 ──────────────────────────────────────────────────
        print(f"=== Phase 1: V0 ({v0_mode}, {total_frames} frames) ===")

        if v0_mode == "sf":
            sf_cond = self.text_encoder(text_prompts=[text_prompts[0]])
            noise_v0 = torch.randn(
                [1, total_frames, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)
            v0_latent = self._generate_v0_sf(
                noise=noise_v0,
                conditional_dict=sf_cond,
            )
        elif v0_mode == "student":
            student_cond = self.text_encoder(
                text_prompts=[text_prompts[0], text_prompts[0]])
            student_cond["cam_emb"] = camera_embeddings_list[0]
            noise_both = torch.randn(
                [2, total_frames, 16, 60, 104],
                device=self.device, dtype=torch.bfloat16)
            v0_latent = self._generate_v0_student(
                noise=noise_both,
                conditional_dict=student_cond,
            )

        prev_pixels = self._decode_to_cpu(v0_latent)
        all_view_pixels = [prev_pixels]
        prev_latents = v0_latent

        use_cutt3r = self.cut3r_model is not None
        accumulated_pixels = [prev_pixels] if (accumulate_views and use_cutt3r) else []

        # Encode prompt once for all subsequent views
        conditional_dict = self.text_encoder(
            text_prompts=[text_prompts[0], text_prompts[0]])

        # ── Phase 2+: Subsequent views ────────────────────────────────────
        for view_idx in range(1, num_views):
            print(f"=== Phase {view_idx + 1}: View {view_idx} ===")

            c2w_source, c2w_target = c2w_matrices[view_idx - 1]

            # ── CUT3R: offload generators, run chunked, reload ───────────
            z_rendered = None
            if use_cutt3r:
                self.student.to("cpu")
                self.sf_generator.to("cpu")
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
                self.sf_generator.to(self.device)

            # ── Generate view ─────────────────────────────────────────────
            view_cond = {
                "prompt_embeds": conditional_dict["prompt_embeds"],
                "cam_emb": camera_embeddings_list[view_idx - 1],
            }

            view_latents = self._generate_v1_student_long_lazy(
                view0_latent=prev_latents,
                z_rendered_full=z_rendered,
                conditional_dict=view_cond,
                total_frames=total_frames,
            )

            del z_rendered
            torch.cuda.empty_cache()

            # ── Decode to CPU ─────────────────────────────────────────────
            view_pixels = self._decode_to_cpu(view_latents)

            all_view_pixels.append(view_pixels)

            del prev_latents
            prev_latents = view_latents
            prev_pixels = view_pixels

            if accumulate_views and use_cutt3r:
                accumulated_pixels.append(prev_pixels)

        del prev_latents
        torch.cuda.empty_cache()

        min_frames = min(arr.shape[0] for arr in all_view_pixels)
        all_view_pixels = [arr[:min_frames] for arr in all_view_pixels]

        return torch.stack(all_view_pixels, dim=0)  # on CPU

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
        return torch.cat(pixel_chunks, dim=0)