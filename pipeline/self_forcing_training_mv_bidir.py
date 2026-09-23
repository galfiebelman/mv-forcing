from utils.wan_wrapper import WanMVBidirDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipelineMVBidir:
    """
    Self-forcing training pipeline for CausalWanMVBidirModel.

    Both views are processed simultaneously (batch dim = num_views).
    Slot 0 = GT view 0 (clean, overwritten after noising).
    Slot 1 = noisy view 1 (self-forcing target).
    No view loop, no view KV cache.
    """

    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanMVBidirDiffusionWrapper,
                 num_frame_per_block=1,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 num_views=2,
                 p_both_noisy=0.0,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise

        self.kv_cache = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        self.num_views = num_views
        self.p_both_noisy = p_both_noisy

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randint(0, num_denoising_steps, (num_blocks,), device=device)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        dist.broadcast(indices, src=0)
        return indices.tolist()

    def generate_and_sync_grad_blocks(self, num_blocks, n_grad_blocks, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            grad_block_indices = torch.randperm(num_blocks, device=device)[:n_grad_blocks]
        else:
            grad_block_indices = torch.empty(n_grad_blocks, dtype=torch.long, device=device)
        dist.broadcast(grad_block_indices, src=0)
        return set(grad_block_indices.tolist())

    def _sync_both_noisy(self, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        val = torch.tensor(
            [1.0 if (rank == 0 and torch.rand(1).item() < self.p_both_noisy) else 0.0],
            device=device)
        if dist.is_initialized():
            dist.broadcast(val, src=0)
        return val.item() > 0.5

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        """
        Args:
            noise: [num_views, F, C, H, W] — noise for ALL views
                   (only view 1's noise is actually used as starting point)
            origin_latents: [num_views, F, C, H, W] — GT latents
                   (only view 0 is used as clean conditioning)
            conditional_dict: must contain 'prompt_embeds' [num_views, L, D]
                              and 'cam_emb' [num_views, F, 12]
        """
        num_views, num_frames, num_channels, height, width = noise.shape
        assert num_views == self.num_views

        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        num_output_frames = num_frames

        # Output buffer for all views
        output = torch.zeros(
            [num_views, num_output_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype
        )

        # Initialize single KV cache (batch_size = num_views)
        self._initialize_kv_cache(
            batch_size=num_views, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(
            batch_size=num_views, dtype=noise.dtype, device=noise.device)

        # Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device)
        grad_block_indices = self.generate_and_sync_grad_blocks(
            len(all_num_frames), 3, device=noise.device)
        both_noisy = self._sync_both_noisy(noise.device)

        prev_v0_denoised = None
        current_start_frame = 0
        origin_latents = conditional_dict["origin_latents"]
        camera_embeddings = conditional_dict["cam_emb"]
        z_rendered_full = conditional_dict.get("z_rendered")
        for block_index, current_num_frames in enumerate(all_num_frames):
            frame_start = current_start_frame
            frame_end = current_start_frame + current_num_frames

            # View 1 noise for this block
            noisy_view1 = noise[1:2, frame_start:frame_end]  # [1, F_block, C, H, W]

            # GT view 0 for this block (will be injected as clean)
            gt_view0 = origin_latents[0:1, frame_start:frame_end]  # [1, F_block, C, H, W]
            if both_noisy:
                v0_input = noise[0:1, frame_start:frame_end]
            else:
                v0_input = gt_view0
            conditional_dict["cam_emb"] = camera_embeddings[:, frame_start:frame_end]
            if z_rendered_full is not None:
                if both_noisy:
                    conditional_dict["z_rendered"] = None
                else:
                    conditional_dict["z_rendered"] = z_rendered_full[:, :,frame_start:frame_end]
            # Denoising loop for this temporal block
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                # Shared timestep for both views
                timestep = torch.ones(
                    [num_views, current_num_frames],
                    device=noise.device, dtype=torch.int64
                ) * current_timestep

                # Stack [GT_view0, noisy_view1] as batch=2
                # But first: noise GT_view0 at same timestep, then overwrite with clean
                # This way the model sees clean slot 0 with timestep t (matches SynCamMaster training)
                # noisy_input = torch.cat([gt_view0, noisy_view1], dim=0)  # [2, F_block, C, H, W]
                noisy_input = torch.cat([v0_input, noisy_view1], dim=0)
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                        # Only keep view 1 output
                        view1_denoised = denoised_pred[1:2]
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_view1 = self.scheduler.add_noise(
                            view1_denoised.flatten(0, 1),
                            torch.randn_like(view1_denoised.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, view1_denoised.shape[:2])
                        if both_noisy:
                            view0_denoised = denoised_pred[0:1]
                            v0_input = self.scheduler.add_noise(
                                view0_denoised.flatten(0, 1),
                                torch.randn_like(view0_denoised.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, view0_denoised.shape[:2])
                else:
                    # Gradient block
                    if block_index not in grad_block_indices:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                    if both_noisy:
                        view1_denoised = denoised_pred[1:2]
                        view0_denoised = denoised_pred[0:1]
                    else:
                        view1_denoised = denoised_pred[1:2]
                    break

            # Record outputs
            if both_noisy:
                output[0, frame_start:frame_end] = view0_denoised[0]  # GT view 0 passthrough
            else:
                output[0, frame_start:frame_end] = gt_view0[0]
            output[1, frame_start:frame_end] = view1_denoised[0]  # Student view 1

            # Re-run with context noise to update KV cache for BOTH slots
            context_timestep_val = self.context_noise
            context_timestep = torch.ones(
                [num_views, current_num_frames],
                device=noise.device, dtype=torch.int64
            ) * context_timestep_val
            if both_noisy:
                cache_input = torch.cat([view0_denoised, view1_denoised], dim=0)
            else:
                cache_input = torch.cat([gt_view0, view1_denoised], dim=0)  # [2, F_block, C, H, W]

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=cache_input,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            current_start_frame += current_num_frames

        # Return timestep info
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1]).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1, both_noisy
        return output, denoised_timestep_from, denoised_timestep_to, both_noisy

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Single KV cache with batch_size = num_views."""
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache = kv_cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache