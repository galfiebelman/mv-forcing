from utils.wan_wrapper import WanMVBidirDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class SelfForcingTrainingPipelineMVBidir:
    """
    Spatio-temporal self-forcing training pipeline for CausalWanMVBidirModel.

    One pass processes two view slots at once (batch dim = view slot):
    slot 0 is the clean conditioning view, slot 1 is the view being generated
    with temporal self-forcing.

    View-sequential unrolling runs that pass ``num_generated_views`` times. The
    first pass conditions on the ground-truth view; every later pass conditions
    on the student's own detached output, with the geometric prior rendered
    online from it, which mirrors inference. The teacher then scores each
    consecutive generated pair.

    With joint view denoising (probability ``p_both_noisy``) both slots start
    from noise and there is no cross-view or geometric conditioning, so that
    branch trains first-view generation from text only and does not unroll.
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
                 num_generated_views=1,
                 render_prior=None,
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
        self.num_generated_views = num_generated_views
        # Callable (prev_latents, c2w_src, c2w_tgt) -> z_rendered, supplied by the
        # model, which owns the VAE and the frozen CUT3R.
        self.render_prior = render_prior

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
        # randperm can only yield num_blocks indices, so clamp before allocating:
        # every rank must broadcast the same shape or the collective deadlocks.
        n_grad_blocks = min(n_grad_blocks, num_blocks)
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
            noise: [num_views, F, C, H, W] — noise for both view slots
            origin_latents: [num_views, F, C, H, W] — ground-truth latents
                   (only view 0 is used, as clean conditioning)
            conditional_dict: 'prompt_embeds' [num_views, L, D],
                   'cam_emb' [num_views, F, 12], optionally 'z_rendered',
                   and for unrolling 'cam_emb_next', 'c2w_src_next', 'c2w_tgt_next'
        Returns:
            outputs: list of [num_views, F, C, H, W] tensors, one per generated
                     pair; element k holds (conditioning view, generated view)
        """
        num_views, num_frames, num_channels, height, width = noise.shape
        assert num_views == self.num_views
        assert num_frames % self.num_frame_per_block == 0

        num_blocks = num_frames // self.num_frame_per_block
        num_denoising_steps = len(self.denoising_step_list)

        # Every collective is drawn here, before any pass: a broadcast inside the
        # per-pass loop would have to interleave exactly with FSDP's own
        # collectives, and any skew between ranks deadlocks them.
        exit_flags = self.generate_and_sync_list(
            num_blocks, num_denoising_steps, device=noise.device)
        grad_block_indices = self.generate_and_sync_grad_blocks(
            num_blocks, 3, device=noise.device)
        both_noisy = self._sync_both_noisy(noise.device)

        outputs = [self._generate_pair(
            noise=noise,
            v0_latent=conditional_dict["origin_latents"][0:1],
            cam_emb=conditional_dict["cam_emb"],
            z_rendered_full=conditional_dict.get("z_rendered"),
            conditional_dict=conditional_dict,
            exit_flags=exit_flags,
            grad_block_indices=grad_block_indices,
            both_noisy=both_noisy,
        )]

        # View-sequential unrolling: condition on the student's own output.
        unroll = (not both_noisy) and self.num_generated_views > 1
        if unroll:
            if self.render_prior is None:
                raise ValueError(
                    "num_generated_views > 1 needs a render_prior callable to build "
                    "the geometric prior of each generated view")
            cam_emb_next = conditional_dict["cam_emb_next"]
            c2w_src_next = conditional_dict["c2w_src_next"]
            c2w_tgt_next = conditional_dict["c2w_tgt_next"]
            prev_latent = outputs[-1][1:2].detach()
            for _ in range(self.num_generated_views - 1):
                z_rendered_next = self.render_prior(
                    prev_latent, c2w_src_next, c2w_tgt_next)
                outputs.append(self._generate_pair(
                    noise=torch.randn_like(noise),
                    v0_latent=prev_latent,
                    cam_emb=cam_emb_next,
                    z_rendered_full=z_rendered_next,
                    conditional_dict=conditional_dict,
                    exit_flags=exit_flags,
                    grad_block_indices=grad_block_indices,
                    both_noisy=False,
                ))
                prev_latent = outputs[-1][1:2].detach()

        denoised_timestep_from, denoised_timestep_to = self._timestep_range(exit_flags)

        if return_sim_step:
            return outputs, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1, both_noisy
        return outputs, denoised_timestep_from, denoised_timestep_to, both_noisy

    def _generate_pair(
            self,
            noise: torch.Tensor,
            v0_latent: torch.Tensor,
            cam_emb: torch.Tensor,
            z_rendered_full: Optional[torch.Tensor],
            conditional_dict: dict,
            exit_flags: List[int],
            grad_block_indices: set,
            both_noisy: bool,
    ) -> torch.Tensor:
        """Generate one view (slot 1) conditioned on a clean slot 0.

        Args:
            v0_latent: [1, F, C, H, W] — clean conditioning view, either the
                ground-truth view 0 or the student's previous detached output
            cam_emb: [2, F, 12] — relative pose of this view transition
            z_rendered_full: [2, 17, F, h, w] geometric prior, or None
        """
        num_views, num_frames, num_channels, height, width = noise.shape
        num_blocks = num_frames // self.num_frame_per_block
        all_num_frames = [self.num_frame_per_block] * num_blocks

        output = torch.zeros(
            [num_views, num_frames, num_channels, height, width],
            device=noise.device, dtype=noise.dtype)

        # A pass is a fresh autoregressive rollout: caches must not carry over.
        self._initialize_kv_cache(
            batch_size=num_views, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(
            batch_size=num_views, dtype=noise.dtype, device=noise.device)

        # Per-pass view of the conditioning, so passes do not clobber each other.
        pass_dict = dict(conditional_dict)
        current_start_frame = 0

        for block_index, current_num_frames in enumerate(all_num_frames):
            frame_start = current_start_frame
            frame_end = current_start_frame + current_num_frames

            noisy_view1 = noise[1:2, frame_start:frame_end]
            gt_view0 = v0_latent[:, frame_start:frame_end]
            v0_input = noise[0:1, frame_start:frame_end] if both_noisy else gt_view0

            pass_dict["cam_emb"] = cam_emb[:, frame_start:frame_end]
            if z_rendered_full is not None:
                pass_dict["z_rendered"] = None if both_noisy else \
                    z_rendered_full[:, :, frame_start:frame_end]

            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])

                timestep = torch.ones(
                    [num_views, current_num_frames],
                    device=noise.device, dtype=torch.int64
                ) * current_timestep

                # Slot 0 stays clean while carrying timestep t, matching how the
                # bidirectional teacher was trained.
                noisy_input = torch.cat([v0_input, noisy_view1], dim=0)
                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=pass_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
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
                    # Gradients flow only through a few randomly chosen blocks.
                    if block_index not in grad_block_indices:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=pass_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=pass_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                    view1_denoised = denoised_pred[1:2]
                    if both_noisy:
                        view0_denoised = denoised_pred[0:1]
                    break

            if both_noisy:
                output[0, frame_start:frame_end] = view0_denoised[0]
            else:
                output[0, frame_start:frame_end] = gt_view0[0]
            output[1, frame_start:frame_end] = view1_denoised[0]

            # Re-run with context noise to update the KV cache for both slots.
            context_timestep = torch.ones(
                [num_views, current_num_frames],
                device=noise.device, dtype=torch.int64
            ) * self.context_noise
            if both_noisy:
                cache_input = torch.cat([view0_denoised, view1_denoised], dim=0)
            else:
                cache_input = torch.cat([gt_view0, view1_denoised], dim=0)

            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=cache_input,
                    conditional_dict=pass_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )

            current_start_frame += current_num_frames

        return output

    def _timestep_range(self, exit_flags):
        if not self.same_step_across_blocks:
            return None, None
        if exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1]).abs(), dim=0).item()
        denoised_timestep_from = 1000 - torch.argmin(
            (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]]).abs(), dim=0).item()
        return denoised_timestep_from, denoised_timestep_to

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
