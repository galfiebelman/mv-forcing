from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipelineMVBidir
from pipeline.cut3r_prior import (
    _postprocess_rendered_to_z, _render_raymap_frames, _run_cut3r_on_frames,
)
from utils.loss import get_denoising_loss
from utils.misc import load_mixed_state_dict
from utils.paths import load_cut3r
from utils.wan_wrapper import WanMVDiffusionWrapper, WanTextEncoder, WanVAEWrapper, WanMVBidirDiffusionWrapper, WanReCamDiffusionWrapper


class BaseMVModel(nn.Module):
    def __init__(self, args, device, teacher_recam=False):
        super().__init__()
        self.use_cut3r = getattr(args, "cutt3r_method", None) is not None
        self.teacher_recam = teacher_recam
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        # View-sequential unrolling: 1 = a single generated view, 2 = also unroll
        # to a second one conditioned on the student's own output.
        self.num_generated_views = getattr(args, "num_generated_views", 1)
        self.cut3r_model = None
        self.cut3r_size = getattr(args, "cut3r_size", 512)
        self.focal_ratio = getattr(args, "focal_ratio", 24.0 / 23.76)
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        # self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        # self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        text_encoder = WanTextEncoder().to(device)
        text_encoder.requires_grad_(False)
        with torch.no_grad():
            unconditional_dict = text_encoder(
                text_prompts=[args.negative_prompt])
            self.unconditional_dict_template = {
                k: v.detach().cpu() for k, v in unconditional_dict.items()
            }
        del text_encoder
        torch.cuda.empty_cache()
        print("Unconditional embeddings cached. Text encoder freed.")

        g_ckpt = getattr(args, "generator_ckpt", None)
        if g_ckpt is None:
            student_state_dict = torch.load(
                getattr(args, "self_forcing_ckpt", "checkpoints/self_forcing_dmd.pt"),
                map_location="cpu")["generator_ema"]
            teacher_state_dict = torch.load(args.teacher_ckpt, map_location="cpu")
            self.generator = WanMVBidirDiffusionWrapper(use_cut3r=self.use_cut3r)
            grad_req = True if not args.mv_only else False
            self.generator.model.requires_grad_(grad_req)
            self.generator = load_mixed_state_dict(
                    self.generator,
                    teacher_state_dict,
                    student_state_dict
                )
        else:
            student_sd = torch.load(g_ckpt, map_location="cpu")
            key = 'generator'#'generator_ema' if 'generator_ema' in student_sd else 'generator'
            self.generator = WanMVBidirDiffusionWrapper(use_cut3r=self.use_cut3r)
            grad_req = True if not args.mv_only else False
            self.generator.model.requires_grad_(grad_req)
            if key in student_sd:
                self.generator.load_state_dict(student_sd[key])
            else:
                self.generator.load_state_dict(student_sd)
            print(f"Student weights loaded (key='{key}').")

        if not self.teacher_recam:
            teacher_state_dict = torch.load(args.teacher_ckpt, map_location="cpu")
            self.real_score = WanMVDiffusionWrapper(is_causal=False)
            self.real_score.model.requires_grad_(False)
            self.real_score.model.load_state_dict(teacher_state_dict)

            self.fake_score = WanMVDiffusionWrapper(is_causal=False)
            self.fake_score.model.requires_grad_(True)
            self.fake_score.model.load_state_dict(teacher_state_dict)
        else:
            self.real_score = WanReCamDiffusionWrapper(use_cut3r=False)
            self.real_score.model.requires_grad_(False)
            self.fake_score = WanReCamDiffusionWrapper(use_cut3r=False)
            self.fake_score.model.requires_grad_(True)
            teacher_state_dict = torch.load(args.teacher_ckpt, map_location="cpu")

            self.real_score.model.load_state_dict(teacher_state_dict)
            self.fake_score.model.load_state_dict(teacher_state_dict)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(2, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModelMV(BaseMVModel):
    def __init__(self, args, device, teacher_recam=False):
        super().__init__(args, device, teacher_recam=teacher_recam)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        pred_pairs, denoised_timestep_from, denoised_timestep_to, both_noisy = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            **conditional_dict
        )

        pred_last_21, gradient_masks = [], []
        for pred_image_or_video in pred_pairs:
            # Slice last 21 frames
            if pred_image_or_video.shape[1] > 21:
                with torch.no_grad():
                    # Reencode to get image latent
                    latent_to_decode = pred_image_or_video[:, :-20, ...]
                    # Deccode to video
                    pixels = self.vae.decode_to_pixel(latent_to_decode)
                    frame = pixels[:, -1:, ...].to(self.dtype)
                    frame = rearrange(frame, "b t c h w -> b c t h w")
                    # Encode frame to get image latent
                    image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
                pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
            else:
                pred_image_or_video_last_21 = pred_image_or_video

            if num_generated_frames != min_num_frames:
                # Currently, we do not use gradient for the first chunk, since it contains image latents
                gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
                if self.args.independent_first_frame:
                    gradient_mask[:, :1] = False
                else:
                    gradient_mask[:, :self.num_frame_per_block] = False
            else:
                gradient_mask = None

            pred_last_21.append(pred_image_or_video_last_21.to(self.dtype))
            gradient_masks.append(gradient_mask)

        return pred_last_21, gradient_masks, denoised_timestep_from, denoised_timestep_to, both_noisy

    def _pair_conditional_dict(self, conditional_dict: dict, pair_index: int) -> dict:
        """Conditioning for the pair-th generated view transition.

        Later pairs are scored with the camera of their own transition. The
        teacher ignores the geometric prior, so it needs no substitution.
        """
        if pair_index == 0:
            return conditional_dict
        pair_dict = dict(conditional_dict)
        pair_dict["cam_emb"] = conditional_dict["cam_emb_next"]
        return pair_dict

    @torch.no_grad()
    def _render_prior(self, prev_latent, c2w_src, c2w_tgt):
        """Render the geometric prior of the next view from a generated view.

        Mirrors what inference does: decode the previous view, accumulate it in
        CUT3R's state, query the next view's raymap, and encode the render with
        its confidence map. The VAE is parked on CPU during training, so it is
        moved to the GPU only for this decode/encode and released afterwards.

        Args:
            prev_latent: [1, F, C, H, W] — the student's previous output
            c2w_src, c2w_tgt: [F, 4, 4] — cameras of this view transition
        Returns:
            z_rendered: [2, 17, F, H_lat, W_lat]; slot 0 is zeros, since the
            conditioning view needs no prior.
        """
        if self.cut3r_model is None:
            self.cut3r_model = load_cut3r(
                self.args.cut3r_path, self.device,
                root=getattr(self.args, "cut3r_root", None))

        self.vae.to(self.device)
        try:
            pixels = self.vae.decode_to_pixel(prev_latent)
            pixels = (pixels * 0.5 + 0.5).clamp(0, 1)[0]      # [T_pix, 3, H, W]
            self.vae.model.clear_cache()

            # One frame per latent frame, matching the camera sampling used when
            # the precomputed priors were produced.
            num_prior_frames = min(prev_latent.shape[1], c2w_src.shape[0])
            frames = pixels[::4][:num_prior_frames]
            c2w_src = c2w_src[:num_prior_frames]
            c2w_tgt = c2w_tgt[:num_prior_frames]
            outputs, state_args, frames_resized, H_c, W_c = _run_cut3r_on_frames(
                frames, self.cut3r_model, self.cut3r_size, self.device)
            H_out, W_out = frames.shape[2], frames.shape[3]

            rendered, conf = _render_raymap_frames(
                outputs, state_args, c2w_src, c2w_tgt, self.focal_ratio,
                list(range(frames.shape[0])), H_c, W_c, H_out, W_out,
                self.device, self.cut3r_model)
            conf_video = torch.stack(conf)
            masks = [((c - 1.0) / (conf_video.max() - 1.0 + 1e-6)).clamp(0, 1)
                     for c in conf]
            z_rendered = _postprocess_rendered_to_z(
                rendered, masks, self.vae, H_out, W_out, self.device)
            self.vae.model.clear_cache()
        finally:
            self.vae.cpu()
            torch.cuda.empty_cache()

        z_rendered = torch.cat([torch.zeros_like(z_rendered), z_rendered], dim=0)
        return z_rendered.to(dtype=self.dtype, device=self.device)

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict,
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = SelfForcingTrainingPipelineMVBidir(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            num_views=self.args.num_views,
            p_both_noisy=getattr(self.args, "p_both_noisy", 0.0),
            num_generated_views=self.num_generated_views,
            render_prior=self._render_prior if self.num_generated_views > 1 else None,
        )
