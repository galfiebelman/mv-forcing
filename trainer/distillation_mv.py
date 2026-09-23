import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset_mv import TensorDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMDMV
import torch
import wandb
import time
import os
from einops import rearrange
from datetime import timedelta, datetime


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = False#:config.disable_wandb
        self.output_path = config.logdir
        date = str(datetime.now()).replace(" ", "-").replace(":", "-")
        self.output_path = os.path.join(
            self.output_path,
            f"{date}_seed{config.seed}"
        )
        self.wandb_save_dir = os.path.join(
            self.output_path,
            "wandb"
        )

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="offline",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=self.wandb_save_dir
            )


        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd_mv":
            self.model = DMDMV(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        if config.mv_only:
            for name, param in self.model.generator.model.named_parameters():
                if any(keyword in name for keyword in ["cam_encoder", "projector", "mvs_attn", "modulation_mvs","rendered_embedding"]):
                    param.requires_grad = True

        # for name, param in self.model.fake_score.model.named_parameters():
        #     if any(keyword in name for keyword in ["cam_encoder", "projector", "mvs_attn", "modulation_mvs"]):
        #         param.requires_grad = True

            trainable_params = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.generator.parameters())
            print(f"Generator: {trainable_params:,} / {total_params:,} trainable")
            print(f"Trainable ratio before gen: {trainable_params / total_params:.2%}")

        # trainable_params = sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in self.model.fake_score.parameters())
        # print(f"Fake score: {trainable_params:,} / {total_params:,} trainable")
        # print(f"Trainable ratio before fake: {trainable_params / total_params:.2%}")

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy
        )

        # self.model.text_encoder = fsdp_wrap(
        #     self.model.text_encoder,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        #     cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        # )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.cpu()
            # self.model.vae = self.model.vae.to(
            #     device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # trainable_after = sum(p.numel() for p in self.model.generator.parameters() if p.requires_grad)
        # total_after = sum(p.numel() for p in self.model.generator.parameters())
        # print(f"After FSDP Gen: {trainable_after:,} / {total_after:,} trainable (sharded)")
        # print(f"Trainable ratio after gen: {trainable_after / total_after:.2%}")

        # trainable_after = sum(p.numel() for p in self.model.fake_score.parameters() if p.requires_grad)
        # total_after = sum(p.numel() for p in self.model.fake_score.parameters())
        # print(f"After FSDP fake: {trainable_after:,} / {total_after:,} trainable (sharded)")
        #
        # print(f"Trainable ratio after fake: {trainable_after / total_after:.2%}")

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = TensorDataset(
                config.data_path,
                cutt3r_method=getattr(config, "cutt3r_method", None),
                num_generated_views=getattr(config, "num_generated_views", 1))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler)#,
            #num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(
                state_dict, strict=True
            )

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        prompt_emb_tensor = batch["prompt_emb"]["context"].to(
            device=self.device, dtype=self.dtype)
        prompt_emb_tensor = rearrange(prompt_emb_tensor, 'b v ... -> (b v) ...')
        cam_emb = batch["camera"].to(device=self.device, dtype=self.dtype)
        cam_emb = rearrange(cam_emb, 'b v ... -> (b v) ...')
        # View-sequential unrolling needs the next transition's cameras.
        cam_emb_next = batch.get("camera_next", None)
        if cam_emb_next is not None:
            cam_emb_next = rearrange(
                cam_emb_next.to(device=self.device, dtype=self.dtype), 'b v ... -> (b v) ...')
            c2w_src_next = batch["c2w_src_next"][0].to(device=self.device)
            c2w_tgt_next = batch["c2w_tgt_next"][0].to(device=self.device)
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None
        origin_latents = batch["latents"].to(device=self.device, dtype=self.dtype)  # [B, 16, 21, 60, 104]
        origin_latents = rearrange(origin_latents, 'b v ... -> (b v) ...').permute(0,2,1,3,4)
        z_rendered = batch.get("z_rendered", None)
        if z_rendered is not None:
            z_rendered_clean = torch.zeros_like(z_rendered)  # v_{k-2}: zeros
            z_rendered = torch.cat([z_rendered_clean, z_rendered], dim=0)
        # batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        # image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = {
                "prompt_embeds": prompt_emb_tensor,
                "cam_emb": cam_emb,
                "origin_latents": origin_latents,  # GT view 0 for pipeline conditioning
                "z_rendered": z_rendered
            }
            if cam_emb_next is not None:
                conditional_dict.update({
                    "cam_emb_next": cam_emb_next,
                    "c2w_src_next": c2w_src_next,
                    "c2w_tgt_next": c2w_tgt_next,
                })
            uncond_template = self.model.unconditional_dict_template
            current_batch_size = conditional_dict["prompt_embeds"].shape[0]
            unconditional_dict = {
                k: v.repeat(current_batch_size, 1, 1).to(self.device, dtype=self.dtype)
                for k, v in uncond_template.items()
            }
            unconditional_dict["cam_emb"] = cam_emb
            # if not getattr(self, "unconditional_dict", None):
            #     unconditional_dict = self.model.text_encoder(
            #         text_prompts=[self.config.negative_prompt] * batch_size)
            #     unconditional_dict = {k: v.detach()
            #                           for k, v in unconditional_dict.items()}
            #     self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            # else:
            #     unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )
            torch.cuda.empty_cache()

            generator_loss.backward()
            torch.cuda.empty_cache()

            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        torch.cuda.empty_cache()

        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )
        torch.cuda.empty_cache()

        critic_loss.backward()
        torch.cuda.empty_cache()

        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        start_step = self.step
        while True:
            # print("step: {}".format(self.step))
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )
                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
