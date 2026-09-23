# from sympy.benchmarks.bench_discrete_log import data_set_1
import os
from utils.dataset_mv import TensorODEDataset
from utils.wan_wrapper import WanMVDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.distributed import launch_distributed_job
from utils.paths import set_wan_root
import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
# from einops import rearrange

def init_model(device, teacher_ckpt):
    model = WanMVDiffusionWrapper(
            is_causal=False
        ).to(device).to(torch.float32)
    teacher_state_dict = torch.load(teacher_ckpt, map_location="cpu")
    model.model.load_state_dict(
        teacher_state_dict, strict=True
    )
    encoder = WanTextEncoder().to(device).to(torch.float32)
    model.requires_grad_(False)

    scheduler = model.get_scheduler()
    scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0)
    scheduler.sigmas = scheduler.sigmas.to(device)

    sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

    unconditional_dict = encoder(
        text_prompts=[sample_neg_prompt]
    )

    return model, scheduler, unconditional_dict


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--chunk_id", type=int, default=0, help="ID of this computer (0 to 3)")
    parser.add_argument("--num_chunks", type=int, default=8, help="Total number of computers (e.g., 4)")
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--ds_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--teacher_ckpt", type=str,
                        default="checkpoints/syncammaster_step20000.ckpt",
                        help="Bidirectional SynCamMaster teacher checkpoint")
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="Number of ODE solution pairs to generate in total "
                             "(across all chunks); -1 for the whole dataset")
    parser.add_argument("--wan_root", type=str, default=None,
                        help="Directory holding Wan2.1-T2V-1.3B (defaults to ./wan_models)")

    args = parser.parse_args()
    set_wan_root(args.wan_root)

    # launch_distributed_job()
    # global_rank = dist.get_rank()
    #
    device = torch.cuda.current_device()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, scheduler, init_unconditional_dict = init_model(device=device, teacher_ckpt=args.teacher_ckpt)
    guidance_scale = args.guidance_scale
    dataset = TensorODEDataset(args.ds_path)
    num_pairs = len(dataset) if args.num_pairs < 0 else min(args.num_pairs, len(dataset))
    print(f"Generating {num_pairs} ODE pairs (dataset holds {len(dataset)})")

    # if global_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)

    with torch.no_grad():
        # for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
        #     prompt_index = index * dist.get_world_size() + dist.get_rank()
        for prompt_index in tqdm(range(args.chunk_id, num_pairs, args.num_chunks)):
            if prompt_index >= len(dataset):
                continue
            output_filepath = os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
            if os.path.exists(output_filepath):
                continue
            batch = dataset[prompt_index]
            scene = batch["scene"]  # e.g., "scene1255"
            source = batch["source_idx"]
            target = batch["target_idx"]
            torch.cuda.empty_cache()
            prompt_emb_tensor = batch["prompt_emb"]["context"].to(
                device=device, dtype=torch.float32)
            # prompt_emb_tensor = rearrange(prompt_emb_tensor, 'b v ... -> (b v) ...')
            cam_emb = batch["camera"].to(device=device, dtype=torch.float32)
            # cam_emb = rearrange(cam_emb, 'b v ... -> (b v) ...')
            latents = torch.randn(
                [2, 21, 16, 60, 104], dtype=torch.float32, device=device
            )
            batch_size = latents.shape[0]
            conditional_dict = {
                "prompt_embeds": prompt_emb_tensor,
                "cam_emb": cam_emb
            }
            unconditional_dict = {
                k: v.repeat(batch_size, 1, 1).to(latents.device, dtype=latents.dtype)
                for k, v in init_unconditional_dict.items()
            }
            unconditional_dict["cam_emb"] = cam_emb
            noisy_input = []
            cutt3r_raymap_path = os.path.join(args.ds_path, scene, "cutt3r_raymap", f"{source:02d}_{target:02d}.pth")

            for progress_id, t in enumerate(scheduler.timesteps):
                timestep = t * \
                    torch.ones([2, 21], device=device, dtype=torch.float32)
                # timestep = t.unsqueeze(0).to(dtype=latents.dtype, device=latents.device).repeat(batch_size,1)

                noisy_input.append(latents)

                flow_pred_cond, _ = model(
                    noisy_image_or_video=latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                )
                flow_pred_uncond, _ = model(
                    noisy_image_or_video=latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                )
                flow_pred = flow_pred_uncond + guidance_scale * (
                        flow_pred_cond - flow_pred_uncond)

                latents = scheduler.step(
                    flow_pred.unsqueeze(0), t.unsqueeze(0).cuda(), latents.unsqueeze(0),
                    # return_dict=False
                )[0].squeeze(0).to(latents.dtype)

            noisy_input.append(latents)

            noisy_inputs = torch.stack(noisy_input, dim=0)

            noisy_inputs = noisy_inputs[[0, 36, 44, -1],:]

            stored_data = noisy_inputs

            torch.save(
                {
                    "traj": stored_data.cpu().detach(),
                    "cams":cam_emb.cpu().detach(),
                    "prompt_emb":prompt_emb_tensor.cpu().detach(),
                    "cutt3r_raymap_path": cutt3r_raymap_path,
                },
                os.path.join(args.output_folder, f"{prompt_index:05d}.pt"),

            )

    dist.barrier()


if __name__ == "__main__":
    main()
