# MV-Forcing: Long Multi-View Video Generation via 4D-Grounded Spatio-Temporal Self-Forcing

### [Project Page](https://galfiebelman.github.io/mv-forcing/) | [ArXiv](https://arxiv.org/abs/2607.05376)

This is the official implementation of **MV-Forcing**.

![Generic badge](https://img.shields.io/badge/conf-ECCV%202026-blue.svg)

> **MV-Forcing: Long Multi-View Video Generation via 4D-Grounded Spatio-Temporal Self-Forcing**<br>
> Gal Fiebelman<sup>1</sup>, Hadar Averbuch-Elor<sup>2</sup>, Sagie Benaim<sup>1</sup><br>
> <sup>1</sup>The Hebrew University of Jerusalem, <sup>2</sup>Cornell University

>**Abstract** <br>
> Recent advances in video diffusion models have enabled either long single-view generation through temporal autoregression, or short multi-view synthesis through bidirectional attention. However, generating long, multi-view consistent videos of dynamic scenes remains unsolved. In this work, we present MV-Forcing, a framework that composes temporal and view-wise autoregression within a single diffusion model by introducing a 4D geometric bridge between sequentially generated views. Our key insight is that an autoregressive 3D reconstruction model naturally interfaces between autoregressively generated views. Given a completed source view, we reconstruct its 3D structure and render a geometric prior of the next target viewpoint, which the diffusion model refines into a high-quality video. To extend generation beyond the teacher's fixed temporal window, we introduce a joint denoising regime where both view slots are initialized from noise during training, enabling temporally unbounded generation. We distill the model via Distribution Matching Distillation with Spatio-Temporal Self-Forcing, closing the train-inference exposure bias gap for both temporal and view-sequential autoregression. Extensive experiments on both synthetic and real-world data demonstrate that MV-Forcing produces geometrically consistent multi-view videos of dynamic scenes at arbitrary lengths and viewpoint counts using a single few-step student model.

</br>

# Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/galfiebelman/mv-forcing.git
cd mv-forcing
```
Run every entry point from the repository root: configs and default asset paths are
resolved relative to it.

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate mv-forcing
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Install flash-attention 2
This is required, not optional. A prebuilt wheel matching torch 2.5 / cu121 / py3.10
avoids a long compile:
```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 4. Set up CUT3R
MV-Forcing renders its geometric prior with [CUT3R](https://github.com/CUT3R/CUT3R).
Clone it next to this repository (or elsewhere and set `MVF_CUT3R_ROOT`), then download
its `cut3r_512_dpt_4_64.pth` checkpoint into `CUT3R/src/` following CUT3R's README:
```bash
git clone https://github.com/CUT3R/CUT3R.git
```

CUT3R's CUDA RoPE kernel **must** be compiled. CUT3R assigns its pose token a position
of `-1`, which the pure-PyTorch fallback cannot index, so without this step the prior
render fails with `IndexError: index out of range in self`:
```bash
conda install -c nvidia cuda-nvcc=12.1 cuda-cccl
export CUDA_HOME=$CONDA_PREFIX
cd CUT3R/src/croco/models/curope && python setup.py build_ext --inplace && cd -
```

### 5. Download the base models
```bash
# Wan2.1-T2V-1.3B backbone (expected at ./wan_models, or set MVF_WAN_ROOT)
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B

# Self-Forcing, which generates the first view at inference
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```
Inference also reads a SynCamMaster camera rig JSON. Clone
[SynCamMaster](https://github.com/KwaiVGI/SynCamMaster) beside this repository so that
`SynCamMaster/example_test_data/cameras/camera_extrinsics.json` exists, or pass your own
file with `--camera_json_path`.

### 6. Download the MV-Forcing weights
```bash
hf download galfiebelman/mv-forcing mv_forcing_real_world.pt --local-dir weights
```

Expected layout:
```
mv-forcing/
├── weights/mv_forcing_real_world.pt
├── checkpoints/self_forcing_dmd.pt
├── wan_models/Wan2.1-T2V-1.3B/
├── CUT3R/src/cut3r_512_dpt_4_64.pth
└── SynCamMaster/example_test_data/cameras/camera_extrinsics.json
```

</br>

# Usage

Inference runs on a single GPU; no `torchrun` is needed. Generate two synchronized views
of 81 frames each:
```bash
python inference_real_world.py \
    --checkpoint_path weights/mv_forcing_real_world.pt \
    --output_folder outputs/demo \
    --prompt "A vintage red convertible driving along a coastal cliff road at sunset, waves breaking on rocks below" \
    --num_views 2 \
    --total_frames 21
```
`--total_frames` counts latent frames: 21 latent frames are 81 pixel frames at 480x832.
Each run writes `merged.mp4`, a `video.mp4` and PNG frames per view, and a
`metadata.json` recording the prompt and camera configuration.

To go beyond the 21-frame temporal window, enable sliding-window attention with
`--local_attn_size` **and** `--sink_size`. Three views of 165 frames each:
```bash
python inference_real_world.py \
    --checkpoint_path weights/mv_forcing_real_world.pt \
    --output_folder outputs/demo_long \
    --prompt "A vintage red convertible driving along a coastal cliff road at sunset, waves breaking on rocks below" \
    --num_views 3 \
    --total_frames 42 \
    --accumulate_views \
    --local_attn_size 21 \
    --sink_size 12
```
Passing `--local_attn_size` without `--sink_size` has no effect on the first view, and
the run then fails with a size-0 tensor once generation passes 21 latent frames.

Useful flags: `--cam_type {az,dis,el}` selects the camera transform applied at each view
step, `--cam_indices` gives an explicit camera chain instead, `--accumulate_views` feeds
every previous view into the CUT3R state rather than only the last, `--prompt_file` reads
one prompt per line, and `--seed` fixes sampling. On a 40 GB GPU, two views of 81 frames
take roughly 4 minutes and three views of 165 frames roughly 8 minutes.

</br>

# Training

Training runs in three stages: ODE initialization, DMD distillation on synthetic
multi-view data, and finetuning on real-world video. Each stage reads one config from
`configs/` and is launched with `torchrun`. The DMD stages hold three 1.3B models
(generator, frozen teacher, trainable critic) plus optimizer states, and the synthetic
stage additionally keeps CUT3R resident, so they need **more than 40 GB per GPU** at the
paper's 21 latent frames.

Both stages that use synthetic data expect the
[SynCamVideo Dataset](https://huggingface.co/datasets/KwaiVGI/SynCamVideo-Dataset) and a
SynCamMaster teacher checkpoint at `checkpoints/syncammaster_step20000.ckpt`, obtained by
following [SynCamMaster](https://github.com/KwaiVGI/SynCamMaster)'s own instructions.
Teacher checkpoints are not redistributed here.

## Step 0: Prepare the data

**Synthetic.** Encode each view to latents, choose valid view pairs, and render a CUT3R
geometric prior per pair:
```bash
python precompute_synthetic.py \
    --dataset_dir data/SynCamVideo-Dataset/train/f24_aperture5 \
    --captions captions.csv
```
Pairs are kept when the two cameras differ by less than `--az_diff_max` (30°) in azimuth
and `--el_diff_max` (15°) in elevation, measured about the point the rig looks at. The
kept pairs form closed chains over distinct views, which view-sequential unrolling
requires. `--captions` is optional: it takes a CSV with `scene,text` columns, and any
scene missing from it falls back to `--default_caption`.

**Real-world.** The real-world stage trains on MixKit clips, taken from the latents LMDB
released with [CausVid](https://github.com/tianweiy/CausVid). Render priors for it and
write the training LMDB:
```bash
python precompute_real_world.py \
    --lmdb_path data/mixkit_latents_lmdb \
    --synccam_path SynCamMaster/example_test_data/cameras/camera_extrinsics.json \
    --output_dir data/real_world_pts \
    --shard_idx 0 --num_shards 1

python create_real_world_lmdb.py \
    --pt_dir data/real_world_pts \
    --lmdb_path data/real_world_lmdb
```

## Step 1: ODE initialization

The student is first regressed onto trajectories produced by the bidirectional teacher.
Generate the ODE pairs, pack them into an LMDB, then train:
```bash
python generate_ode_pairs.py \
    --ds_path data/SynCamVideo-Dataset/train \
    --output_folder data/ode_pairs \
    --num_pairs 1000 \
    --guidance_scale 5.0 \
    --chunk_id 0 --num_chunks 1

python create_ode_lmdb.py \
    --data_path data/ode_pairs \
    --lmdb_path data/ode_lmdb

torchrun --nproc_per_node 8 train.py --config_path configs/train_ode_synthetic.yaml
```
Each pair is a full teacher sampling run and takes several minutes, so `--chunk_id` /
`--num_chunks` let you shard generation across machines. This stage runs for 3,000
iterations.

## Step 2: DMD distillation on synthetic data

Distribution Matching Distillation with Spatio-Temporal Self-Forcing, for 1,600
iterations. `num_generated_views: 2` turns on view-sequential unrolling, where the second
view is conditioned on the student's own detached output with its geometric prior
rendered online:
```bash
torchrun --nproc_per_node 8 train.py --config_path configs/train_dmd_synthetic.yaml
```
Set `generator_ckpt` in the config to the checkpoint from Step 1.

## Step 3: Real-world finetuning

500 iterations against a [ReCamMaster](https://github.com/KwaiVGI/ReCamMaster) teacher
finetuned for non-overlapping viewpoint pairs, expected at
`checkpoints/recam_teacher.ckpt`. That teacher drifts between separate
trajectory generations, so this stage supervises a single view transition
(`num_generated_views: 1`) and disables joint denoising (`p_both_noisy: 0.0`):
```bash
torchrun --nproc_per_node 8 train.py --config_path configs/train_dmd_real_world.yaml
```
Set `generator_ckpt` to the checkpoint from Step 2. This produces the released weights.

</br>

# Configurations

`configs/base.yaml` holds the shared defaults; each stage config layers on top of it.
`configs/train_ode_synthetic.yaml`, `configs/train_dmd_synthetic.yaml` and
`configs/train_dmd_real_world.yaml` correspond to the three training steps, and
`configs/inference_real_world.yaml` to inference. Checkpoint paths, `data_path`, and the
iteration counts are set there rather than on the command line. Leaving `wandb_key` empty
disables Weights & Biases, in which case losses are printed to stdout.

</br>

# License

The code and released weights are licensed under
[PolyForm Noncommercial 1.0.0](./LICENSE) together with
[`ADDITIONAL_TERMS.md`](./ADDITIONAL_TERMS.md), which restrict use to noncommercial
purposes and prohibit military and surveillance use. For commercial licensing, contact
galf251@gmail.com. Note that MV-Forcing requires CUT3R at both training and inference
time, and CUT3R is licensed CC BY-NC-SA 4.0, so a commercial license for MV-Forcing does
not by itself permit commercial use of the full pipeline. Third-party code and its
licenses are inventoried in [`NOTICE.md`](./NOTICE.md).

</br>

# BibTeX

If you find our work useful in your research, please consider citing:

```bibtex
@misc{fiebelman2026mvforcinglongmultiview,
    title={MV-Forcing: Long Multi-View Video Generation via 4D-Grounded Spatio-Temporal Self-Forcing},
    author={Gal Fiebelman and Hadar Averbuch-Elor and Sagie Benaim},
    year={2026},
    eprint={2607.05376},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2607.05376},
}
```

</br>

# Acknowledgements

Built on [Self-Forcing](https://github.com/guandeh17/Self-Forcing),
[CUT3R](https://github.com/CUT3R/CUT3R),
[SynCamMaster](https://github.com/KwaiVGI/SynCamMaster),
[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster),
[Wan2.1](https://github.com/Wan-Video/Wan2.1),
[CausVid](https://github.com/tianweiy/CausVid), and
[DMD2](https://github.com/tianweiy/DMD2). This research was supported by The Israel
Science Foundation (grant No. 2416/25) and EuroHPC JU (Application ID
EHPC-DEV-2025D08-098).
