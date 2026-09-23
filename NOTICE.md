# Third-party code and licenses

MV-Forcing builds on the following third-party work. The Licensor's own code is
licensed under [`LICENSE`](./LICENSE) (PolyForm Noncommercial 1.0.0) subject to
[`ADDITIONAL_TERMS.md`](./ADDITIONAL_TERMS.md); everything listed here remains
under its own license.

## Fork base: Self-Forcing (Apache-2.0)

This repository is derived from **Self-Forcing**.
Source: https://github.com/guandeh17/Self-Forcing (commit `33593df`).
Licensed under the Apache License 2.0; a copy of that license is in
[`licenses/LICENSE.Apache-2.0`](./licenses/LICENSE.Apache-2.0).

Self-Forcing in turn vendors **Wan2.1** (https://github.com/Wan-Video/Wan2.1,
Apache-2.0) as the `wan/` package. The Wan2.1 and Self-Forcing model weights are
not redistributed here; users download them from their original releases.

As required by Apache-2.0 §4(b), the files inherited from Self-Forcing are
listed below, separated into those carried over unchanged and those modified for
MV-Forcing. Modified files additionally carry a notice in their header.

**Inherited and modified for MV-Forcing:**

```
train.py
model/__init__.py
model/base.py
model/ode_regression.py
pipeline/__init__.py
trainer/__init__.py
trainer/ode.py
utils/misc.py
utils/scheduler.py
utils/wan_wrapper.py
wan/__init__.py
wan/modules/model.py
wan/utils/__init__.py
```

**Inherited unchanged:**

```
utils/dataset.py
utils/distributed.py
utils/lmdb.py
utils/loss.py
utils/memory.py                 (Self-Forcing: demo_utils/memory.py; see FramePack below)
wan/README.md
wan/modules/__init__.py
wan/modules/attention.py
wan/modules/causal_model.py
wan/modules/t5.py
wan/modules/tokenizers.py
wan/modules/vae.py
wan/utils/fm_solvers_unipc.py
```

**FramePack** — `utils/memory.py` (GPU memory helpers and `DynamicSwapInstaller`)
was copied by Self-Forcing from https://github.com/lllyasviel/FramePack
(`demo_utils/`), Apache-2.0, and is inherited here unchanged; the original
attribution is retained in the file header.

Three files are new in this repository but substantially adapted from Wan2.1's
`wan/modules/model.py` and Self-Forcing's `wan/modules/causal_model.py`, and
carry the same Apache-2.0 attribution in their headers:

```
wan/modules/causal_mv_bidir_model.py
wan/modules/causal_mv_model.py
wan/modules/causal_model_DS.py
```

## Required at run time, not redistributed

- **CUT3R** — https://github.com/CUT3R/CUT3R, licensed under
  **CC BY-NC-SA 4.0**. Used as the recurrent 3D reconstruction model behind the
  4D-grounded geometric prior, at both training and inference time. Neither its
  code nor its weights (`cut3r_512_dpt_4_64.pth`) are included here; the user
  supplies them via `--cut3r_root`. Because CUT3R is required to run
  MV-Forcing, its noncommercial terms apply to the pipeline as a whole,
  independently of this repository's own license.
- **Wan2.1-T2V-1.3B** weights — https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
  (Apache-2.0). Backbone, 3D VAE and umT5 text encoder.
- **Self-Forcing DMD checkpoint** (`self_forcing_dmd.pt`) —
  https://huggingface.co/gdhe17/Self-Forcing (Apache-2.0). Supplies the frozen
  causal temporal layers and generates the first view in the real-world setting.
- **SynCamMaster** — https://github.com/KlingAIResearch/SynCamMaster; weights at
  https://huggingface.co/KlingTeam/SynCamMaster-Wan2.1 (Apache-2.0). The
  bidirectional teacher for synthetic distillation, the initialization of the
  multi-view synchronization layers, and the source of the camera rig
  (`example_test_data/cameras/camera_extrinsics.json`) used for all camera
  configurations.
- **ReCamMaster** — https://github.com/KlingAIResearch/ReCamMaster (MIT); weights
  at https://huggingface.co/KlingTeam/ReCamMaster-Wan2.1 (Apache-2.0). The
  teacher for the real-world stage.

## Datasets

- **SynCamVideo Dataset** — released with SynCamMaster; used for synthetic
  training and for the camera pairs in real-world preprocessing.
- **Open-Sora Dataset**, MixKit subset —
  https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.1.0; used for
  real-world finetuning.

Both are obtained by the user under their own terms and are not redistributed.

## Released model weights

The MV-Forcing checkpoint published at
https://huggingface.co/galfiebelman/mv-forcing is a derivative of the components
above: the Wan2.1-T2V-1.3B backbone, frozen causal temporal layers from
Self-Forcing, multi-view synchronization layers initialized from SynCamMaster and
finetuned here, and a zero-initialized geometric-prior projection trained here.
Distillation used SynCamMaster and ReCamMaster as score functions; no ReCamMaster
parameters are included. Those upstream components remain under their own
licenses (Apache-2.0); the Licensor's own trained contribution is released under
[`LICENSE`](./LICENSE) subject to
[`ADDITIONAL_TERMS.md`](./ADDITIONAL_TERMS.md).
