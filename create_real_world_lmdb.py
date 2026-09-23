"""
Aggregate precomputed MixKit .pt files into a single LMDB for training.

Reads per-sample .pt files from precompute_real_world.py and writes
a training-ready LMDB with keys matching ODEMVRegressionLMDBDataset:
  - latents:    (N, 21, 16, 60, 104)  float16
  - prompts:    (N, 512, 4096)         float16  (precomputed T5)
  - cams:       (N, 2, 21, 12)         float16
  - z_rendered: (N, 17, 21, 60, 104)   float16

Usage:
    python create_real_world_lmdb.py \
        --pt_dir /path/to/mixkit_precomputed_pts \
        --lmdb_path /path/to/mixkit_training_lmdb
"""

import argparse
import glob
import os
import numpy as np
import torch
import lmdb
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", type=str, required=True,
                        help="Directory containing precomputed .pt files")
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="Output LMDB path")
    parser.add_argument("--map_size_tb", type=float, default=1.0,
                        help="LMDB max map size in TB")
    args = parser.parse_args()

    # ── Gather .pt files ──
    pt_files = sorted(glob.glob(os.path.join(args.pt_dir, "*.pt")))
    print(f"Found {len(pt_files)} .pt files")
    assert len(pt_files) > 0, f"No .pt files in {args.pt_dir}"

    # ── Peek at first file for shapes ──
    sample = torch.load(pt_files[0], map_location="cpu", weights_only=True)
    latent_shape = sample['latents'].shape       # (21, 16, 60, 104)
    prompt_shape = sample['prompt_emb'].shape     # (512, 4096)
    camera_shape = sample['camera'].shape         # (2, 21, 12)
    z_rendered_shape = sample['z_rendered'].shape  # (17, 21, 60, 104)
    print(f"Per-sample shapes:")
    print(f"  latents:    {latent_shape}")
    print(f"  prompts:    {prompt_shape}")
    print(f"  cams:       {camera_shape}")
    print(f"  z_rendered: {z_rendered_shape}")

    # ── Open LMDB ──
    map_size = int(args.map_size_tb * 1024 ** 4)
    env = lmdb.open(args.lmdb_path, map_size=map_size)

    counter = 0
    skipped = 0

    for pt_path in tqdm(pt_files, desc="Writing LMDB"):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)

            latent = data['latents'].half().numpy()
            prompt = data['prompt_emb'].half().numpy()
            camera = data['camera'].half().numpy()
            z_rendered = data['z_rendered'].half().numpy()

            # Sanity check shapes
            assert latent.shape == latent_shape, \
                f"Shape mismatch: latent {latent.shape} vs {latent_shape}"

            with env.begin(write=True) as txn:
                txn.put(f'latents_{counter}_data'.encode(), latent.tobytes())
                txn.put(f'prompts_{counter}_data'.encode(), prompt.tobytes())
                txn.put(f'cams_{counter}_data'.encode(), camera.tobytes())
                txn.put(f'z_rendered_{counter}_data'.encode(), z_rendered.tobytes())

            counter += 1

        except Exception as e:
            print(f"Skipping {pt_path}: {e}")
            skipped += 1
            continue

    # ── Write shapes (N, ...) ──
    with env.begin(write=True) as txn:
        for key, shape in [
            ("latents", (counter,) + latent_shape),
            ("prompts", (counter,) + prompt_shape),
            ("cams", (counter,) + camera_shape),
            ("z_rendered", (counter,) + z_rendered_shape),
        ]:
            shape_str = " ".join(map(str, shape))
            txn.put(f"{key}_shape".encode(), shape_str.encode())

    env.close()
    print(f"\nDone. Wrote {counter} samples to {args.lmdb_path}")
    if skipped > 0:
        print(f"Skipped {skipped} files due to errors")


if __name__ == "__main__":
    main()