"""
MixKit LMDB Dataset for ReCamMaster inference on real-world videos.

Reads precomputed VAE latents and T5 prompt embeddings from CausVid's MixKit LMDB,
and generates camera embeddings from SynCamVideo camera extrinsics JSON.
"""
import random

from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
import os
from einops import rearrange


def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))
    return image_shape


def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    data_key = f'{array_name}_{row_index}_data'.encode()
    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)
    if dtype == str:
        return row_bytes.decode()
    array = np.frombuffer(row_bytes, dtype=dtype)
    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array


class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class MixKitLMDBDataset(Dataset):
    """
    Reads MixKit LMDB (latents + prompts) and pairs each video with
    a fixed camera trajectory from SynCamVideo extrinsics JSON.

    Returns per sample:
        - latents:    [F, C, H, W]  source video VAE latent
        - prompt_emb: [L, D]        T5 text embedding
        - camera:     [F, 12]       relative pose embedding for ReCamMaster
        - c2w_cond:   [F, 4, 4]     source camera c2w (for CUT3R, optional)
        - c2w_tgt:    [F, 4, 4]     target camera c2w (for CUT3R, optional)
        - idx:        int
    """

    def __init__(
        self,
        lmdb_path: str,
        camera_json_path: str,
        cam_type: str = "az",
    ):
        # ── LMDB ──
        self.env = lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.prompts_shape = get_array_shape_from_lmdb(self.env, 'prompts')
        self.num_samples = self.latents_shape[0]
        self.seen=set()
        print(f"LMDB: {self.num_samples} samples, "
              f"latents={self.latents_shape}, prompts={self.prompts_shape}")

        # ── Camera trajectories (precomputed once, shared across samples) ──
        self.cam_types = ["az", "el", "dis"]
        self.cameras = {}
        for ct in self.cam_types:
            self.cameras[ct] = self._load_cameras(camera_json_path, ct)

    def _load_cameras(self, camera_json_path, cam_type):
        """Load camera extrinsics and compute relative poses (same for all samples)."""
        with open(camera_json_path, 'r') as f:
            cam_data = json.load(f)

        # Camera pair selection (matches TextCameraDataset / inference_recammaster_cutt3r.py)
        if cam_type == "az":
            cond_idx, tgt_idx = 1, 3#9,2#
        elif cam_type == "el":
            cond_idx, tgt_idx = 7, 3
        elif cam_type == "dis":
            cond_idx, tgt_idx = 10, 9
        else:
            raise ValueError(f"Unknown cam_type: {cam_type}")

        cam_idx = list(range(81))[::4]  # 21 latent frames

        multiview_c2ws = []
        for view_idx in [cond_idx, tgt_idx]:
            traj = [self._parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"])
                    for idx in cam_idx]
            traj = np.stack(traj).transpose(0, 2, 1)
            c2ws = []
            for c2w in traj:
                c2w = c2w[:, [1, 2, 0, 3]]
                c2w[:3, 1] *= -1.
                c2w[:3, 3] /= 200
                c2ws.append(c2w)
            multiview_c2ws.append(c2ws)

        # Compute relative pose embedding (ReCamMaster convention: tgt relative to cond)
        cond_cam_params = [Camera(c) for c in multiview_c2ws[0]]
        tgt_cam_params = [Camera(c) for c in multiview_c2ws[1]]

        relative_poses = []
        for i in range(len(tgt_cam_params)):
            relative_pose = self._get_relative_pose(
                [cond_cam_params[i], tgt_cam_params[i]])
            relative_poses.append(
                torch.as_tensor(relative_pose)[:, :3, :])  # [2, 3, 4]

        pose_embedding = torch.stack(relative_poses, dim=1)  # [2, 21, 3, 4]
        camera_embedding = rearrange(
            pose_embedding, 'v f c d -> v f (c d)').to(torch.bfloat16)  # [2, 21, 12]

        # Store c2w matrices for optional CUT3R use later
        c2w_cond = torch.tensor(
            np.stack(multiview_c2ws[0]), dtype=torch.float32)  # [21, 4, 4]
        c2w_tgt = torch.tensor(
            np.stack(multiview_c2ws[1]), dtype=torch.float32)   # [21, 4, 4]
        return {
            "camera": camera_embedding,
            "c2w_cond": c2w_cond,
            "c2w_tgt": c2w_tgt,
        }

    @staticmethod
    def _parse_matrix(matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    @staticmethod
    def _get_relative_pose(cam_params):
        abs_w2cs = [c.w2c_mat for c in cam_params]
        abs_c2ws = [c.c2w_mat for c in cam_params]
        target_cam_c2w = np.eye(4, dtype=np.float32)
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w] + [abs2rel @ c for c in abs_c2ws[1:]]
        return np.array(ret_poses, dtype=np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        idx = random.randint(0, self.num_samples - 1)
        while idx in self.seen:
            idx = random.randint(0, self.num_samples - 1)
        self.seen.add(idx)
        # idx = [339, 400, 314, 743, 793, 770, 377, 784, 774][idx]
        # ── Latents: source video VAE encoding ──
        latents = retrieve_row_from_lmdb(
            self.env, "latents", np.float16, idx,
            shape=self.latents_shape[1:])
        latents = torch.tensor(latents, dtype=torch.float32)

        # Handle shape: LMDB stores (C, F, H, W) = (16, 21, 60, 104)
        # Pipeline expects (F, C, H, W) for consistency with baseline code
        if latents.dim() == 4 and latents.shape[0] == 16:
            latents = latents.permute(1, 0, 2, 3)  # → (F, C, H, W)

        # ── Prompt embeddings: precomputed T5 ──
        # prompt_emb = retrieve_row_from_lmdb(
        #     self.env, "prompts", np.float16, idx,
        #     shape=self.prompts_shape[1:])
        # prompt_emb = torch.tensor(prompt_emb, dtype=torch.float32)
        cam_type = "el"#random.choice(self.cam_types)
        cam_data = self.cameras[cam_type]

        prompt_text = retrieve_row_from_lmdb(
            self.env, "prompts", str, idx)

        return {
            "latents": latents,
            "prompt_text": prompt_text,
            "camera": cam_data["camera"].clone(),
            "c2w_cond": cam_data["c2w_cond"].clone(),
            "c2w_tgt": cam_data["c2w_tgt"].clone(),
            "cam_type": cam_type,
            "idx": idx,
        }


class MixKitTrainingLMDBDataset(Dataset):
    """
    Reads MixKit training LMDB (precomputed by create_real_world_lmdb.py).

    LMDB keys:
        latents:    (N, 21, 16, 60, 104)   float16  — source VAE latent (F, C, H, W)
        prompts:    (N, 512, 4096)          float16  — precomputed T5
        cams:       (N, 2, 21, 12)          float16  — relative camera embedding
        z_rendered: (N, 17, 21, 60, 104)    float16  — CUT3R raymap rendering

    Returns dict matching TensorDataset / distillation_mv.py conventions:
        latents:    (2, 16, 21, 60, 104)  — V0=source, V1=source (placeholder)
        prompt_emb: {"context": (2, 512, 4096)}
        camera:     (2, 21, 12)
        z_rendered: (17, 21, 60, 104)     — only for V1
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8),
                 cutt3r_method=None):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.prompts_shape = get_array_shape_from_lmdb(self.env, 'prompts')
        self.cameras_shape = get_array_shape_from_lmdb(self.env, 'cams')
        self.cutt3r_method = cutt3r_method

        if self.cutt3r_method is not None:
            self.z_rendered_shape = get_array_shape_from_lmdb(self.env, 'z_rendered')

        self.max_pair = max_pair
        print(f"MixKitTrainingLMDB: {self.latents_shape[0]} samples, "
              f"latents={self.latents_shape[1:]}, "
              f"prompts={self.prompts_shape[1:]}, "
              f"cams={self.cameras_shape[1:]}")

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        while True:
            try:
                # ── Latents: (F, C, H, W) = (21, 16, 60, 104) ──
                latent = retrieve_row_from_lmdb(
                    self.env, "latents", np.float16, idx,
                    shape=self.latents_shape[1:])
                latent = torch.tensor(latent, dtype=torch.float32)

                # Convert (F, C, H, W) → (C, F, H, W) to match TensorDataset
                latent = latent.permute(1, 0, 2, 3)  # (16, 21, 60, 104)

                # Stack as 2 views: V0=source, V1=source (placeholder, not used as GT)
                latents = torch.stack([latent, latent], dim=0)  # (2, 16, 21, 60, 104)

                # ── Prompt: (512, 4096) ──
                prompt = retrieve_row_from_lmdb(
                    self.env, "prompts", np.float16, idx,
                    shape=self.prompts_shape[1:])
                prompt = torch.tensor(prompt, dtype=torch.float32)
                # Repeat for 2 views and wrap in dict matching TensorDataset convention
                prompt_emb = {"context": prompt.unsqueeze(0).repeat(2, 1, 1)}  # (2, 512, 4096)

                # ── Camera: (2, 21, 12) ──
                camera = retrieve_row_from_lmdb(
                    self.env, "cams", np.float16, idx,
                    shape=self.cameras_shape[1:])
                camera = torch.tensor(camera, dtype=torch.float32)  # (2, 21, 12)

                result = {
                    "latents": latents,
                    "prompt_emb": prompt_emb,
                    "camera": camera,
                }

                # ── z_rendered: (17, 21, 60, 104) ──
                if self.cutt3r_method is not None:
                    z_rendered = retrieve_row_from_lmdb(
                        self.env, "z_rendered", np.float16, idx,
                        shape=self.z_rendered_shape[1:])
                    result["z_rendered"] = torch.tensor(z_rendered, dtype=torch.float32)

                return result

            except Exception as e:
                print(f"Error loading sample {idx}: {e}")
                idx = random.randrange(len(self))