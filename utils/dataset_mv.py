# from utils.lmdb import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os
from einops import rearrange
import pandas as pd
import random
import glob
import re

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


class TextCameraDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, args, num_frames=81, height=480, width=832, baseline_3=False, shuffle=False):
        metadata = pd.read_csv(metadata_path)
        self.base_path = base_path
        self.text = metadata["text"].to_list()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.args = args
        self.cam_type = self.args.cam_type
        self.baseline_3 = baseline_3
        self.shuffle = shuffle
        if self.shuffle:
            self.seen = set()

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def __getitem__(self, data_id):
        if self.shuffle:
            data_id = random.randint(0, len(self.text) - 1)
            while data_id in self.seen:
                data_id = random.randint(0, len(self.text) - 1)
            self.seen.add(data_id)
        text = self.text[data_id]
        data = {"text": text}
        tgt_camera_path = os.path.join(self.base_path,"cameras/camera_extrinsics.json")
        with open(tgt_camera_path, 'r') as file:
            cam_data = json.load(file)
        multiview_c2ws = []
        cam_idx = list(range(81))[::4]  # 21 frames
        if self.cam_type == "az":
            tgt_idx = 1
            cond_idx = 3
        elif self.cam_type == "el":
            tgt_idx = 3
            cond_idx = 7
        elif self.cam_type == "dis":
            tgt_idx = 9
            cond_idx = 10
        if not self.baseline_3:
            for view_idx in [cond_idx, tgt_idx]:
                traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                traj = np.stack(traj).transpose(0, 2, 1)
                c2ws = []
                for c2w in traj:
                    c2w = c2w[:, [1, 2, 0, 3]]
                    c2w[:3, 1] *= -1.
                    c2w[:3, 3] /= 200
                    c2ws.append(c2w)
                multiview_c2ws.append(c2ws)
            cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
            tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
            relative_poses = []
            for i in range(len(tgt_cam_params)):
                relative_pose = self.get_relative_pose([tgt_cam_params[i], cond_cam_params[i]])
                relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :])
            pose_embedding = torch.stack(relative_poses, dim=1)  # v,21,3,4
            pose_embedding = rearrange(pose_embedding, 'v f c d -> v f (c d)')
            data['camera'] = pose_embedding.to(torch.bfloat16)
            data["idx"] = data_id
            data['c2w_cond'] = torch.tensor(np.stack(multiview_c2ws[1]), dtype=torch.float32)  # (21, 4, 4)
            data['c2w_tgt'] = torch.tensor(np.stack(multiview_c2ws[0]), dtype=torch.float32)   # (21, 4, 4)
            return data
        else:
            for view_idx in [cond_idx, tgt_idx]:
                traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                traj = np.stack(traj).transpose(0, 2, 1)
                c2ws = []
                for c2w in traj:
                    c2w = c2w[:, [1, 2, 0, 3]]
                    c2w[:3, 1] *= -1.
                    c2w[:3, 3] /= 200
                    c2ws.append(c2w)
                multiview_c2ws.append(c2ws)
            cond_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
            tgt_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
            relative_poses = []
            for i in range(len(tgt_cam_params)):
                relative_pose = self.get_relative_pose([tgt_cam_params[i], cond_cam_params[i]])
                relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :])
            pose_embedding = torch.stack(relative_poses, dim=1)  # v,21,3,4
            double_relative_poses = []
            for i in range(len(tgt_cam_params)):
                single_rel = self.get_relative_pose([tgt_cam_params[i], cond_cam_params[i]])  # [2, 4, 4]
                R = single_rel[1]  # [4, 4] — the tgt->cond transform
                R2 = R @ R  # [4, 4] — apply it twice
                double_relative_poses.append(torch.as_tensor(R2)[:3, :].unsqueeze(0))  # [1, 3, 4]
            double_embedding = torch.stack(double_relative_poses, dim=1)  # [1, F, 3, 4]
            pose_embedding = torch.cat([pose_embedding, double_embedding], dim=0)  # [3, F, 12]
            pose_embedding = rearrange(pose_embedding, 'v f c d -> v f (c d)')
            data['camera'] = pose_embedding.to(torch.bfloat16)
            data["idx"] = data_id
            return data

    def __len__(self):
        return len(self.text)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, cutt3r_method=None):
        metadata = pd.read_csv(os.path.join(base_path, "metadata.csv"))
        self.path = [os.path.join(base_path, file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i.replace('.mp4', '.pth') for i in self.path]# if os.path.exists(i.replace('.mp4', '.pth'))]
        print(len(self.path), "tensors cached in metadata.")
        self.cutt3r_method = cutt3r_method  # None or "raymap"
        assert len(self.path) > 0

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def __getitem__(self, index):
        # Return:
        # data['latents']: torch.Size([v, 16, 21, 60, 104])
        # data['camera']: torch.Size([v, 21, 3, 4])
        # data['prompt_emb']["context"][0]: torch.Size([v, 512, 4096])
        while True:
            try:
                data = {}
                data_id = torch.randint(0, len(self.path), (1,))[0]
                data_id = (data_id + index) % len(self.path)  # For fixed seed.
                path = self.path[data_id]
                base_path = path.rsplit('/', 2)[0]
                availble_list = np.load(os.path.join(base_path, "final_pairs.npy"))
                # if index %2 == 0:
                view1_idx, view2_idx = random.choice(availble_list) #9,2 #
                # else:
                #     view1_idx, view2_idx = 2,9
                view1_path = re.sub(r'cam(\d+).pth', f'cam{view1_idx:02}.pth', path)#"../SynCamMaster/toy/f24_aperture5/scene1255/videos/cam09.pth" #
                view1_data = torch.load(view1_path, weights_only=True, map_location="cpu")
                view2_path = re.sub(r'cam(\d+).pth', f'cam{view2_idx:02}.pth', path)#"../SynCamMaster/toy/f24_aperture5/scene1255/videos/cam02.pth" #
                view2_data = torch.load(view2_path, weights_only=True, map_location="cpu")
                data['latents'] = torch.stack((view1_data['latents'],view2_data['latents']),dim=0)
                # data['latents'] = torch.stack((view1_data['latents'], view2_data['latents']), dim=0)
                data['prompt_emb'] = view1_data['prompt_emb']
                data['prompt_emb']['context'] = data['prompt_emb']['context'].repeat(2, 1, 1)
                data['image_emb'] = {}

                camera_path = os.path.join(base_path, "cameras", "camera_extrinsics.json")#"../SynCamMaster/toy/f24_aperture5/scene1255/cameras/camera_extrinsics.json"#
                with open(camera_path, 'r') as file:
                    cam_data = json.load(file)
                multiview_c2ws = []
                cam_idx = list(range(81))[::4]
                for view_idx in [view1_idx, view2_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        c2w[:3, 3] /= 200
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)
                view1_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                view2_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]
                relative_poses = []
                for i in range(len(view1_cam_params)):
                    relative_pose = self.get_relative_pose([view1_cam_params[i], view2_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :])
                pose_embedding = torch.stack(relative_poses, dim=1)  # v,21,3,4
                pose_embedding = rearrange(pose_embedding, 'v f c d -> v f (c d)')
                data['camera'] = pose_embedding.to(torch.bfloat16)

                data['view1_path'] = view1_path
                data['view2_path'] = view2_path

                # --- CUT3R rendered geometric prior ---
                if self.cutt3r_method is not None:
                    rendered_path = os.path.join(
                        base_path, f"cutt3r_{self.cutt3r_method}",
                        f"{view1_idx:02d}_{view2_idx:02d}.pth"
                    )
                    data["z_rendered"] = torch.load(
                        rendered_path, weights_only=True, map_location="cpu")["z_rendered"]

                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.path))
        return data

    def __len__(self):
        return len(self.path)


class TensorODEDataset(torch.utils.data.Dataset):
    def __init__(self, base_path):
        self.base_path = base_path
        # 1. Find all final_ode_pair files dynamically
        pair_files = glob.glob(os.path.join(base_path, "f24_aperture5/scene*", "final_ode_pair.npy"))

        self.samples = []
        for pf in pair_files:
            scene_dir = os.path.dirname(pf)
            scene_name = os.path.basename(scene_dir)
            pairs = np.load(pf)
            # If the script saved a 1D array [src, tgt], reshape/wrap it safely
            if pairs.ndim == 1:
                pairs = [pairs]
            for pair in pairs:
                self.samples.append({
                    'scene_name': os.path.join("f24_aperture5", scene_name),
                    'scene_dir': scene_dir,
                    'view1_idx': int(pair[0]),
                    'view2_idx': int(pair[1])
                })
        print(f"Loaded {len(self.samples)} ODE pairs.")
        assert len(self.samples) > 0

    def parse_matrix(self, matrix_str):
        rows = matrix_str.strip().split('] [')
        matrix = []
        for row in rows:
            row = row.replace('[', '').replace(']', '')
            matrix.append(list(map(float, row.split())))
        return np.array(matrix)

    def get_relative_pose(self, cam_params):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses

    def __getitem__(self, index):
        # Return:
        # data['latents']: torch.Size([v, 16, 21, 60, 104])
        # data['camera']: torch.Size([v, 21, 3, 4])
        # data['prompt_emb']["context"][0]: torch.Size([v, 512, 4096])
        while True:
            try:
                sample = self.samples[index]
                scene_name = sample['scene_name']
                scene_dir = sample['scene_dir']
                view1_idx = sample['view1_idx']
                view2_idx = sample['view2_idx']

                # Load view1 data directly from the scene directory
                view1_path = os.path.join(scene_dir, "videos", f"cam{view1_idx:02d}.pth")
                view1_data = torch.load(view1_path, weights_only=True, map_location="cpu")

                data = {}
                data['prompt_emb'] = view1_data['prompt_emb']
                data['prompt_emb']['context'] = data['prompt_emb']['context'].repeat(2, 1, 1)

                # 2. Add scene and indices to the returned dict
                data['scene'] = scene_name
                data['source_idx'] = view1_idx
                data['target_idx'] = view2_idx

                camera_path = os.path.join(scene_dir, "cameras", "camera_extrinsics.json")
                with open(camera_path, 'r') as file:
                    cam_data = json.load(file)

                multiview_c2ws = []
                cam_idx = list(range(81))[::4]
                for view_idx in [view1_idx, view2_idx]:
                    traj = [self.parse_matrix(cam_data[f"frame{idx}"][f"cam{view_idx:02d}"]) for idx in cam_idx]
                    traj = np.stack(traj).transpose(0, 2, 1)
                    c2ws = []
                    for c2w in traj:
                        c2w = c2w[:, [1, 2, 0, 3]]
                        c2w[:3, 1] *= -1.
                        c2w[:3, 3] /= 200
                        c2ws.append(c2w)
                    multiview_c2ws.append(c2ws)

                view1_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[0]]
                view2_cam_params = [Camera(cam_param) for cam_param in multiview_c2ws[1]]

                relative_poses = []
                for i in range(len(view1_cam_params)):
                    relative_pose = self.get_relative_pose([view1_cam_params[i], view2_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:, :3, :])

                pose_embedding = torch.stack(relative_poses, dim=1)
                pose_embedding = rearrange(pose_embedding, 'v f c d -> v f (c d)')
                data['camera'] = pose_embedding.to(torch.bfloat16)
                break
            except Exception as e:
                print(f"ERROR WHEN LOADING: {e}")
                index = random.randrange(len(self.samples))
        return data

    def __len__(self):
        return len(self.samples)



def cycle(dl):
    while True:
        for data in dl:
            yield data

def get_array_shape_from_lmdb(env, array_name):
    with env.begin() as txn:
        image_shape = txn.get(f"{array_name}_shape".encode()).decode()
        image_shape = tuple(map(int, image_shape.split()))
    return image_shape

def retrieve_row_from_lmdb(lmdb_env, array_name, dtype, row_index, shape=None):
    """
    Retrieve a specific row from a specific array in the LMDB.
    """
    data_key = f'{array_name}_{row_index}_data'.encode()

    with lmdb_env.begin() as txn:
        row_bytes = txn.get(data_key)

    if dtype == str:
        array = row_bytes.decode()
    else:
        array = np.frombuffer(row_bytes, dtype=dtype)

    if shape is not None and len(shape) > 0:
        array = array.reshape(shape)
    return array

class ODEMVRegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8), cutt3r_method=None):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.prompts_shape = get_array_shape_from_lmdb(self.env, 'prompts')
        self.cameras_shape = get_array_shape_from_lmdb(self.env, 'cams')
        self.max_pair = max_pair
        self.cutt3r_method = cutt3r_method

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", np.float16, idx, shape=self.prompts_shape[1:]
        )
        cameras = retrieve_row_from_lmdb(
            self.env,
            "cams", np.float16, idx, shape=self.cameras_shape[1:]
        )
        result = {
            "prompt_emb": torch.tensor(prompts, dtype=torch.float32),
            "latents": torch.tensor(latents, dtype=torch.float32),
            "camera": torch.tensor(cameras, dtype=torch.float32)
        }

        if self.cutt3r_method is not None:
            cutt3r_path = retrieve_row_from_lmdb(self.env, "cutt3r_raymap_path", str, idx)
            cutt3r_data = torch.load(cutt3r_path, map_location="cpu", weights_only=True)
            result["z_rendered"] = cutt3r_data["z_rendered"]

        return result