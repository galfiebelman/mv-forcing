# Derived from Self-Forcing (https://github.com/guandeh17/Self-Forcing, commit 33593df), licensed under the
# Apache License 2.0. Modified for MV-Forcing.
# See NOTICE.md and licenses/LICENSE.Apache-2.0.

import numpy as np
import random
import torch

def load_mixed_state_dict(model, teacher_state_dict, student_state_dict):
    """
    Load temporal modules from student and view sync modules from teacher.

    Args:
        model: Target model to load weights into
        teacher_state_dict: Contains pretrained view sync weights
        student_state_dict: Contains pretrained temporal weights
    """
    # Define view sync module names (based on WanMVAttentionBlock)
    view_sync_keywords = [
        'mvs_attn',  # Multi-view self-attention
        'cam_encoder',  # Camera embedding encoder
        'projector',  # View projection layer
        'norm_mvs',  # MVS normalization
        'modulation_mvs',  # MVS modulation parameters
        "rendered_embedding"
    ]

    # Create merged state dict
    merged_state_dict = {}

    # First, load all temporal parameters from student
    for key, value in student_state_dict.items():
        # Check if this is a view sync parameter
        is_view_sync = any(keyword in key for keyword in view_sync_keywords)

        if not is_view_sync:
            merged_state_dict[key] = value

    # Then, override with view sync parameters from teacher
    for key, value in teacher_state_dict.items():
        # Check if this is a view sync parameter
        is_view_sync = any(keyword in key for keyword in view_sync_keywords)

        if is_view_sync:
            merged_state_dict["model."+key] = value

    # Load the merged state dict
    missing_keys, unexpected_keys = model.load_state_dict(merged_state_dict, strict=False)

    if len(missing_keys) > 0:
        print(f"\nMissing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    return model


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict
