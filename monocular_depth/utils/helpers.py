import os
import sys
import numpy as np
import torch

from tqdm import tqdm
from monocular_depth.models.apple.config import INPUT_SIZE

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def target_transform(depth):
    # Resize the depth map to match input size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(0).unsqueeze(0), 
        size=INPUT_SIZE, 
        mode='bilinear', 
        align_corners=True
    ).squeeze()
    
    # Add channel dimension to match model output
    depth = depth.unsqueeze(0)
    return depth

def custom_collate_fn(batch):
    # handle None values in a batch
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def print_tqdm(message):
    tqdm.write(message, file=sys.stderr)
