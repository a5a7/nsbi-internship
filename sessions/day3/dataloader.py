import torch
import h5py
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
import requests
import re
import os
from urllib.parse import urljoin
import numpy as np
from pathlib import Path


def collate_point_cloud(batch, max_part=150):
    """
    Collate function for point clouds and labels with truncation performed per batch.

    Args:
        batch (list of dicts): Each element is a dictionary with keys:
            - "X" (Tensor): Point cloud of shape (N, F)
            - "y" (Tensor): Label tensor
            - "cond" (optional, Tensor): Conditional info
            - "pid" (optional, Tensor): Particle IDs
            - "add_info" (optional, Tensor): Extra features

    Returns:
        Dict[str, torch.Tensor]: Dictionary containing collated tensors:
            - "X": (B, M, F) Truncated point clouds
            - "y": (B, num_classes)
            - "cond", "pid", "add_info" (optional, shape (B, M, ...))
    """
    batch_X = [item["X"] for item in batch]
    batch_y = [item["y"] for item in batch]

    # Stack once to avoid repeated slicing
    point_clouds = torch.stack(batch_X)  # (B, N, F)
    labels = torch.stack(batch_y)  # (B, num_classes)

    # Use validity mask based on feature index 2
    valid_mask = point_clouds[:, :, 2] != 0
    max_particles = min(valid_mask.sum(dim=1).max().item(), max_part)

    # Truncate point clouds
    truncated_X = point_clouds[:, :max_particles, :]  # (B, M, F)
    result = {"X": truncated_X, "y": labels}
    return result

class HEPDataset(Dataset):
    def __init__(
        self,
        file_path,            
        num_evt = -1,
        num_part = 30,
    ):
        """
        Args:
            file_paths (list): List of file paths.
        """
        self.file_path = file_path
        self.num_evt = num_evt
        self.num_part = num_part
        self._file_cache = self._get_file(self.file_path)

    def __len__(self):
        if self.num_evt > 0:
            return self.num_evt
        else:
            return len(self._file_cache['data'])

    def _get_file(self, file_path):
        self._file_cache = h5py.File(file_path, "r")
        return self._file_cache

    def __getitem__(self, idx):
        sample = {}

        sample["X"] = torch.tensor(self._file_cache["data"][idx][:self.num_part], dtype=torch.float32)
        sample["y"] = torch.tensor(self._file_cache["pid"][idx], dtype=torch.int64)
        return sample

    def __del__(self):
        # Clean up: close all cached file handles.
        for f in self._file_cache.values():
            try:
                f.close()
            except Exception as e:
                print(f"Error closing file: {e}")


def load_data(
    dataset_name,
    path,
    batch=100,
    dataset_type="train",
    num_evt = -1,
    num_workers=16,
    rank=0,
    size=1,
):
    names = [dataset_name]
    types = [dataset_type]

    dataset_path = os.path.join(path, dataset_name, dataset_type)
    h5_file = list(Path(dataset_path).glob("*.h5"))[0]

    data = HEPDataset(h5_file,num_evt)

    loader = DataLoader(
        data,
        batch_size=batch,
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        sampler=None,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_point_cloud,
    )
    return loader


