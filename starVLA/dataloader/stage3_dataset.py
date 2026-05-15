"""
Dataset for Stage 3 Recurrent-JEPA training.

Returns per-timestep samples that include:
  - vj_obs / vj_pred  (pre-extracted JEPA features → compute Δz)
  - PIL images         (loaded from HDF5 on-the-fly → feed to QwenVL)
  - instruction        (string → feed to QwenVL)
  - actions / states   (for DiT action loss)

The .pt files must have been created by the updated
scripts/extract_recurrent_tokens.py which saves Stage 3 metadata
(instruction, hdf5_path, demo_key, timestep).
"""

import glob
import os
import random

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_SIZE = (224, 224)


class Stage3Dataset(Dataset):
    def __init__(self, data_dirs, split="train", seed=42):
        """
        Args:
            data_dirs: list of dataset directories, each with <split>/*.pt files
            split:     "train" or "val"
        """
        self.paths = []
        for d in data_dirs:
            pt_dir = os.path.join(d, split)
            if os.path.isdir(pt_dir):
                self.paths.extend(sorted(glob.glob(os.path.join(pt_dir, "*.pt"))))

        rng = random.Random(seed)
        rng.shuffle(self.paths)

        # Filter out samples without Stage 3 metadata (old extraction format)
        valid = []
        for p in self.paths:
            data = torch.load(p, weights_only=False)
            if "hdf5_path" in data and "instruction" in data:
                valid.append(p)
        self.paths = valid
        print(f"[Stage3Dataset] {split}: {len(self.paths)} samples with metadata")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)

        # Load images from HDF5 on-the-fly
        hdf5_path = data["hdf5_path"]
        demo_key  = data["demo_key"]
        t         = data["timestep"]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]
            agent_raw = demo["obs"]["agentview_rgb"][t]   # (H, W, 3) uint8
            wrist_raw = demo["obs"]["eye_in_hand_rgb"][t] # (H, W, 3) uint8

        agent_img = Image.fromarray(agent_raw).resize(IMAGE_SIZE, Image.BILINEAR)
        wrist_img = Image.fromarray(wrist_raw).resize(IMAGE_SIZE, Image.BILINEAR)

        return {
            "vj_obs":      torch.from_numpy(np.array(data["vj_obs"],  dtype=np.float32)),
            "vj_pred":     torch.from_numpy(np.array(data["vj_pred"], dtype=np.float32)),
            "actions":     torch.from_numpy(np.array(data["actions"], dtype=np.float32)),
            "states":      torch.from_numpy(np.array(data["states"],  dtype=np.float32)),
            "instruction": data["instruction"],
            "images":      [agent_img, wrist_img],  # list of PIL Images
        }


def stage3_collate_fn(batch):
    """Custom collate: stack tensors, keep images/instructions as lists."""
    return {
        "vj_obs":       torch.stack([b["vj_obs"]   for b in batch]),
        "vj_pred":      torch.stack([b["vj_pred"]  for b in batch]),
        "actions":      torch.stack([b["actions"]  for b in batch]),
        "states":       torch.stack([b["states"]   for b in batch]),
        "instructions": [b["instruction"] for b in batch],
        "images":       [b["images"]      for b in batch],  # list of [PIL, PIL] per sample
    }
