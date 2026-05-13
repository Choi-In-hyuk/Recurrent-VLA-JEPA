"""
Dataset for offline Recurrent-JEPA fusion training.
Reads per-timestep .pt files from scripts/extract_recurrent_tokens.py.
Each .pt file = one training sample (one timestep of one demo).
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class RecurrentTokenDataset(Dataset):
    def __init__(self, data_dirs, split="train", seed=42):
        """
        Args:
            data_dirs: list of dataset directories, each with <split>/*.pt files
                       e.g. [".../libero_spatial", ".../libero_10"]
            split:     "train" or "val"
        """
        self.paths = []
        for d in data_dirs:
            pt_dir = os.path.join(d, split)
            if os.path.isdir(pt_dir):
                self.paths.extend(sorted(glob.glob(os.path.join(pt_dir, "*.pt"))))

        rng = random.Random(seed)
        rng.shuffle(self.paths)
        print(f"[RecurrentTokenDataset] {split}: {len(self.paths)} samples")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=False)
        return {
            "vj_obs":     torch.from_numpy(np.array(data["vj_obs"],     dtype=np.float32)),
            "vj_pred":    torch.from_numpy(np.array(data["vj_pred"],    dtype=np.float32)),
            "emb_tokens": torch.from_numpy(np.array(data["emb_tokens"], dtype=np.float32)),
            "actions":    torch.from_numpy(np.array(data["actions"],    dtype=np.float32)),
            "states":     torch.from_numpy(np.array(data["states"],     dtype=np.float32)),
        }
