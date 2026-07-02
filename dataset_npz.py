"""
dataset_npz.py
==============
Fast Dataset for pre-sliced .npz files.
Each file contains 't1' and 't1ce' arrays of shape (240,240) in [-1,1].
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random


class BraTSDatasetNPZ(Dataset):
    def __init__(self, root_dir, img_size=240, augment=False):
        super().__init__()
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment  = augment

        self.files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith('.npz')
        ])

        if len(self.files) == 0:
            raise RuntimeError(f"No .npz files found in: {root_dir}")

        print(f"[Dataset] Found {len(self.files):,} slices in '{root_dir}'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        data = np.load(path)

        t1   = torch.from_numpy(data['t1'].astype(np.float32)).unsqueeze(0)
        t1ce = torch.from_numpy(data['t1ce'].astype(np.float32)).unsqueeze(0)

        # Resize only if needed
        if self.img_size != 240:
            t1   = TF.resize(t1,   [self.img_size, self.img_size], antialias=True)
            t1ce = TF.resize(t1ce, [self.img_size, self.img_size], antialias=True)

        # Augmentation
        if self.augment and random.random() > 0.5:
            t1   = TF.hflip(t1)
            t1ce = TF.hflip(t1ce)
        if self.augment and random.random() > 0.5:
            t1   = TF.vflip(t1)
            t1ce = TF.vflip(t1ce)

        return t1.float(), t1ce.float()
