import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

def _load_volume(path):
    img = nib.load(path)
    return img.get_fdata(dtype=np.float32)

def _normalise(vol):
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    if p99 - p1 < 1e-6:
        return np.zeros_like(vol)
    vol = (vol - p1) / (p99 - p1)
    vol = vol * 2.0 - 1.0
    return np.clip(vol, -1.0, 1.0)

class BraTSDataset(Dataset):
    def __init__(self, root_dir, img_size=256, trim_frac=0.15, augment=False):
        super().__init__()
        self.img_size = img_size
        self.trim_frac = trim_frac
        self.augment = augment
        self.slices = []

        patient_dirs = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        if len(patient_dirs) == 0:
            raise RuntimeError(f"No patient folders found in: {root_dir}")

        loaded = 0
        for patient in patient_dirs:
            p_path = os.path.join(root_dir, patient)
            t1_files   = glob.glob(os.path.join(p_path, "*_t1.nii.gz"))
            t1ce_files = glob.glob(os.path.join(p_path, "*_t1ce.nii.gz"))

            if not t1_files or not t1ce_files:
                print(f"[Dataset] Skipping {patient} - missing files")
                continue

            try:
                t1_vol   = _normalise(_load_volume(t1_files[0]))
                t1ce_vol = _normalise(_load_volume(t1ce_files[0]))
            except Exception as e:
                print(f"[Dataset] Could not load {patient}: {e}")
                continue

            D = t1_vol.shape[2]
            lo = int(np.ceil(D * self.trim_frac))
            hi = int(np.floor(D * (1.0 - self.trim_frac)))

            for s in range(lo, hi):
                t1_s   = t1_vol[:, :, s]
                t1ce_s = t1ce_vol[:, :, s]
                if t1_s.mean() < -0.95:
                    continue
                self.slices.append((t1_s.copy(), t1ce_s.copy()))

            loaded += 1

        if len(self.slices) == 0:
            raise RuntimeError("No valid slices found!")

        print(f"[Dataset] Loaded {len(self.slices)} slices from {loaded} patients in '{root_dir}'")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        t1_np, t1ce_np = self.slices[idx]
        t1   = torch.from_numpy(t1_np).unsqueeze(0)
        t1ce = torch.from_numpy(t1ce_np).unsqueeze(0)
        t1   = TF.resize(t1,   [self.img_size, self.img_size], antialias=True)
        t1ce = TF.resize(t1ce, [self.img_size, self.img_size], antialias=True)
        if self.augment and random.random() > 0.5:
            t1   = TF.hflip(t1)
            t1ce = TF.hflip(t1ce)
        return t1.float(), t1ce.float()
