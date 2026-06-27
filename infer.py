"""
infer.py
========
Run on a new patient after training is complete.

Usage
-----
python infer.py \
    --t1   data/val/BraTS2021_00000/BraTS2021_00000_t1.nii.gz \
    --ckpt checkpoints/generator_epoch0100.pth \
    --out  predictions/result.nii.gz
"""

import argparse
import os
import numpy as np
import nibabel as nib
import torch
import torchvision.transforms.functional as TF

from train import load_generator
from dataset import _load_volume, _normalise


def infer_volume(t1_path, checkpoint_path, output_path, img_size=256, trim_frac=0.15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    G = load_generator(checkpoint_path, device)

    print(f"[Infer] Loading T1 from: {t1_path}")
    original_img = nib.load(t1_path)
    t1_vol = _normalise(_load_volume(t1_path))
    H_orig, W_orig, D = t1_vol.shape

    lo = int(np.ceil(D * trim_frac))
    hi = int(np.floor(D * (1.0 - trim_frac)))

    predicted_vol = np.zeros_like(t1_vol)
    print(f"[Infer] Processing {hi - lo} slices ...")

    with torch.no_grad():
        for s_idx in range(lo, hi):
            t1_slice = torch.from_numpy(t1_vol[:, :, s_idx]).unsqueeze(0)
            t1_slice = TF.resize(t1_slice, [img_size, img_size],
                                 antialias=True).unsqueeze(0).to(device)
            pred = G(t1_slice)
            pred = torch.clamp(pred, -1., 1.)
            pred_np = TF.resize(pred.squeeze(0), [H_orig, W_orig],
                                antialias=True).squeeze(0).cpu().numpy()
            predicted_vol[:, :, s_idx] = pred_np

            if (s_idx - lo + 1) % 20 == 0:
                print(f"  Slice {s_idx - lo + 1}/{hi - lo} done")

    predicted_vol = (predicted_vol + 1.0) / 2.0
    predicted_vol = predicted_vol.astype(np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"[Infer] Saving to: {output_path}")
    out_img = nib.Nifti1Image(predicted_vol,
                               affine=original_img.affine,
                               header=original_img.header)
    nib.save(out_img, output_path)
    print("[Infer] Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1",   required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out",  required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    infer_volume(args.t1, args.ckpt, args.out, args.size)
