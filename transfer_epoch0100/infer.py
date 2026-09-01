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
import matplotlib.pyplot as plt

from train import load_generator
from preprocess import _load_volume, _normalise


def infer_volume(t1_path, checkpoint_path, output_path, img_size=256, trim_frac=0.15, save_png=False, slice_idx=None):
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

    # If requested, save comparison PNG showing input / real / predicted
    if save_png:
        # try to locate a T1ce file if present in same folder
        t1ce_path = None
        # prefer an explicitly provided matching _t1ce file nearby
        base_dir = os.path.dirname(t1_path)
        base_name = os.path.basename(t1_path)
        if base_name.endswith("_t1.nii.gz"):
            candidate = os.path.join(base_dir, base_name.replace("_t1.nii.gz", "_t1ce.nii.gz"))
            if os.path.exists(candidate):
                t1ce_path = candidate

        # load real T1ce if available
        real_vol = None
        if t1ce_path is not None:
            try:
                real_img_obj = nib.load(t1ce_path)
                real_vol = _normalise(_load_volume(t1ce_path))
                print(f"[Infer] Loaded ground-truth T1ce from: {t1ce_path}")
            except Exception:
                real_vol = None

        # choose slice index (central by default)
        z = slice_idx if slice_idx is not None else predicted_vol.shape[2] // 2
        png_path = os.path.splitext(output_path)[0] + f"_compare_z{z}.png"

        inp = t1_vol[:, :, z]
        pred = predicted_vol[:, :, z]
        if real_vol is not None:
            real = real_vol[:, :, z]

        # display side-by-side: input | real (if available) | predicted
        ncols = 3 if real_vol is not None else 2
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        if ncols == 3:
            ax0, ax1, ax2 = axes
            ax0.imshow(inp, cmap="gray")
            ax0.set_title("Input T1")
            ax0.axis("off")

            ax1.imshow(real, cmap="gray")
            ax1.set_title("Real T1ce")
            ax1.axis("off")

            ax2.imshow(pred, cmap="gray")
            ax2.set_title("Predicted")
            ax2.axis("off")
        else:
            ax0, ax1 = axes
            ax0.imshow(inp, cmap="gray")
            ax0.set_title("Input T1")
            ax0.axis("off")

            ax1.imshow(pred, cmap="gray")
            ax1.set_title("Predicted")
            ax1.axis("off")

        plt.tight_layout()
        plt.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"[Infer] Saved comparison PNG to: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--t1",   required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out",  required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--save-png", action="store_true", help="Also save a central axial PNG slice next to the output .nii.gz")
    parser.add_argument("--slice", type=int, default=None, help="Which axial slice index to save (default: center)")
    args = parser.parse_args()
    infer_volume(args.t1, args.ckpt, args.out, args.size, save_png=args.save_png, slice_idx=args.slice)
