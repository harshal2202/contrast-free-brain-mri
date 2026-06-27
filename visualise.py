"""
visualise.py
============
Side-by-side comparison: T1 Input | T1CE Ground Truth | Predicted T1CE

Edit the three paths below then run:
    python visualise.py
"""

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# ── Edit these paths ──────────────────────────────────────────────────
T1_PATH        = "data/val/BraTS2021_00000/BraTS2021_00000_t1.nii.gz"
T1CE_PATH      = "data/val/BraTS2021_00000/BraTS2021_00000_t1ce.nii.gz"
PREDICTED_PATH = "predictions/result.nii.gz"
OUTPUT_PNG     = "comparison.png"
# ─────────────────────────────────────────────────────────────────────


def load_and_clip(path):
    vol = nib.load(path).get_fdata(dtype=np.float32)
    lo, hi = np.percentile(vol, 1), np.percentile(vol, 99)
    if hi - lo < 1e-6:
        return vol
    return np.clip((vol - lo) / (hi - lo), 0, 1)


def main():
    t1   = load_and_clip(T1_PATH)
    t1ce = load_and_clip(T1CE_PATH)
    pred = load_and_clip(PREDICTED_PATH)

    mid = t1.shape[2] // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='black')
    titles  = ["T1 Input (Non-Contrast)",
               "T1CE Ground Truth",
               "Predicted T1CE (Synthetic)"]
    volumes = [t1[:, :, mid], t1ce[:, :, mid], pred[:, :, mid]]

    for ax, vol, title in zip(axes, volumes, titles):
        ax.imshow(np.rot90(vol), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, color='white', fontsize=13, pad=8)
        ax.axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches='tight', facecolor='black')
    print(f"[Visualise] Saved → {OUTPUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
