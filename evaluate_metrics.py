"""
evaluate_metrics.py
===================
Calculates explicit PSNR, SSIM, and MAE values for a specific predicted output volume.
"""
import argparse
import os
import numpy as np
import nibabel as nib
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def evaluate_patient_scan(gt_path, pred_path):
    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        raise FileNotFoundError("Make sure both the Ground Truth and Prediction files exist.")

    # Load 3D volumes
    gt_vol = nib.load(gt_path).get_fdata(dtype=np.float32)
    pred_vol = nib.load(pred_path).get_fdata(dtype=np.float32)

    # Standardize intensity range to [0, 1] for metrics calculation
    def to_zero_one(v):
        p1, p99 = np.percentile(v, 1), np.percentile(v, 99)
        if p99 - p1 < 1e-6: return np.zeros_like(v)
        return np.clip((v - p1) / (p99 - p1), 0.0, 1.0)

    gt_norm = to_zero_one(gt_vol)
    pred_norm = to_zero_one(pred_vol)

    # Focus evaluation on the non-empty middle slices where the brain is present
    mid_idx = gt_norm.shape[2] // 2
    gt_slice = gt_norm[:, :, mid_idx]
    pred_slice = pred_norm[:, :, mid_idx]

    # Calculate metrics
    calc_psnr = psnr(gt_slice, pred_slice, data_range=1.0)
    calc_ssim = ssim(gt_slice, pred_slice, data_range=1.0)
    calc_mae = np.mean(np.abs(gt_slice - pred_slice))

    print("\n" + "="*40)
    print(f" METRICS FOR SCAN: {os.path.basename(pred_path)}")
    print("="*40)
    print(f" Peak Signal-to-Noise Ratio (PSNR)   : {calc_psnr:.2f} dB")
    print(f" Structural Similarity Index (SSIM) : {calc_ssim:.4f}")
    print(f" Mean Absolute Error (MAE)          : {calc_mae:.4f}")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to real ground-truth T1CE (.nii.gz)")
    parser.add_argument("--pred", required=True, help="Path to your generated result (.nii.gz)")
    args = parser.parse_args()
    
    evaluate_patient_scan(args.gt, args.pred)