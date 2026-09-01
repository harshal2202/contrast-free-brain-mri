import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from train import compute_psnr, compute_ssim_simple, load_generator
from dataset import BraTSDataset

def evaluate(ckpt_path="checkpoints/generator_epoch0010.pth", img_size=256, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")
    if device.type == "cuda":
        print(f"[Evaluate] GPU: {torch.cuda.get_device_name(0)}")

    G = load_generator(ckpt_path, device)
    val_dataset = BraTSDataset("data/val", img_size=img_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    psnr_list = []
    ssim_list = []
    mae_list = []
    mse_list = []

    print(f"[Evaluate] Evaluating {len(val_dataset)} slices across {len(val_loader)} batches...")

    with torch.no_grad():
        for t1, t1ce in tqdm(val_loader, desc="Evaluating"):
            t1 = t1.to(device)
            t1ce = t1ce.to(device)

            pred = G(t1)
            pred = torch.clamp(pred, -1.0, 1.0)

            for p, gt in zip(pred, t1ce):
                p_val = p.squeeze()
                gt_val = gt.squeeze()

                psnr_val = compute_psnr(p_val, gt_val)
                ssim_val = compute_ssim_simple(p_val, gt_val)
                mae_val = (p_val - gt_val).abs().mean().item()
                mse_val = ((p_val - gt_val) ** 2).mean().item()

                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                mae_list.append(mae_val)
                mse_list.append(mse_val)

    results = {
        "checkpoint": ckpt_path,
        "total_slices": len(psnr_list),
        "metrics": {
            "SSIM": {
                "mean": float(np.mean(ssim_list)),
                "std": float(np.std(ssim_list)),
                "median": float(np.median(ssim_list)),
                "min": float(np.min(ssim_list)),
                "max": float(np.max(ssim_list)),
                "p95": float(np.percentile(ssim_list, 95))
            },
            "PSNR_dB": {
                "mean": float(np.mean(psnr_list)),
                "std": float(np.std(psnr_list)),
                "median": float(np.median(psnr_list)),
                "min": float(np.min(psnr_list)),
                "max": float(np.max(psnr_list)),
                "p95": float(np.percentile(psnr_list, 95))
            },
            "MAE": {
                "mean": float(np.mean(mae_list)),
                "std": float(np.std(mae_list)),
                "median": float(np.median(mae_list)),
                "min": float(np.min(mae_list)),
                "max": float(np.max(mae_list)),
                "p95": float(np.percentile(mae_list, 95))
            },
            "MSE": {
                "mean": float(np.mean(mse_list)),
                "std": float(np.std(mse_list)),
                "median": float(np.median(mse_list)),
                "min": float(np.min(mse_list)),
                "max": float(np.max(mse_list))
            }
        }
    }

    print("\n" + "=" * 60)
    print("                 MODEL EVALUATION REPORT                 ")
    print("=" * 60)
    print(f" Total Validation Slices : {results['total_slices']:,}")
    print(f" Model Checkpoint        : {ckpt_path}")
    print("-" * 60)
    print(f" SSIM (Structural Sim)   : {results['metrics']['SSIM']['mean']:.4f} +- {results['metrics']['SSIM']['std']:.4f} (Median: {results['metrics']['SSIM']['median']:.4f})")
    print(f" PSNR (Signal Quality)   : {results['metrics']['PSNR_dB']['mean']:.2f} dB +- {results['metrics']['PSNR_dB']['std']:.2f} dB (Median: {results['metrics']['PSNR_dB']['median']:.2f} dB)")
    print(f" MAE  (Intensity Error)  : {results['metrics']['MAE']['mean']:.4f} +- {results['metrics']['MAE']['std']:.4f} (Median: {results['metrics']['MAE']['median']:.4f})")
    print(f" MSE  (Squared Error)    : {results['metrics']['MSE']['mean']:.6f} +- {results['metrics']['MSE']['std']:.6f}")
    print("=" * 60 + "\n")

    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("[Evaluate] Full results saved to evaluation_results.json")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a generator on the validation set")
    parser.add_argument(
        "--ckpt",
        default="checkpoints/generator_epoch0010.pth",
        help="Generator checkpoint to evaluate",
    )
    parser.add_argument("--size", type=int, default=256, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Validation batch size")
    args = parser.parse_args()

    evaluate(args.ckpt, img_size=args.size, batch_size=args.batch_size)
