"""
train_250.py
============
Fresh 250-epoch training using pre-sliced .npz data.
Saves ONLY to checkpoints_250/ — old checkpoints/ is NEVER touched.

Run:
    venv\Scripts\python.exe train_250.py
"""

import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_npz import BraTSDatasetNPZ
from modules import AttentionUNetGenerator, PatchGANDiscriminator, init_weights
from loss import GeneratorLoss, AdversarialLoss

# ─────────────────────────────────────────────────────────────────────
#  CONFIG
#  Old weights → checkpoints/       ✅ NEVER TOUCHED
#  New weights → checkpoints_250/   ✅ fresh folder
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root":   "data_processed",
    "output_dir":  "checkpoints_250",

    "img_size":    256,    # MUST be 256 — fixes U-Net bottleneck error
    "batch_size":  4,      # safe for RTX 2080 Ti 11GB
    "num_epochs":  250,
    "lr_g":        1e-4,   # lower = more stable long training
    "lr_d":        5e-5,   # discriminator learns slower = better balance
    "beta1":       0.5,
    "beta2":       0.999,

    # Higher weights = better contrast enhancement vs old training
    "lambda_adv":  1.0,
    "lambda_l1":   15.0,   # was 10 → sharper output
    "lambda_perc": 8.0,    # was 5  → better contrast detail
    "lambda_ssim": 8.0,    # was 5  → better structure

    "save_every":  10,
    "num_workers": 0,      # KEEP 0 on Windows
}


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * (torch.log10(torch.tensor(4.0 / mse))).item()


def compute_ssim_simple(pred, target):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1   = pred.mean()
    mu2   = target.mean()
    sig1  = pred.var()
    sig2  = target.var()
    sig12 = ((pred - mu1) * (target - mu2)).mean()
    ssim  = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
            ((mu1 ** 2 + mu2 ** 2 + C1) * (sig1 + sig2 + C2))
    return ssim.item()


def load_generator(checkpoint_path, device):
    G = AttentionUNetGenerator().to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    G.load_state_dict(state)
    G.eval()
    print(f"[Infer] Loaded generator from: {checkpoint_path}")
    return G


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Train] Using device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU : {torch.cuda.get_device_name(0)}")
        print(f"[Train] VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # Safety check — never overwrite old checkpoints
    assert CONFIG["output_dir"] != "checkpoints", \
        "ERROR: output_dir must not be 'checkpoints' — old weights protected!"
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"[Train] Saving to  : {CONFIG['output_dir']}/")
    print(f"[Train] Old weights: checkpoints/ (untouched)")

    # ── Datasets ──────────────────────────────────────────────────────
    print("\n[Train] Loading datasets ...")
    train_ds = BraTSDatasetNPZ(
        root_dir=os.path.join(CONFIG["data_root"], "train"),
        img_size=CONFIG["img_size"],
        augment=True,
    )
    val_ds = BraTSDatasetNPZ(
        root_dir=os.path.join(CONFIG["data_root"], "val"),
        img_size=CONFIG["img_size"],
        augment=False,
    )
    print(f"[Dataset] Train : {len(train_ds):,} slices")
    print(f"[Dataset] Val   : {len(val_ds):,} slices")

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Models ────────────────────────────────────────────────────────
    G = AttentionUNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)
    init_weights(G)
    init_weights(D)
    print(f"\n[Train] Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"[Train] Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Losses ────────────────────────────────────────────────────────
    criterion_G = GeneratorLoss(
        device=device,
        lambda_adv=CONFIG["lambda_adv"],
        lambda_l1=CONFIG["lambda_l1"],
        lambda_perc=CONFIG["lambda_perc"],
        lambda_ssim=CONFIG["lambda_ssim"],
    )
    criterion_D = AdversarialLoss()

    # ── Optimizers ────────────────────────────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=CONFIG["lr_g"],
                       betas=(CONFIG["beta1"], CONFIG["beta2"]))
    opt_D = optim.Adam(D.parameters(), lr=CONFIG["lr_d"],
                       betas=(CONFIG["beta1"], CONFIG["beta2"]))

    # Cosine annealing — smooth LR decay over 250 epochs
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(
        opt_G, T_max=CONFIG["num_epochs"], eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(
        opt_D, T_max=CONFIG["num_epochs"], eta_min=1e-6)

    # ── Training log ──────────────────────────────────────────────────
    log_path = os.path.join(CONFIG["output_dir"], "training_log.csv")
    log_file = open(log_path, "w")
    log_file.write("epoch,d_loss,g_loss,val_psnr,val_ssim,val_mae,time_s\n")

    hrs  = CONFIG["num_epochs"] * 7 // 60
    mins = (CONFIG["num_epochs"] * 7) % 60
    print(f"\n[Train] Starting {CONFIG['num_epochs']} epochs ...")
    print(f"[Train] Estimated time : ~{hrs}h {mins}m on RTX 2080 Ti")
    print(f"[Train] Log file       : {log_path}")
    print("=" * 70)

    best_psnr  = 0.0
    best_epoch = 0

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        G.train()
        D.train()
        t_start   = time.time()
        sum_d     = 0.0
        sum_g     = 0.0
        sum_comp  = {'adv': 0.0, 'l1': 0.0, 'perc': 0.0, 'ssim': 0.0}
        n_batches = len(train_loader)

        for batch_idx, (t1, t1ce) in enumerate(train_loader, 1):
            t1   = t1.to(device)
            t1ce = t1ce.to(device)

            # ── Discriminator ─────────────────────────────────────────
            opt_D.zero_grad()
            with torch.no_grad():
                fake = G(t1)
            loss_D = (criterion_D(D(t1, t1ce),         is_real=True) +
                      criterion_D(D(t1, fake.detach()), is_real=False)) * 0.5
            loss_D.backward()
            opt_D.step()

            # ── Generator ─────────────────────────────────────────────
            opt_G.zero_grad()
            fake = G(t1)
            loss_G, components = criterion_G(D(t1, fake), fake, t1ce)
            loss_G.backward()
            opt_G.step()

            sum_d += loss_D.item()
            sum_g += loss_G.item()
            for k in sum_comp:
                sum_comp[k] += components[k]

            if batch_idx % 100 == 0:
                print(f"  Ep {epoch}/{CONFIG['num_epochs']} "
                      f"Batch {batch_idx}/{n_batches} "
                      f"| D:{loss_D.item():.4f} "
                      f"G:{loss_G.item():.4f} "
                      f"[adv:{components['adv']:.3f} "
                      f"l1:{components['l1']:.3f} "
                      f"perc:{components['perc']:.3f} "
                      f"ssim:{components['ssim']:.3f}]")

        # ── Epoch summary ─────────────────────────────────────────────
        elapsed = time.time() - t_start
        avg_d   = sum_d / n_batches
        avg_g   = sum_g / n_batches
        avg_c   = {k: v / n_batches for k, v in sum_comp.items()}
        lr_now  = opt_G.param_groups[0]['lr']

        print(f"\nEpoch [{epoch:3d}/{CONFIG['num_epochs']}] "
              f"Time:{elapsed:.0f}s | LR:{lr_now:.2e} | "
              f"D:{avg_d:.4f} G:{avg_g:.4f} "
              f"(adv:{avg_c['adv']:.3f} "
              f"l1:{avg_c['l1']:.3f} "
              f"perc:{avg_c['perc']:.3f} "
              f"ssim:{avg_c['ssim']:.3f})")

        # ── Validation ────────────────────────────────────────────────
        G.eval()
        val_psnr = val_ssim = val_mae = 0.0
        n_val = 0
        with torch.no_grad():
            for t1_v, t1ce_v in val_loader:
                t1_v   = t1_v.to(device)
                t1ce_v = t1ce_v.to(device)
                pred_v = G(t1_v).clamp(-1, 1)
                val_psnr += compute_psnr(pred_v, t1ce_v)
                val_ssim += compute_ssim_simple(pred_v, t1ce_v)
                val_mae  += (pred_v - t1ce_v).abs().mean().item()
                n_val += 1

        avg_psnr = val_psnr / n_val
        avg_ssim = val_ssim / n_val
        avg_mae  = val_mae  / n_val

        print(f"  Validation → "
              f"PSNR:{avg_psnr:.2f}dB | "
              f"SSIM:{avg_ssim:.4f} | "
              f"MAE:{avg_mae:.4f}")

        # Auto save best model
        if avg_psnr > best_psnr:
            best_psnr  = avg_psnr
            best_epoch = epoch
            best_path  = os.path.join(CONFIG["output_dir"], "generator_best.pth")
            torch.save(G.state_dict(), best_path)
            print(f"  ★ New best PSNR {best_psnr:.2f}dB → {best_path}")

        print(f"  Best: {best_psnr:.2f}dB @ epoch {best_epoch}\n")

        # Log to CSV
        log_file.write(f"{epoch},{avg_d:.4f},{avg_g:.4f},"
                       f"{avg_psnr:.4f},{avg_ssim:.4f},{avg_mae:.4f},{elapsed:.0f}\n")
        log_file.flush()

        scheduler_G.step()
        scheduler_D.step()

        # Save checkpoint every N epochs
        if epoch % CONFIG["save_every"] == 0:
            g_path = os.path.join(CONFIG["output_dir"],
                                  f"generator_epoch{epoch:04d}.pth")
            d_path = os.path.join(CONFIG["output_dir"],
                                  f"discriminator_epoch{epoch:04d}.pth")
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Checkpoint] Saved → {g_path}\n")

    log_file.close()
    print("\n" + "=" * 70)
    print("[Train] Training complete!")
    print(f"[Train] Best PSNR  : {best_psnr:.2f} dB @ epoch {best_epoch}")
    print(f"[Train] Best model : {CONFIG['output_dir']}/generator_best.pth")
    print(f"[Train] All weights: {CONFIG['output_dir']}/")
    print(f"[Train] Log file   : {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    train()