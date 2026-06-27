import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import BraTSDataset
from modules import AttentionUNetGenerator, PatchGANDiscriminator, init_weights
from loss import GeneratorLoss, AdversarialLoss

# ─────────────────────────────────────────────────────────────────────
#  CONFIG — edit these before running
#
#  For RTX 2080 Ti (11GB):  img_size=256, batch_size=4  ← default
#  For 8GB GPU:             img_size=128, batch_size=2
#  For CPU only:            img_size=128, batch_size=1
# ─────────────────────────────────────────────────────────────────────
CONFIG = {
    "data_root":   "data_processed",
    "output_dir":  "checkpoints",
    "img_size":    256,
    "batch_size":  4,
    "num_epochs":  100,
    "lr_g":        2e-4,
    "lr_d":        2e-4,
    "beta1":       0.5,
    "beta2":       0.999,
    "lambda_adv":  1.0,
    "lambda_l1":   10.0,
    "lambda_perc": 5.0,
    "lambda_ssim": 5.0,
    "save_every":  10,
    "num_workers": 0,   # KEEP 0 on Windows
}


def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * (torch.log10(torch.tensor(4.0 / mse))).item()


def compute_ssim_simple(pred, target):
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1  = pred.mean()
    mu2  = target.mean()
    sig1 = pred.var()
    sig2 = target.var()
    sig12 = ((pred - mu1) * (target - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) / \
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
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("\n[Train] Loading training data (3-5 mins for 1000+ patients)...")
    train_ds = BraTSDataset(
        root_dir=os.path.join(CONFIG["data_root"], "train"),
        img_size=CONFIG["img_size"],
        augment=True,
    )
    print("[Train] Loading validation data...")
    val_ds = BraTSDataset(
        root_dir=os.path.join(CONFIG["data_root"], "val"),
        img_size=CONFIG["img_size"],
        augment=False,
    )
    print(f"[Dataset] Train slices : {len(train_ds):,}")
    print(f"[Dataset] Val   slices : {len(val_ds):,}")

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

    G = AttentionUNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)
    init_weights(G)
    init_weights(D)

    criterion_G = GeneratorLoss(
        device=device,
        lambda_adv=CONFIG["lambda_adv"],
        lambda_l1=CONFIG["lambda_l1"],
        lambda_perc=CONFIG["lambda_perc"],
        lambda_ssim=CONFIG["lambda_ssim"],
    )
    criterion_D = AdversarialLoss()

    opt_G = optim.Adam(G.parameters(), lr=CONFIG["lr_g"],
                       betas=(CONFIG["beta1"], CONFIG["beta2"]))
    opt_D = optim.Adam(D.parameters(), lr=CONFIG["lr_d"],
                       betas=(CONFIG["beta1"], CONFIG["beta2"]))

    def lr_lambda(epoch):
        decay_start = CONFIG["num_epochs"] // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / decay_start)

    scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    scheduler_D = optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    print(f"\n[Train] Starting {CONFIG['num_epochs']} epochs ...\n")
    print(f"[Train] Estimated time: ~{CONFIG['num_epochs'] * 7 // 60}h "
          f"{(CONFIG['num_epochs'] * 7) % 60}m on RTX 2080 Ti\n")

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

            # ── Discriminator ────────────────────────────────────────
            opt_D.zero_grad()
            with torch.no_grad():
                fake = G(t1)
            loss_D = (criterion_D(D(t1, t1ce), is_real=True) +
                      criterion_D(D(t1, fake.detach()), is_real=False)) * 0.5
            loss_D.backward()
            opt_D.step()

            # ── Generator ────────────────────────────────────────────
            opt_G.zero_grad()
            fake = G(t1)
            loss_G, components = criterion_G(D(t1, fake), fake, t1ce)
            loss_G.backward()
            opt_G.step()

            sum_d += loss_D.item()
            sum_g += loss_G.item()
            for k in sum_comp:
                sum_comp[k] += components[k]

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{CONFIG['num_epochs']} "
                      f"Batch {batch_idx}/{n_batches} "
                      f"| D: {loss_D.item():.4f} "
                      f"G: {loss_G.item():.4f} "
                      f"[adv:{components['adv']:.3f} "
                      f"l1:{components['l1']:.3f} "
                      f"perc:{components['perc']:.3f} "
                      f"ssim:{components['ssim']:.3f}]")

        elapsed = time.time() - t_start
        avg_c   = {k: v / n_batches for k, v in sum_comp.items()}

        print(f"\nEpoch [{epoch:3d}/{CONFIG['num_epochs']}] "
              f"Time: {elapsed:.1f}s | "
              f"D: {sum_d/n_batches:.4f} | "
              f"G: {sum_g/n_batches:.4f} "
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

        print(f"  Validation → "
              f"PSNR: {val_psnr/n_val:.2f} dB | "
              f"SSIM: {val_ssim/n_val:.4f} | "
              f"MAE: {val_mae/n_val:.4f}\n")

        scheduler_G.step()
        scheduler_D.step()

        if epoch % CONFIG["save_every"] == 0:
            g_path = os.path.join(CONFIG["output_dir"], f"generator_epoch{epoch:04d}.pth")
            d_path = os.path.join(CONFIG["output_dir"], f"discriminator_epoch{epoch:04d}.pth")
            torch.save(G.state_dict(), g_path)
            torch.save(D.state_dict(), d_path)
            print(f"  [Saved] {g_path}\n")

    print("\n[Train] Training complete!")
    print(f"[Train] Final model: {CONFIG['output_dir']}/generator_epoch{CONFIG['num_epochs']:04d}.pth")


if __name__ == "__main__":
    train()
