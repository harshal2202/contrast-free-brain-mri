# Contrast-Free Brain MRI Enhancement
### T1 → T1CE Synthesis using Attention U-Net GAN
**St Joseph Engineering College, Mangaluru | VTU Belagavi | 2025-26**

---

## What This Does
Synthesises **contrast-enhanced T1CE MRI** from **non-contrast T1 MRI** using a GAN —
eliminating the need for gadolinium injection.

```
Non-Contrast T1 MRI  →  [Attention U-Net GAN]  →  Synthetic T1CE MRI
```

---

## Team
| Name | USN |
|---|---|
| Harshal S Poojary | 4SO23CS091 |
| Himansh S Puthran | 4SO23CS093 |
| Hitha Kaje | 4SO23CS094 |
| Lakshith | 4SO23CS126 |

**Guide:** Ms Rakshitha Naresh, Assistant Professor, CSE

---

## Files
```
├── dataset.py       ← BraTS NIfTI data loader
├── modules.py       ← Attention U-Net Generator + PatchGAN Discriminator
├── loss.py          ← Adversarial + L1 + Perceptual + SSIM losses
├── train.py         ← Training script (edit CONFIG at top)
├── infer.py         ← Run on new patient after training
├── visualise.py     ← Side-by-side comparison plot
└── requirements.txt
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Dataset
Download **BraTS 2021** from Kaggle:
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

Extract to `E:\BraTS2021\` then split into train/val:
```powershell
$src = "E:\BraTS2021"
$patients = Get-ChildItem $src -Directory | Where-Object { $_.Name -like "BraTS2021_*" } | Sort-Object { Get-Random }
$patients | Select-Object -First 1000 | ForEach-Object { Copy-Item $_.FullName data\train -Recurse }
$patients | Select-Object -Last  251  | ForEach-Object { Copy-Item $_.FullName data\val   -Recurse }
```

---

## Train
```bash
python train.py
```

**GPU Settings in train.py CONFIG:**
| GPU VRAM | img_size | batch_size |
|---|---|---|
| 11 GB (RTX 2080 Ti) | 256 | 4 |
| 8 GB | 128 | 2 |
| CPU only | 128 | 1 |

**Training time:**
| Hardware | Per epoch | 100 epochs |
|---|---|---|
| RTX 2080 Ti | ~7 mins | ~12 hours |
| 8GB GPU | ~18 mins | ~30 hours |

---

## Infer on New Patient
```bash
python infer.py \
    --t1   data/val/BraTS2021_00000/BraTS2021_00000_t1.nii.gz \
    --ckpt checkpoints/generator_epoch0100.pth \
    --out  predictions/result.nii.gz
```

---

## Visualise Results
Edit paths in `visualise.py` then:
```bash
python visualise.py
```
Saves `comparison.png` showing T1 | T1CE Ground Truth | Predicted T1CE

---

## Expected Metrics (Well-Trained Model)
| Metric | Target |
|---|---|
| PSNR | > 32 dB |
| SSIM | > 0.90 |
| MAE  | < 0.05 |

---

## Architecture
- **Generator:** Attention U-Net with skip connections and attention gates
- **Discriminator:** 70×70 PatchGAN
- **Loss:** Adversarial (LSGAN) + L1 + Perceptual (VGG-16) + SSIM
- **Dataset:** BraTS 2021 (1251 patients, T1/T1CE paired NIfTI volumes)
