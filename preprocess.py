import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm

def _load_volume(path):
    return nib.load(path).get_fdata(dtype=np.float32)

def _normalise(vol):
    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    if p99 - p1 < 1e-6:
        return np.zeros_like(vol)
    vol = (vol - p1) / (p99 - p1)
    vol = vol * 2.0 - 1.0
    return np.clip(vol, -1.0, 1.0)

def extract_splits(split_name, trim_frac=0.15):
    src_dir = os.path.join("data", split_name)
    dst_dir = os.path.join("data_processed", split_name)
    os.makedirs(dst_dir, exist_ok=True)
    
    patient_dirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    print(f"\nPreprocessing {split_name} split ({len(patient_dirs)} patients)...")
    
    for patient in tqdm(patient_dirs):
        p_path = os.path.join(src_dir, patient)
        t1_files = glob.glob(os.path.join(p_path, "*_t1.nii.gz"))
        t1ce_files = glob.glob(os.path.join(p_path, "*_t1ce.nii.gz"))
        
        if not t1_files or not t1ce_files:
            continue
            
        try:
            t1_vol = _normalise(_load_volume(t1_files[0]))
            t1ce_vol = _normalise(_load_volume(t1ce_files[0]))
        except Exception:
            continue
            
        D = t1_vol.shape[2]
        lo, hi = int(np.ceil(D * trim_frac)), int(np.floor(D * (1.0 - trim_frac)))
        
        for s in range(lo, hi):
            t1_s = t1_vol[:, :, s].astype(np.float32)
            t1ce_s = t1ce_vol[:, :, s].astype(np.float32)
            
            # Filter out empty background slices
            if t1_s.mean() < -0.95:
                continue
                
            np.savez_compressed(
                os.path.join(dst_dir, f"{patient}_slice_{s}.npz"),
                t1=t1_s, t1ce=t1ce_s
            )

if __name__ == "__main__":
    extract_splits("train")
    extract_splits("val")
    print("\nPreprocessing complete! All 2D slices extracted to 'data_processed'.")