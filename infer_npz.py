import os
import glob
import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from train import load_generator

def infer_processed_patient(patient_id, checkpoint_path, processed_val_dir="data_processed/val", img_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Using device: {device}")
    
    # Load your trained generator
    G = load_generator(checkpoint_path, device)
    
    # Find all 2D slices belonging to this specific patient
    search_pattern = os.path.join(processed_val_dir, f"{patient_id}_slice_*.npz")
    slice_files = sorted(glob.glob(search_pattern), key=lambda x: int(x.split("_slice_")[-1].split(".npz")[0]))
    
    if not slice_files:
        print(f"[Error] No preprocessed slices found for patient {patient_id} in {processed_val_dir}")
        return

    print(f"[Infer] Found {len(slice_files)} slices for {patient_id}. Processing...")
    
    # Let's grab a middle slice to visualize the tumor transition clearly
    middle_idx = len(slice_files) // 2
    target_slice_path = slice_files[middle_idx]
    
    # Load the 2D array data
    data = np.load(target_slice_path)
    t1_slice = data['t1']
    t1ce_slice = data['t1ce']
    
    # Convert to tensors for network evaluation
    t1_tensor = torch.from_numpy(t1_slice).unsqueeze(0)
    t1_tensor = TF.resize(t1_tensor, [img_size, img_size], antialias=True).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tensor = G(t1_tensor)
        pred_tensor = torch.clamp(pred_tensor, -1.0, 1.0)
    
    # Convert back to viewable numpy arrays
    pred_np = pred_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
    # Plot side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(t1_slice, cmap='gray')
    axes[0].set_title("Input T1")
    axes[0].axis('off')
    
    axes[1].imshow(t1ce_slice, cmap='gray')
    axes[1].set_title("Ground Truth T1CE")
    axes[1].axis('off')
    
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title("Synthetic T1CE (AI)")
    axes[2].axis('off')
    
    output_img = f"predictions_{patient_id}.png"
    plt.tight_layout()
    plt.savefig(output_img, bbox_inches='tight', facecolor='black')
    print(f"[Infer] Done! Visual comparison saved to {output_img}")

if __name__ == "__main__":
    # Test on your patient visible in the file tree
    infer_processed_patient(
        patient_id="BraTS2021_00000", 
        checkpoint_path="checkpoints/generator_epoch0100.pth"
    )