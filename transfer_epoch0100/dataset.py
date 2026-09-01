import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random

class BraTSDataset(Dataset):
    def __init__(self, root_dir, img_size=256, trim_frac=0.15, augment=False):
        super().__init__()
        self.img_size = img_size
        self.augment = augment
        
        # Determine whether this instance is for train or val
        split = os.path.basename(os.path.normpath(root_dir))
        self.processed_dir = os.path.join("data_processed", split)
        
        if not os.path.exists(self.processed_dir):
            raise RuntimeError(f"Processed folder missing! Run preprocess.py first. Looking for: {self.processed_dir}")
            
        self.file_list = sorted([
            os.path.join(self.processed_dir, f) 
            for f in os.listdir(self.processed_dir) if f.endswith('.npz')
        ])
        
        print(f"[Dataset] Found {len(self.file_list)} preprocessed slices in '{self.processed_dir}'")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Microsecond disk read speed
        data = np.load(self.file_list[idx])
        t1_s = data['t1']
        t1ce_s = data['t1ce']
        
        t1 = torch.from_numpy(t1_s).unsqueeze(0)
        t1ce = torch.from_numpy(t1ce_s).unsqueeze(0)
        
        t1 = TF.resize(t1, [self.img_size, self.img_size], antialias=True)
        t1ce = TF.resize(t1ce, [self.img_size, self.img_size], antialias=True)
        
        if self.augment:
            if random.random() > 0.5:
                t1 = TF.hflip(t1)
                t1ce = TF.hflip(t1ce)
            if random.random() > 0.5:
                t1 = TF.vflip(t1)
                t1ce = TF.vflip(t1ce)
                
        return t1, t1ce