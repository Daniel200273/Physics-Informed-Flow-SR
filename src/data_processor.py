import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class FluidDataset(Dataset):
    def __init__(self, data_dir="../data", stats_file="normalization_stats.json", 
                 target_res=256, noise_std=0.0, cache_data=True):
        """
        Args:
            data_dir (str): Path to folder containing .npz files.
            stats_file (str): Path to normalization stats JSON.
            target_res (int): The resolution the network expects (e.g., 256).
            noise_std (float): Standard deviation of Gaussian noise to add (Augmentation).
            cache_data (bool): If True, loads all sims into RAM.
        """
        self.cache_data = cache_data
        self.target_res = target_res
        self.noise_std = noise_std
        
        # 1. Load Normalization Constants
        if not os.path.exists(stats_file):
            print(f"⚠️ Warning: {stats_file} not found. Using default K=1.0")
            self.K = 1.0
        else:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            self.K = float(stats.get('scaling_factor', 1.0))
        
        # 2. Get File List
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # 3. Build Index Map
        self.indices = []
        self.data_cache = {} 
        
        print(f"📦 Loading dataset from {len(self.files)} simulations...")
        
        for i, fpath in enumerate(self.files):
            try:
                # Load header/data
                if self.cache_data:
                    with np.load(fpath) as data:
                        hr = data['hr'].astype(np.float32)
                        lr = data['lr'].astype(np.float32)
                        self.data_cache[i] = (hr, lr)
                        num_frames = hr.shape[0]
                else:
                    # Just peek at shape without loading full data
                    with np.load(fpath) as data:
                        num_frames = data['hr'].shape[0]

                # Create indices for window [t-1, t, t+1]
                # Valid t range: 1 to N-2
                for t in range(1, num_frames - 1):
                    self.indices.append((i, t))
                    
            except Exception as e:
                print(f"⚠️ Skipping file {fpath}: {e}")
        
        print(f"✅ Dataset ready: {len(self.indices)} samples | Target Res: {self.target_res}x{self.target_res}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, t = self.indices[idx]
        
        # A. Get Data
        if self.cache_data:
            hr_seq, lr_seq = self.data_cache[file_idx]
        else:
            fpath = self.files[file_idx]
            with np.load(fpath) as data:
                hr_seq = data['hr']
                lr_seq = data['lr']
        
        # B. Extract Window [t-1, t, t+1]
        # lr_seq shape: (Time, H, W, C)
        lr_window = lr_seq[t-1 : t+2]  # (3, H_in, W_in, 2)
        hr_target = hr_seq[t]          # (256, 256, 2)

        # C. To Tensor & Permute
        # (3, H, W, 2) -> (3, 2, H, W)
        lr_tensor = torch.from_numpy(lr_window).permute(0, 3, 1, 2).float()
        # (H, W, 2) -> (2, H, W)
        hr_tensor = torch.from_numpy(hr_target).permute(2, 0, 1).float()

        # D. Normalize
        lr_tensor = lr_tensor / self.K
        hr_tensor = hr_tensor / self.K
        
        # E. Dynamic Upscaling
        # Automatically detects input size and upscales to target_res (256)
        if lr_tensor.shape[-1] != self.target_res:
            lr_tensor = F.interpolate(
                lr_tensor, 
                size=(self.target_res, self.target_res), 
                mode='bilinear', 
                align_corners=False
            )
            
        # F. Noise Augmentation (Only if enabled)
        if self.noise_std > 0:
            noise = torch.randn_like(lr_tensor) * self.noise_std
            lr_tensor += noise

        # G. Flatten Channels for Network
        # (3, 2, 256, 256) -> (6, 256, 256)
        network_input = lr_tensor.reshape(-1, self.target_res, self.target_res)
        
        return network_input, hr_tensor