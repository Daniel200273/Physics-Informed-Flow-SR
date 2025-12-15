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
            noise_std (float): Standard deviation of Gaussian noise (Augmentation).
            cache_data (bool): If True, loads all sims into RAM for speed.
        """
        self.cache_data = cache_data
        self.target_res = target_res
        self.noise_std = noise_std
        
        # 1. Load Normalization Constants (Velocity, Pressure, Smoke)
        if not os.path.exists(stats_file):
            print(f"⚠️ Warning: {stats_file} not found. Using default K=1.0 for all.")
            self.K_vel = 1.0
            self.K_pres = 1.0
            self.K_smoke = 1.0
        else:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            # Load specific Ks, fallback to 1.0 if missing
            self.K_vel = float(stats.get('K_vel', 1.0))
            self.K_pres = float(stats.get('K_pres', 1.0))
            self.K_smoke = float(stats.get('K_smoke', 1.0))
        
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

                # Create indices for Single Frame [t]
                # We skip the very first/last frames to be safe, though not strictly needed for single-frame
                for t in range(0, num_frames):
                    self.indices.append((i, t))
                    
            except Exception as e:
                print(f"⚠️ Skipping file {fpath}: {e}")
        
        print(f"✅ Dataset ready: {len(self.indices)} samples | Target Res: {self.target_res}x{self.target_res}")
        print(f"   Normalization: K_v={self.K_vel}, K_p={self.K_pres}, K_s={self.K_smoke}")

    def __len__(self):
        return len(self.indices)

    def normalize_channels(self, tensor):
        """
        Normalizes (C, H, W) tensor where:
        C0=u, C1=v (Velocity) -> / K_vel
        C2=p (Pressure)       -> / K_pres
        C3=s (Smoke)          -> / K_smoke
        """
        # Clone to avoid modifying the cached data in-place
        out = tensor.clone()
        out[0] /= self.K_vel
        out[1] /= self.K_vel
        out[2] /= self.K_pres
        out[3] /= self.K_smoke
        return out

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
        
        # B. Extract Single Frame [t]
        # Data shape in NPZ is (Time, H, W, 4) -> [u, v, p, s]
        lr_frame = lr_seq[t]  # (H_in, W_in, 4)
        hr_frame = hr_seq[t]  # (256, 256, 4)

        # C. To Tensor & Permute
        # (H, W, 4) -> (4, H, W)
        lr_tensor = torch.from_numpy(lr_frame).permute(2, 0, 1).float()
        hr_tensor = torch.from_numpy(hr_frame).permute(2, 0, 1).float()

        # D. Normalize Channels Independently
        lr_tensor = self.normalize_channels(lr_tensor)
        hr_tensor = self.normalize_channels(hr_tensor)
        
        # E. Dynamic Upscaling (if input is Low Res)
        if lr_tensor.shape[-1] != self.target_res:
            lr_tensor = F.interpolate(
                lr_tensor.unsqueeze(0), # Add batch dim for interpolate: (1, 4, H, W)
                size=(self.target_res, self.target_res), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0) # Remove batch dim: (4, 256, 256)
            
        # F. Noise Augmentation (Only if enabled)
        if self.noise_std > 0:
            noise = torch.randn_like(lr_tensor) * self.noise_std
            lr_tensor += noise

        # G. Return Tensors
        # Input: (4, 256, 256) -> [u, v, p, s]
        # Target: (4, 256, 256) -> [u, v, p, s]
        return lr_tensor, hr_tensor