import os
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class FluidDataset(Dataset):
    def __init__(self, data_dir="../data", stats_file="normalization_stats.json", cache_data=True):
        """
        Args:
            data_dir (str): Path to folder containing .npz files.
            stats_file (str): Path to the JSON file containing the 'scaling_factor' K.
            cache_data (bool): If True, loads all sims into RAM for faster access.
        """
        self.cache_data = cache_data
        
        # 1. Load Normalization Constants
        if not os.path.exists(stats_file):
            raise FileNotFoundError(f"Run preprocess_data.py first! Missing {stats_file}")
            
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        self.K = stats['scaling_factor']
        
        # 2. Get File List (All files)
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if len(self.files) == 0:
            raise ValueError(f"No .npz files found in {data_dir}")

        # 3. Build Index Map (Flatten the dataset)
        # We need a list of valid (file_index, frame_index) tuples for the ENTIRE dataset.
        # Valid frames for target 't' are 1 to N-2 (so t-1 and t+1 exist).
        self.indices = []
        self.data_cache = {} # Used if cache_data is True
        
        print(f"📦 Loading dataset indices from {len(self.files)} simulations...")
        
        for i, fpath in enumerate(self.files):
            try:
                # If caching, load data now. If not, just peek at shape.
                if self.cache_data:
                    with np.load(fpath) as data:
                        hr = data['hr'].astype(np.float32)
                        lr = data['lr'].astype(np.float32)
                        self.data_cache[i] = (hr, lr)
                        num_frames = hr.shape[0]
                else:
                    with np.load(fpath) as data:
                        num_frames = data['hr'].shape[0]

                # Create valid indices: Frame 1 to N-2
                # e.g. if 150 frames (0..149), valid t are 1..148
                for t in range(1, num_frames - 1):
                    self.indices.append((i, t))
                    
            except Exception as e:
                print(f"⚠️ Skipping corrupt file {fpath}: {e}")
        
        print(f"✅ Dataset ready with {len(self.indices)} total samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_idx, t = self.indices[idx]
        
        # A. Load Data
        if self.cache_data:
            hr_seq, lr_seq = self.data_cache[file_idx]
        else:
            fpath = self.files[file_idx]
            with np.load(fpath) as data:
                hr_seq = data['hr']
                lr_seq = data['lr']
        
        # B. Extract Window
        # Inputs: Low Res frames [t-1, t, t+1]
        # Shape in NPZ: (T, H, W, C) -> Slice: (3, 32, 32, 2)
        lr_window = lr_seq[t-1 : t+2] 
        
        # Target: High Res frame [t]
        # Shape in NPZ: (H, W, C) -> (256, 256, 2)
        hr_target = hr_seq[t]

        # C. Convert to Tensor & Arrange Dimensions
        # PyTorch expects (Batch, Channel, Height, Width)
        lr_tensor = torch.from_numpy(lr_window).float() # (3, 32, 32, 2)
        hr_tensor = torch.from_numpy(hr_target).float() # (256, 256, 2)
        
        # Permute to (Time, Channel, H, W) or (Channel, H, W)
        lr_tensor = lr_tensor.permute(0, 3, 1, 2) # -> (3, 2, 32, 32)
        hr_tensor = hr_tensor.permute(2, 0, 1)    # -> (2, 256, 256)

        # D. Normalize using Global K
        lr_tensor = lr_tensor / self.K
        hr_tensor = hr_tensor / self.K
        
        # E. Pre-Upscale (Bilinear Interpolation)
        # Input shape: (3, 2, 32, 32) -> (3, 2, 256, 256)
        # We treat the first dim (3) as batch for interpolate
        lr_upscaled = F.interpolate(
            lr_tensor, 
            scale_factor=8,  # 32 * 8 = 256
            mode='bilinear', 
            align_corners=False
        ) 
        
        # F. Flatten Channels
        # Merge Time and Channels: (3, 2, 256, 256) -> (6, 256, 256)
        network_input = lr_upscaled.reshape(-1, 256, 256)
        
        return network_input, hr_tensor