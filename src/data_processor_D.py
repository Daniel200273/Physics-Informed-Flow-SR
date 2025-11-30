import torch
from torch.utils.data import Dataset
import numpy as np
import glob
import torch.nn.functional as F

class FluidDataset(Dataset):
    def __init__(self, data_dir, global_max, split='train'):
        """
        data_dir: Path to folder with .npy files
        global_max: The value printed by your generation script
        split: 'train' (first 16 sims) or 'val' (last 4 sims)
        """
        self.max_val = global_max
        
        # 1. Get List of Files
        all_hr_files = sorted(glob.glob(f"{data_dir}/*_hr.npy"))
        all_lr_files = sorted(glob.glob(f"{data_dir}/*_lr.npy"))
        
        # 2. Split Logic (Keep simulations intact!)
        if split == 'train':
            self.hr_files = all_hr_files[:16] # First 16 simulations
            self.lr_files = all_lr_files[:16]
        else:
            self.hr_files = all_hr_files[16:] # Last 4 simulations
            self.lr_files = all_lr_files[16:]
            
        # 3. Create Index Map
        # We need a list of every valid (Sim_Index, Frame_Index) tuple
        self.samples = []
        for sim_idx, fpath in enumerate(self.hr_files):
            # Load just to check length (inefficient but safe)
            # Or assume 200 frames if fixed.
            data = np.load(fpath) 
            num_frames = data.shape[0] # Should be 200
            
            # We can't use frame 0 (no prev) or frame 199 (no next)
            for t in range(1, num_frames - 1):
                self.samples.append((sim_idx, t))

        # Cache data in RAM (If 4000 frames is too big, implement lazy loading)
        self.cache_hr = [np.load(f) for f in self.hr_files]
        self.cache_lr = [np.load(f) for f in self.lr_files]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sim_id, t = self.samples[idx]
        
        # A. GET DATA
        # Shape: (3, 2, 32, 32) -> Frames t-1, t, t+1
        lr_window_np = self.cache_lr[sim_id][t-1 : t+2]
        # Shape: (2, 128, 128) -> Frame t only
        hr_target_np = self.cache_hr[sim_id][t]
        
        # Convert to Tensor
        lr_tensor = torch.from_numpy(lr_window_np).float()
        hr_tensor = torch.from_numpy(hr_target_np).float()
        
        # B. NORMALIZE
        lr_tensor = lr_tensor / self.max_val
        hr_tensor = hr_tensor / self.max_val
        
        # C. PRE-UPSCALE (The Refinement Step)
        # We must resize the (3, 2, 32, 32) input to (3, 2, 128, 128)
        # F.interpolate expects (Batch, Channel, H, W).
        # We treat "Time" as "Batch" for a moment to resize all 3 frames at once.
        lr_upscaled = F.interpolate(lr_tensor, scale_factor=4, mode='bilinear', align_corners=False)
        
        # D. FLATTEN CHANNELS
        # Current shape: (3, 2, 128, 128)
        # Desired shape: (6, 128, 128)
        network_input = lr_upscaled.view(-1, 128, 128)
        
        return network_input, hr_tensor