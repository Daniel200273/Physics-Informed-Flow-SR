import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
from data_processor import FluidDataset
from k_finder import calculate_global_max

def build_and_save(data_dir, stats_file, output_file, target_res):
    print(f"🏗️  STARTING DATASET BUILD PIPELINE (Low RAM Mode)")
    print(f"   > Input:  {data_dir}")
    print(f"   > Output: {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 1. Pre-processing
    print(f"\nStep 1: Calculating Normalization Factors...")
    try:
        calculate_global_max(data_dir=data_dir, output_file=stats_file)
    except Exception as e:
        print(f"⚠️ Warning: {e}")

    # 2. Instantiate Dataset
    print(f"\nStep 2: Loading Raw Data map...")
    # We use cache_data=False to save RAM. We only need the file list for now.
    dataset = FluidDataset(
        data_dir=data_dir, 
        stats_file=stats_file, 
        target_res=target_res, 
        noise_std=0.0, 
        cache_data=False 
    )
    
    num_samples = len(dataset)
    print(f"   > Found {num_samples} samples.")
    
    # 3. Pre-allocate Memory (The Fix)
    # Instead of a list, we create the final Float16 tensor immediately.
    # This reserves exactly ~6GB RAM and never spikes higher.
    print(f"\nStep 3: Allocating Memory for {num_samples} samples...")
    
    # Shape: (N, 4, 256, 256)
    final_inputs = torch.empty((num_samples, 4, target_res, target_res), dtype=torch.float16)
    final_targets = torch.empty((num_samples, 4, 256, 256), dtype=torch.float16)
    
    # 4. Fill Tensors Incrementally
    print(f"Step 4: Processing and Filling Tensors...")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    for i, (lr_frame, hr_frame) in enumerate(tqdm(loader, desc="Building")):
        # lr_frame is (1, 4, 256, 256) -> squeeze to (4, 256, 256)
        # We convert to half() IMMEDIATELY before storing
        final_inputs[i] = lr_frame.squeeze(0).half()
        final_targets[i] = hr_frame.squeeze(0).half()
        
    print(f"   > Final Input Shape:  {final_inputs.shape} (Float16)")
    
    # 5. Save to Disk
    print(f"\nStep 5: Saving to {output_file}...")
    torch.save({
        'inputs': final_inputs,
        'targets': final_targets,
        'K_vel': dataset.K_vel,
        'K_pres': dataset.K_pres,
        'K_smoke': dataset.K_smoke
    }, output_file)
    
    print("✅ Dataset built and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data", help="Folder containing raw .npz files")
    parser.add_argument("--out", type=str, default="../dataset/processed_dataset.pt", help="Path to save the output .pt file")
    parser.add_argument("--stats", type=str, default="normalization_stats.json", help="Path to normalization stats json")
    parser.add_argument("--target_res", type=int, default=256, help="Target resolution")
    
    args = parser.parse_args()
    
    build_and_save(args.data_dir, args.stats, args.out, args.target_res)