import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm  # Progress bar

# Import your custom modules
from data_processor import FluidDataset
from k_finder import calculate_global_max

# --- Configuration ---
DATA_DIR = '../data'
STATS_FILE = 'normalization_stats.json'
OUTPUT_FILE = '../dataset/processed_dataset.pt'

def build_and_save():
    print("🏗️  STARTING DATASET BUILD PIPELINE")
    
    # 1. Pre-processing: Calculate Max Velocity (K)
    print(f"\nStep 1: Calculating Global Normalization Factor (K)...")
    calculate_global_max(data_dir=DATA_DIR, output_file=STATS_FILE)
    
    # 2. Instantiate Dataset
    # This loads the raw .npz data into RAM
    print(f"\nStep 2: Loading Raw Data into Memory...")
    dataset = FluidDataset(data_dir=DATA_DIR, stats_file=STATS_FILE, cache_data=True)
    print(f"   > Loaded {len(dataset)} total samples.")
    
    # 3. Process All Frames
    # We iterate through the dataset to apply the __getitem__ logic 
    # (Normalization -> Interpolation -> Reshaping)
    print(f"\nStep 3: Processing Tensors (Upscaling & Normalizing)...")
    
    # We use a DataLoader with batch_size=1 just to iterate efficiently
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    all_inputs = []
    all_targets = []
    
    # Wrap in tqdm for a progress bar
    for inputs, targets in tqdm(loader, desc="Processing"):
        # Inputs are already (Batch, 6, 256, 256) from the Dataset class
        # We remove the batch dimension to stack them later
        all_inputs.append(inputs.squeeze(0))
        all_targets.append(targets.squeeze(0))
        
    # 4. Stack into Master Tensors
    print(f"\nStep 4: Stacking Tensors...")
    final_inputs = torch.stack(all_inputs)
    final_targets = torch.stack(all_targets)
    
    print(f"   > Input Shape:  {final_inputs.shape} (N, 6, 256, 256)")
    print(f"   > Target Shape: {final_targets.shape} (N, 2, 256, 256)")
    
    # 5. Save to Disk
    print(f"\nStep 5: Saving to {OUTPUT_FILE}...")
    torch.save({
        'inputs': final_inputs,
        'targets': final_targets,
        'K': dataset.K  # Save K so we know how to un-normalize later!
    }, OUTPUT_FILE)
    
    print("✅ Dataset built and saved successfully.")

if __name__ == "__main__":
    build_and_save()