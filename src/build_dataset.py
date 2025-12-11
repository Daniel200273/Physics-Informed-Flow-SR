import os
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your custom modules
# Ensure data_processor.py contains the "Flexible" FluidDataset class
from data_processor import FluidDataset
from k_finder import calculate_global_max

def build_and_save(data_dir, stats_file, output_file, target_res):
    print(f"🏗️  STARTING DATASET BUILD PIPELINE")
    print(f"   > Input:  {data_dir}")
    print(f"   > Output: {output_file}")
    print(f"   > Target Resolution: {target_res}x{target_res}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 1. Pre-processing: Calculate Max Velocity (K)
    print(f"\nStep 1: Calculating Global Normalization Factor (K)...")
    try:
        calculate_global_max(data_dir=data_dir, output_file=stats_file)
    except Exception as e:
        print(f"⚠️ Warning during K calculation: {e}")
        print("   If normalization_stats.json already exists, we will use it.")
    
    # 2. Instantiate Dataset
    # We set noise_std=0.0 because we want the saved dataset to be clean.
    # Augmentation should happen during training, not baked into the file.
    print(f"\nStep 2: Loading Raw Data into Memory...")
    try:
        dataset = FluidDataset(
            data_dir=data_dir, 
            stats_file=stats_file, 
            target_res=target_res, 
            noise_std=0.0, 
            cache_data=True
        )
    except TypeError:
        # Fallback if FluidDataset hasn't been updated with target_res yet
        print("⚠️ Warning: FluidDataset does not accept target_res/noise_std. Using legacy init.")
        dataset = FluidDataset(data_dir=data_dir, stats_file=stats_file)

    print(f"   > Loaded {len(dataset)} total samples.")
    
    # 3. Process All Frames
    print(f"\nStep 3: Processing Tensors (Upscaling to {target_res}x{target_res} & Normalizing)...")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    all_inputs = []
    all_targets = []
    
    for inputs, targets in tqdm(loader, desc="Processing"):
        # inputs shape: (1, 6, 256, 256) -> squeeze to (6, 256, 256)
        all_inputs.append(inputs.squeeze(0))
        all_targets.append(targets.squeeze(0))
        
    # 4. Stack into Master Tensors
    print(f"\nStep 4: Stacking Tensors...")
    final_inputs = torch.stack(all_inputs)
    final_targets = torch.stack(all_targets)
    
    print(f"   > Input Shape:  {final_inputs.shape} (N, 6, {target_res}, {target_res})")
    print(f"   > Target Shape: {final_targets.shape} (N, 2, 256, 256)")
    
    # 5. Save to Disk
    print(f"\nStep 5: Saving to {output_file}...")
    torch.save({
        'inputs': final_inputs,
        'targets': final_targets,
        'K': dataset.K
    }, output_file)
    
    print("✅ Dataset built and saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build processed .pt dataset from .npz files.")
    
    # Arguments for flexibility
    parser.add_argument("--data_dir", type=str, default="../data", help="Folder containing raw .npz files")
    parser.add_argument("--out", type=str, default="../dataset/processed_dataset.pt", help="Path to save the output .pt file")
    parser.add_argument("--stats", type=str, default="normalization_stats.json", help="Path to normalization stats json")
    parser.add_argument("--target_res", type=int, default=256, help="Target resolution for input upscaling (default: 256)")
    
    args = parser.parse_args()
    
    build_and_save(args.data_dir, args.stats, args.out, args.target_res)