import os
import glob
import numpy as np
import json
import argparse

def calculate_global_max(data_dir, output_file='normalization_stats.json'):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    
    if not files:
        print(f"❌ No .npz files found in {data_dir}")
        return

    print(f"🔍 Scanning {len(files)} files for global maximum...")
    
    global_max_val = 0.0
    
    for i, f in enumerate(files):
        try:
            with np.load(f) as data:
                # We only check 'hr' because it contains the sharpest peaks
                hr_data = data['hr'] # Shape: (T, H, W, C)
                
                # Calculate max absolute velocity component
                # We use abs() because flow can be negative (left/down)
                local_max = np.max(np.abs(hr_data))
                
                if local_max > global_max_val:
                    global_max_val = local_max
                    
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    # Add a tiny safety buffer (e.g., 1%) to ensure no value hits exactly 1.0 or -1.0
    # or just ceil it for a clean number.
    scaling_factor = float(np.ceil(global_max_val))
    
    stats = {
        "global_max_velocity": float(global_max_val),
        "scaling_factor": scaling_factor
    }

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ Stats saved to {output_file}")
    print(f"   Global Max: {global_max_val}")
    print(f"   Selected K (Scaling Factor): {scaling_factor}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to .npz files')
    args = parser.parse_args()
    
    calculate_global_max(args.data_dir)