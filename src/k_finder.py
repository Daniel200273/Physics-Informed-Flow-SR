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

    print(f"🔍 Scanning {len(files)} files for global maximums (Velocity, Pressure, Smoke)...")
    
    max_vel = 0.0
    max_pres = 0.0
    max_smoke = 0.0
    
    for i, f in enumerate(files):
        try:
            with np.load(f) as data:
                # hr shape: (T, H, W, 4) -> [u, v, p, s]
                hr_data = data['hr'] 
                
                # 1. Velocity (Channels 0, 1)
                vel_mag = np.abs(hr_data[..., 0:2])
                local_max_vel = np.max(vel_mag)
                if local_max_vel > max_vel: max_vel = local_max_vel
                
                # 2. Pressure (Channel 2)
                pres_mag = np.abs(hr_data[..., 2])
                local_max_pres = np.max(pres_mag)
                if local_max_pres > max_pres: max_pres = local_max_pres
                
                # 3. Smoke (Channel 3)
                smoke_mag = np.abs(hr_data[..., 3])
                local_max_smoke = np.max(smoke_mag)
                if local_max_smoke > max_smoke: max_smoke = local_max_smoke
                    
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")

    # Safety buffer
    K_vel = float(np.ceil(max_vel)) if max_vel > 0 else 1.0
    K_pres = float(np.ceil(max_pres)) if max_pres > 0 else 1.0
    K_smoke = float(np.ceil(max_smoke)) if max_smoke > 0 else 1.0
    
    stats = {
        "scaling_factor": K_vel,       # Kept for backward compatibility
        "K_vel": K_vel,
        "K_pres": K_pres,
        "K_smoke": K_smoke,
        "raw_max_vel": float(max_vel),
        "raw_max_pres": float(max_pres),
        "raw_max_smoke": float(max_smoke)
    }

    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ Stats saved to {output_file}")
    print(f"   K_vel: {K_vel} | K_pres: {K_pres} | K_smoke: {K_smoke}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data', help='Path to .npz files')
    args = parser.parse_args()
    
    calculate_global_max(args.data_dir)