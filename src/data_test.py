import os
import torch
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader
import shutil

# --- Import your modules ---
# Ensure this matches your local filename for the generator
import generate_dataset_flexible as generator 
from data_processor import FluidDataset   
from model import ResUNet # Import model definition from your notebook or file

# --- Configuration ---
MODEL_PATH = "checkpoints/PINN_NS_best_3.pth" 
STATS_FILE = "normalization_stats.json"      
TEST_DATA_DIR = "../data_test"               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_test_data():
    """Generates 1 new simulation for testing and SAVES it."""
    print(f"⚙️  Generating test simulation in {TEST_DATA_DIR}...")
    
    # 1. Clean Directory
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)
    os.makedirs(TEST_DATA_DIR)
    
    # 2. Run Generation (Native 64x64)
    # capture the data!
    sim_data, scenario = generator.run_simulation(0, 1, TEST_DATA_DIR, 64, int(time.time()))
    
    # 3. SAVE THE DATA (This was missing!)
    save_path = os.path.join(TEST_DATA_DIR, "test_sim.npz")
    # Save as both LR and HR (since we just need input for testing)
    np.savez_compressed(save_path, hr=sim_data, lr=sim_data)
    
    print(f"✅ Simulation generated and saved to {save_path}.")

def run_inference():
    # 1. Generate Fresh Data
    generate_test_data()

    # 2. Load Data using Processor
    # This handles the loading, normalizing, and upscaling for us
    dataset = FluidDataset(
        data_dir=TEST_DATA_DIR, 
        stats_file=STATS_FILE, 
        target_res=256,
        cache_data=True
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Load Model
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = ResUNet(in_channels=4, out_channels=4).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # 4. Inference
    print("🚀 Running Inference...")
    
    results = {
        'lr_input': [],
        'sr_pred': [],
        'hr_target': []
    }

    with torch.no_grad():
        for lr, hr in loader:
            lr = lr.to(DEVICE)
            
            # Predict
            sr = model(lr)
            
            # Move to CPU and un-normalize
            # We need to manually un-normalize using the dataset's stored K values
            # shape: (1, 4, H, W)
            lr_np = lr.cpu().numpy()[0]
            sr_np = sr.cpu().numpy()[0]
            hr_np = hr.cpu().numpy()[0]
            
            # Un-normalize Channels (u,v / K_vel), (p / K_pres), (s / K_smoke)
            for arr in [lr_np, sr_np, hr_np]:
                arr[0:2] *= dataset.K_vel
                arr[2]   *= dataset.K_pres
                arr[3]   *= dataset.K_smoke
            
            results['lr_input'].append(lr_np)
            results['sr_pred'].append(sr_np)
            results['hr_target'].append(hr_np)

    # 5. Visualize
    print("🎬 Creating Physics Comparison GIF...")
    create_physics_animation(results)

def create_physics_animation(results):
    # Unpack lists
    # Each element is (4, H, W). Stack to (Time, 4, H, W)
    lr = np.stack(results['lr_input'])
    sr = np.stack(results['sr_pred'])
    hr = np.stack(results['hr_target'])
    
    frames = lr.shape[0]
    
    # Extract quantities
    # Velocity Magnitude
    def get_vel(x): return np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    vel_lr, vel_sr, vel_hr = get_vel(lr), get_vel(sr), get_vel(hr)
    
    # Pressure
    pres_lr, pres_sr, pres_hr = lr[:, 2], sr[:, 2], hr[:, 2]
    
    # Smoke
    smoke_lr, smoke_sr, smoke_hr = lr[:, 3], sr[:, 3], hr[:, 3]

    # Global Max/Min for consistent colors
    v_max = max(np.max(vel_lr), np.max(vel_sr), np.max(vel_hr))
    p_max = max(np.max(np.abs(pres_lr)), np.max(np.abs(pres_sr)), np.max(np.abs(pres_hr)))
    s_max = max(np.max(smoke_lr), np.max(smoke_sr), np.max(smoke_hr))

    # --- Plotting (3 Rows x 3 Cols) ---
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    
    # Columns: Input (LR), Prediction (SR), Ground Truth (HR)
    # Rows: Velocity, Pressure, Smoke
    
    # Row 1: Velocity
    ax_v = axes[0]
    im_v_lr = ax_v[0].imshow(vel_lr[0], vmin=0, vmax=v_max, cmap='inferno', origin='lower')
    im_v_sr = ax_v[1].imshow(vel_sr[0], vmin=0, vmax=v_max, cmap='inferno', origin='lower')
    im_v_hr = ax_v[2].imshow(vel_hr[0], vmin=0, vmax=v_max, cmap='inferno', origin='lower')
    
    ax_v[0].set_title("Input (Upscaled)", fontweight='bold')
    ax_v[1].set_title("SR Prediction", fontweight='bold', color='blue')
    ax_v[2].set_title("Ground Truth", fontweight='bold')
    ax_v[0].set_ylabel("Velocity", fontsize=12)

    # Row 2: Pressure
    ax_p = axes[1]
    im_p_lr = ax_p[0].imshow(pres_lr[0], vmin=-p_max, vmax=p_max, cmap='RdBu', origin='lower')
    im_p_sr = ax_p[1].imshow(pres_sr[0], vmin=-p_max, vmax=p_max, cmap='RdBu', origin='lower')
    im_p_hr = ax_p[2].imshow(pres_hr[0], vmin=-p_max, vmax=p_max, cmap='RdBu', origin='lower')
    ax_p[0].set_ylabel("Pressure", fontsize=12)

    # Row 3: Smoke
    ax_s = axes[2]
    im_s_lr = ax_s[0].imshow(smoke_lr[0], vmin=0, vmax=s_max, cmap='magma', origin='lower')
    im_s_sr = ax_s[1].imshow(smoke_sr[0], vmin=0, vmax=s_max, cmap='magma', origin='lower')
    im_s_hr = ax_s[2].imshow(smoke_hr[0], vmin=0, vmax=s_max, cmap='magma', origin='lower')
    ax_s[0].set_ylabel("Smoke Density", fontsize=12)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Animation
    def update(f):
        # Velocity
        im_v_lr.set_data(vel_lr[f])
        im_v_sr.set_data(vel_sr[f])
        im_v_hr.set_data(vel_hr[f])
        # Pressure
        im_p_lr.set_data(pres_lr[f])
        im_p_sr.set_data(pres_sr[f])
        im_p_hr.set_data(pres_hr[f])
        # Smoke
        im_s_lr.set_data(smoke_lr[f])
        im_s_sr.set_data(smoke_sr[f])
        im_s_hr.set_data(smoke_hr[f])
        return [im_v_lr, im_v_sr, im_v_hr, im_p_lr, im_p_sr, im_p_hr, im_s_lr, im_s_sr, im_s_hr]

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    out_path = "inference_physics_test.gif"
    ani.save(out_path, writer='pillow', fps=20)
    print(f"✨ Animation saved to {out_path}")

if __name__ == "__main__":
    run_inference()