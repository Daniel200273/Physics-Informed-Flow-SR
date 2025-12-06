import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader

# --- Import your modules ---
# Assuming these are in the python path
import data_generator_test as generator 
from data_processor import FluidDataset   
from model import ResUNet

# --- Configuration ---
MODEL_PATH = "checkpoints/Baseline_epoch_50.pth" 
STATS_FILE = "normalization_stats.json"      
TEST_DATA_DIR = "../data_test"               
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_test_data():
    """Generates 1 new simulation for testing and tracks time."""
    print(f"⚙️  Generating test simulation in {TEST_DATA_DIR}...")
    
    # Run simulation ID 0 (1 simulation)
    # The new generator script (data_generator_test.py) saves to the output_dir we pass
    start_time = time.time()
    try:
        # Note: The new generator takes (sim_id, n_sims, output_dir)
        generator.run_simulation(0, 1, TEST_DATA_DIR)
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        exit()
    end_time = time.time()
    
    # Estimate HR vs LR time roughly (since they run simultaneously in the new script)
    # The solver steps are the bottleneck.
    # Total time measures the full physics solve for both resolutions.
    total_time = end_time - start_time
    print(f"✅ Simulation generated in {total_time:.2f} seconds.")
    return total_time

def run_inference():
    # 1. Setup Folders
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
    
    # Clean up old test data
    for f in os.listdir(TEST_DATA_DIR):
        if f.endswith('.npz'):
            os.remove(os.path.join(TEST_DATA_DIR, f))

    # 2. Generate Data & Measure Time
    sim_time = generate_test_data()

    # 3. Load Data
    print("⏳ Loading test dataset...")
    try:
        test_dataset = FluidDataset(
            data_dir=TEST_DATA_DIR, 
            stats_file=STATS_FILE, 
            cache_data=True
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    K = test_dataset.K
    print(f"   > Scaling Factor K: {K}")

    # 4. Load Model
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = ResUNet(in_channels=6, out_channels=2).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 5. Inference Loop
    print("🚀 Running Inference...")
    
    predictions = []
    ground_truths = []
    inputs_lr = []
    
    inference_start = time.time()

    with torch.no_grad():
        for i, (lr_input, hr_target) in enumerate(test_loader):
            lr_input = lr_input.to(DEVICE)
            
            # Forward Pass (SR)
            sr_output = model(lr_input)
            
            # Move to CPU for storage
            sr_np = sr_output.cpu().numpy()     
            hr_np = hr_target.cpu().numpy()     
            lr_np = lr_input.cpu().numpy()      

            predictions.append(sr_np[0])
            ground_truths.append(hr_np[0])
            
            # Middle frame for visualization
            inputs_lr.append(lr_np[0, 2:4, :, :])

    inference_end = time.time()
    inference_time = inference_end - inference_start
    
    avg_inference_fps = len(predictions) / inference_time
    
    print(f"⏱️  Performance Stats:")
    print(f"   > HR/LR Simulation (Physics): {sim_time:.2f} s")
    print(f"   > SR Inference (Neural Net):  {inference_time:.2f} s")
    print(f"   > Inference Speed:            {avg_inference_fps:.1f} FPS")

    # 6. Create Animation
    print("🎬 Creating GIF...")
    visualize_results(inputs_lr, predictions, ground_truths, K, sim_time, inference_time)

def visualize_results(lr_list, sr_list, hr_list, K, sim_time, infer_time):
    """
    Creates a side-by-side comparison GIF with timing info.
    """
    frames = len(sr_list)
    
    # Helper to calculate velocity magnitude
    def get_mag(u_vec):
        u = u_vec[0] * K
        v = u_vec[1] * K
        return np.sqrt(u**2 + v**2)

    # Determine global max for consistent colors
    max_val = 0
    for item in hr_list:
        max_val = max(max_val, np.max(get_mag(item)))
    if max_val == 0: max_val = 1.0

    # Setup Plot (Increased height for titles)
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    ax_lr, ax_sr, ax_hr = axes

    # Placeholders
    dummy = np.zeros((256, 256))
    im_lr = ax_lr.imshow(dummy, cmap='inferno', vmin=0, vmax=max_val, origin='lower')
    im_sr = ax_sr.imshow(dummy, cmap='inferno', vmin=0, vmax=max_val, origin='lower')
    im_hr = ax_hr.imshow(dummy, cmap='inferno', vmin=0, vmax=max_val, origin='lower')

    # Titles and Formatting
    ax_lr.set_title("Low Res Simulation\n(Input)", fontsize=12)
    ax_sr.set_title(f"Super Resolution\n(Inference: {infer_time:.2f}s total)", fontsize=12, color='darkblue')
    ax_hr.set_title(f"High Res Simulation\n(Physics: {sim_time:.2f}s total)", fontsize=12)

    for ax in axes:
        ax.axis('off')

    # Add Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # x, y, width, height
    fig.colorbar(im_hr, cax=cbar_ax, label='Velocity Magnitude (m/s)')

    # Add Frame Counter (Top Left)
    fig.text(0.02, 0.95, f"Comparison", fontsize=16, fontweight='bold')
    txt_frame = fig.text(0.5, 0.95, "", fontsize=14, ha='center')

    # Adjust layout to prevent overlap
    plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.9, wspace=0.1)

    def update(frame_idx):
        im_lr.set_data(get_mag(lr_list[frame_idx]))
        im_sr.set_data(get_mag(sr_list[frame_idx]))
        im_hr.set_data(get_mag(hr_list[frame_idx]))
        txt_frame.set_text(f"Frame {frame_idx}/{frames}")
        return im_lr, im_sr, im_hr

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    save_path = "inference_result.gif"
    ani.save(save_path, writer='pillow', fps=20)
    print(f"✨ Animation saved to {save_path}")

if __name__ == "__main__":
    run_inference()