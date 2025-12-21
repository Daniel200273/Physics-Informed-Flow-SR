import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader

# --- Import your modules ---
import data_generator_test as generator
from data_processor import FluidDataset
from model import ResUNet

# --- Configuration ---
MODEL_PATH = "checkpoints/PINN_best(3).pth"
STATS_FILE = "normalization_stats.json"
TEST_DATA_DIR = "../data_test"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_test_data():
    """Generates 1 new simulation for testing and tracks time."""
    print(f"⚙️  Generating test simulation in {TEST_DATA_DIR}...")

    # 1. Clean up old data
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
    for f in os.listdir(TEST_DATA_DIR):
        if f.endswith('.npz'):
            os.remove(os.path.join(TEST_DATA_DIR, f))

    # 2. Run Generation & Time it
    start_time = time.perf_counter()
    try:
        # Pass (sim_id, n_sims, output_dir)
        generator.run_simulation(0, 1, TEST_DATA_DIR)
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        exit()
    end_time = time.perf_counter()

    total_gen_time = end_time - start_time
    print(f"✅ Simulation generated in {total_gen_time:.2f} seconds.")

    return total_gen_time


def run_inference():
    # 1. Generate Data & Measure Time
    gen_time = generate_test_data()

    # 2. Load Data
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

    # 3. Load Model
    print(f"🧠 Loading model from {MODEL_PATH}...")
    model = ResUNet(in_channels=6, out_channels=2).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found at {MODEL_PATH}")
        return

    # Load checkpoint correctly: expect a dict with 'model_state_dict'
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        # Fallback for raw state_dict files
        state = ckpt
    # Use strict=True by default; set to False only if minor key mismatches occur
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        # If keys slightly differ (e.g., missing buffers), try non-strict load
        print(f"[warn] Strict load failed: {e}\nAttempting non-strict load...")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[info] Missing keys: {missing}\n[info] Unexpected keys: {unexpected}")
    model.eval()

    # 4. Inference Loop
    print("🚀 Running Inference...")

    predictions = []
    ground_truths = []
    inputs_lr_bilinear = []  # The upscaled input the model sees
    inputs_lr_raw = []      # The actual 32x32 pixels

    inference_start = time.perf_counter()

    with torch.no_grad():
        for i, (lr_input, hr_target) in enumerate(test_loader):
            lr_input = lr_input.to(DEVICE)

            # Forward Pass (SR)
            sr_output = model(lr_input)

            # Move to CPU
            sr_np = sr_output.cpu().numpy()
            hr_np = hr_target.cpu().numpy()
            lr_np = lr_input.cpu().numpy()

            predictions.append(sr_np[0])
            ground_truths.append(hr_np[0])

            # Extract Middle Frame (channels 2-3)
            # lr_np is already upscaled to 256x256 by FluidDataset
            mid_frame_bilinear = lr_np[0, 2:4, :, :]
            inputs_lr_bilinear.append(mid_frame_bilinear)

            # Recover Raw 32x32 by slicing (stride 8)
            # Since FluidDataset used F.interpolate(scale=8), we can just sample every 8th pixel
            mid_frame_raw = mid_frame_bilinear[:, ::8, ::8]
            inputs_lr_raw.append(mid_frame_raw)

    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start

    avg_inference_fps = len(predictions) / inference_time

    print(f"⏱️  Performance Stats:")
    print(f"   > Generation (HR+LR Physics): {gen_time:.4f} s")
    print(f"   > SR Inference (Neural Net):  {inference_time:.4f} s")
    print(f"   > Inference Speed:            {avg_inference_fps:.1f} FPS")

    # 5. Create Animation
    print("🎬 Creating GIF...")
    visualize_results(inputs_lr_raw, inputs_lr_bilinear,
                      predictions, ground_truths, K, gen_time, inference_time)


def visualize_results(raw_list, bilinear_list, sr_list, hr_list, K, gen_time, infer_time):
    """
    Creates a 4-panel comparison GIF.
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
    if max_val == 0:
        max_val = 1.0

    # Setup Plot (4 columns now)
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    ax_raw, ax_bi, ax_sr, ax_hr = axes

    # Placeholders
    # Raw is 32x32
    im_raw = ax_raw.imshow(np.zeros((32, 32)), cmap='inferno',
                           vmin=0, vmax=max_val, origin='lower', interpolation='nearest')
    # Others are 256x256
    dummy_hr = np.zeros((256, 256))
    im_bi = ax_bi.imshow(dummy_hr, cmap='inferno', vmin=0,
                         vmax=max_val, origin='lower')
    im_sr = ax_sr.imshow(dummy_hr, cmap='inferno', vmin=0,
                         vmax=max_val, origin='lower')
    im_hr = ax_hr.imshow(dummy_hr, cmap='inferno', vmin=0,
                         vmax=max_val, origin='lower')

    # Titles and Formatting (Padded to avoid overlap)
    ax_raw.set_title("Input (32x32)\nNative Resolution", fontsize=12, pad=15)
    ax_bi.set_title(
        "Rough Upscaling (256x256)\nBilinear Interpolation", fontsize=12, pad=15)
    ax_sr.set_title(f"ResUNet Super-Res\n(Inference: {infer_time:.2f}s)",
                    fontsize=12, color='darkblue', fontweight='bold', pad=15)
    ax_hr.set_title(
        f"Ground Truth High-Res\n(Sim Time: {gen_time:.2f}s)", fontsize=12, pad=15)

    for ax in axes:
        ax.axis('off')

    # Add Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.7])  # x, y, width, height
    fig.colorbar(im_hr, cax=cbar_ax, label='Velocity Magnitude (m/s)')

    # Add Frame Counter
    fig.text(0.02, 0.95, "Fluid Super-Resolution Comparison",
             fontsize=16, fontweight='bold')
    txt_frame = fig.text(0.5, 0.95, "", fontsize=14, ha='center', color='gray')

    # Adjust layout to prevent overlap
    plt.subplots_adjust(top=0.80, bottom=0.05, left=0.02,
                        right=0.90, wspace=0.1)

    def update(frame_idx):
        im_raw.set_data(get_mag(raw_list[frame_idx]))
        im_bi.set_data(get_mag(bilinear_list[frame_idx]))
        im_sr.set_data(get_mag(sr_list[frame_idx]))
        im_hr.set_data(get_mag(hr_list[frame_idx]))
        txt_frame.set_text(f"Frame {frame_idx}/{frames}")
        return im_raw, im_bi, im_sr, im_hr

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=50, blit=False)

    save_path = "inference_result_4panel_new_new.gif"
    ani.save(save_path, writer='pillow', fps=20)
    print(f"✨ Animation saved to {save_path}")


if __name__ == "__main__":
    run_inference()
