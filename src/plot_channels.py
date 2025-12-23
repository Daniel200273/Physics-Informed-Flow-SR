import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def visualize_simulation_decomposition(file_path, frame_idx=100, output_base="analysis_results"):
    """
    Loads a simulation frame, visualizes decomposition, and saves INDIVIDUAL PNGs.
    Zoom window is fixed to 64x64 and shifted upwards.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found at {file_path}")
        return

    # 1. Load Data
    print(f"📂 Loading {file_path}...")
    try:
        data = np.load(file_path)
        if 'hr' in data:
            sim_data = data['hr']
        elif 'lr' in data:
            sim_data = data['lr']
        else:
            sim_data = data[list(data.keys())[0]]
    except Exception as e:
        print(f"❌ Failed to load file: {e}")
        return

    # Ensure shape is (Time, H, W, C)
    if sim_data.ndim == 4 and sim_data.shape[1] == 4:
        sim_data = sim_data.transpose(0, 2, 3, 1)

    # Handle Frame Index
    total_frames = sim_data.shape[0]
    if frame_idx >= total_frames:
        frame_idx = total_frames - 1

    # Extract Channels
    frame = sim_data[frame_idx]
    u = frame[..., 0]
    v = frame[..., 1]
    p = frame[..., 2]
    s = frame[..., 3]
    
    height, width = u.shape
    magnitude = np.sqrt(u**2 + v**2)

    # 2. Prepare Output Directory
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    save_dir = os.path.join(output_base, f"{file_name}_frame_{frame_idx}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"💾 Saving images to: {save_dir}/")

    # --- HELPER: Save Heatmaps ---
    def save_heatmap(arr, name, title, cmap='RdBu'):
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(arr, cmap=cmap, origin='lower')
        ax.set_title(title, fontsize=18, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{name}.png"), dpi=100)
        plt.close(fig)

    # 3. Save Individual Channels
    save_heatmap(u, "01_velocity_x", "Channel 0: Velocity X (u)")
    save_heatmap(v, "02_velocity_y", "Channel 1: Velocity Y (v)")
    save_heatmap(p, "03_pressure",   "Channel 2: Pressure (p)")
    save_heatmap(s, "04_smoke",      "Channel 3: Smoke Density (s)", cmap='magma')

    # 4. Save Full Vector Field
    fig, ax = plt.subplots(figsize=(10, 10))
    # Background magnitude
    ax.imshow(magnitude, extent=[0, width, 0, height], origin='lower', cmap='gray', alpha=0.3)
    
    # Vector Field (Decimated)
    Y, X = np.mgrid[0:height, 0:width]
    step = max(1, width // 32)
    ax.quiver(X[::step, ::step], Y[::step, ::step], 
              u[::step, ::step], v[::step, ::step], 
              magnitude[::step, ::step], 
              cmap='inferno', scale_units='xy', scale=0.5, width=0.004)
    
    ax.set_title("Full Velocity Vector Field", fontsize=18, fontweight='bold')
    ax.set_xlim(0, width); ax.set_ylim(0, height)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_vector_field_full.png"), dpi=100)
    plt.close(fig)

    # 5. Save Zoomed Vector Field (Fixed 64x64, Shifted Up)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- ZOOM CONFIGURATION ---
    zoom_w, zoom_h = 64, 64   # Fixed window size
    
    cx = width // 2
    cy = height // 2
    
    # Shift UP by 15% of height (to catch the plume)
    # Note: In matplotlib 'lower' origin, higher index = higher pixel position
    shift_y = int(height * 0.15) 
    cy += shift_y
    
    # Calculate slices ensuring we don't go out of bounds
    y_start = max(0, cy - zoom_h // 2)
    y_end   = min(height, cy + zoom_h // 2)
    x_start = max(0, cx - zoom_w // 2)
    x_end   = min(width, cx + zoom_w // 2)
    
    # Crop Data
    u_zoom = u[y_start:y_end, x_start:x_end]
    v_zoom = v[y_start:y_end, x_start:x_end]
    mag_zoom = magnitude[y_start:y_end, x_start:x_end]
    X_zoom = X[y_start:y_end, x_start:x_end]
    Y_zoom = Y[y_start:y_end, x_start:x_end]
    
    # Background
    ax.imshow(mag_zoom, extent=[x_start, x_end, y_start, y_end], 
              origin='lower', cmap='gray', alpha=0.3)
    
    # Quiver (Less decimated for detail)
    zoom_step = 1 # Show every arrow? Or maybe 2 if it's too crowded
    ax.quiver(X_zoom[::zoom_step, ::zoom_step], Y_zoom[::zoom_step, ::zoom_step], 
              u_zoom[::zoom_step, ::zoom_step], v_zoom[::zoom_step, ::zoom_step], 
              color='red', # Red arrows pop against gray
              scale_units='xy', scale=0.4, width=0.005)
    
    ax.set_title(f"Zoomed Vector Field ({zoom_w}x{zoom_h})", fontsize=18, fontweight='bold')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(y_start, y_end)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_vector_field_zoom_64x64.png"), dpi=100)
    plt.close(fig)
    
    print(f"✅ Analysis complete. Check the folder: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="../data_test/test_sim.npz", help="Path to .npz file")
    parser.add_argument("--frame", type=int, default=100, help="Frame index to visualize")
    parser.add_argument("--out", type=str, default="analysis_output", help="Output root directory")
    args = parser.parse_args()
    
    visualize_simulation_decomposition(args.file, args.frame, args.out)