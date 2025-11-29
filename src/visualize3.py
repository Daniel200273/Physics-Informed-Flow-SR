import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

print("=== VISUALIZE3.PY - Adapted for generate3.py data ===")

try:
    # 1. Load the Data - Matches generate3.py output files
    hrdataset = np.load('../data/trainhr.npy')  # Shape: (4000, 128, 128, 2)
    lrdataset = np.load('../data/trainlr.npy')  # Shape: (4000, 32, 32, 2)

    print("Data Loaded.")
    print(f"Dataset Shape HR: {hrdataset.shape}")  # e.g., (4000, 128, 128, 2)
    print(f"Dataset Shape LR: {lrdataset.shape}")  # e.g., (4000, 32, 32, 2)

    # Data is already flattened (TotalFrames, H, W, 2) - no reshape needed
    hrdata = hrdataset
    lrdata = lrdataset
    print(f"Flattened for animation: HR {hrdata.shape}, LR {lrdata.shape}")

except FileNotFoundError:
    print("Error: Data not found. Run generate3.py first!")
    exit()

# 2. Helper to convert Velocity (x,y) -> Speed (Scalar)


def get_speed(velocity_field):
    """Magnitude: sqrt(x^2 + y^2)"""
    return np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)


# 3. Setup the Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle('Dataset Preview - 4000 Frames from 20 Simulations', fontsize=16)

# Calculate global max speed to fix color scale (no flickering)
max_speed = np.max(get_speed(hrdata))
print(f"Global max speed: {max_speed:.2f}")

# Initial plots
im_lr = ax1.imshow(get_speed(lrdata[0]), cmap='magma', origin='lower',
                   vmin=0, vmax=max_speed)
im_hr = ax2.imshow(get_speed(hrdata[0]), cmap='magma', origin='lower',
                   vmin=0, vmax=max_speed)

ax1.set_title('Input Low Res (32x32)')
ax2.set_title('Target High Res (128x128)')
ax1.axis('off')
ax2.axis('off')

# 4. Animation Function


def update(frame_idx):
    """Update animation for frame"""
    total_frames = len(hrdata)
    progress = frame_idx / total_frames

    # Update image data
    im_lr.set_data(get_speed(lrdata[frame_idx]))
    im_hr.set_data(get_speed(hrdata[frame_idx]))

    # Update titles with frame info
    ax1.set_title(
        f'Low Res Input (32x32) - Frame {frame_idx+1}/{total_frames}')
    ax2.set_title(
        f'High Res Target (128x128) - Frame {frame_idx+1}/{total_frames}')

    # Main title with progress
    fig.suptitle(f'Dataset Preview - Frame {frame_idx+1}/{total_frames} ({progress:.1%})',
                 fontsize=16)

    return im_lr, im_hr


# 5. Run Animation
print("Rendering animation window...")
print("interval=50 means 50ms per frame (~20 fps)")
ani = animation.FuncAnimation(
    fig, update, frames=len(hrdata), interval=50, blit=False)
plt.tight_layout()
plt.show()

print("Animation complete! Close window to exit.")
