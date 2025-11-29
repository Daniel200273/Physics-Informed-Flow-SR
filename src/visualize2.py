import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Load the Data
try:
    # Note: Filenames changed to match your new generation script
    hr_dataset = np.load('../data/train_hr.npy') # Shape: (Sims, Frames, 128, 128, 2)
    lr_dataset = np.load('../data/train_lr.npy') # Shape: (Sims, Frames, 32, 32, 2)
    
    print(f"✅ Data Loaded.")
    print(f"Dataset Shape (HR): {hr_dataset.shape}") # e.g., (10, 100, 128, 128, 2)
    
    # EXTRACT METADATA
    num_sims = hr_dataset.shape[0]
    frames_per_sim = hr_dataset.shape[1]
    
    # FLATTEN THE DATA
    # We merge (Sims, Frames) into a single "Total_Frames" dimension
    # New Shape: (Total_Frames, H, W, 2)
    hr_data = hr_dataset.reshape(-1, hr_dataset.shape[2], hr_dataset.shape[3], 2)
    lr_data = lr_dataset.reshape(-1, lr_dataset.shape[2], lr_dataset.shape[3], 2)
    
    print(f"➡️  Flattened for animation: {hr_data.shape}")

except FileNotFoundError:
    print("❌ Error: Data not found. Run the generation script first!")
    exit()

# 2. Helper to convert Velocity (x,y) -> Speed (Scalar)
def get_speed(velocity_field):
    # Magnitude = sqrt(x^2 + y^2)
    return np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)

# 3. Setup the Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle(f'Dataset Preview: {num_sims} Simulations Merged', fontsize=16)

# Calculate global max speed to fix the color scale (so colors don't flicker)
max_speed = np.max(get_speed(hr_data))

# Initial Plots
im_lr = ax1.imshow(get_speed(lr_data[0]), cmap='magma', origin='lower', vmin=0, vmax=max_speed)
im_hr = ax2.imshow(get_speed(hr_data[0]), cmap='magma', origin='lower', vmin=0, vmax=max_speed)

ax1.set_title("Input (Low Res)")
ax2.set_title("Target (High Res)")
# Remove ticks for cleaner look
ax1.axis('off')
ax2.axis('off')

# 4. Animation Function
def update(frame_idx):
    # Calculate which simulation we are currently watching
    current_sim = frame_idx // frames_per_sim
    current_sim_frame = frame_idx % frames_per_sim
    
    # Update Image Data
    im_lr.set_data(get_speed(lr_data[frame_idx]))
    im_hr.set_data(get_speed(hr_data[frame_idx]))
    
    # Update Titles
    ax1.set_title(f"Low Res Input\n(32x32)")
    ax2.set_title(f"High Res Target\n(128x128)")
    
    # Main Title showing progress
    fig.suptitle(f'Simulation {current_sim + 1}/{num_sims} | Frame {current_sim_frame}/{frames_per_sim}', fontsize=16)
    
    return im_lr, im_hr

# 5. Run Animation
print("🎥 Rendering animation window...")
# interval=50 means 50ms per frame (20 fps)
ani = animation.FuncAnimation(fig, update, frames=len(hr_data), interval=50, blit=False)

plt.show()