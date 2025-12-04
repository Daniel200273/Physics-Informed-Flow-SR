import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Load the Data
try:
    hr_data = np.load('../data/train_hr.npy') # Shape: (frames, 64, 64, 2)
    lr_data = np.load('../data/train_lr.npy') # Shape: (frames, 16, 16, 2)
    print(f"✅ Data Loaded.")
    print(f"High Res Shape: {hr_data.shape}")
    print(f"Low Res Shape:  {lr_data.shape}")
except FileNotFoundError:
    print("❌ Error: Data not found. Run generate.py first!")
    exit()

# 2. Helper to convert Velocity (x,y) -> Speed (Scalar)
def get_speed(velocity_field):
    # Magnitude = sqrt(x^2 + y^2)
    return np.sqrt(velocity_field[..., 0]**2 + velocity_field[..., 1]**2)

# 3. Setup the Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Fluid Simulation: Low Res vs High Res')

# We use 'imshow' to display the grid
# vmin/vmax ensures the colors stay consistent as speed increases
max_speed = np.max(get_speed(hr_data))
im_lr = ax1.imshow(get_speed(lr_data[0]), cmap='inferno', origin='lower', vmin=0, vmax=max_speed)
im_hr = ax2.imshow(get_speed(hr_data[0]), cmap='inferno', origin='lower', vmin=0, vmax=max_speed)

ax1.set_title("Input (Low Res)")
ax2.set_title("Target (High Res)")

# 4. Animation Function
def update(frame):
    # Update Low Res Image
    speed_lr = get_speed(lr_data[frame])
    im_lr.set_data(speed_lr)
    
    # Update High Res Image
    speed_hr = get_speed(hr_data[frame])
    im_hr.set_data(speed_hr)
    
    ax1.set_title(f"Input (Low Res) - Frame {frame}")
    return im_lr, im_hr

# 5. Run Animation
print("🎥 Rendering animation...")
ani = animation.FuncAnimation(fig, update, frames=len(hr_data), interval=50, blit=False)

plt.show()