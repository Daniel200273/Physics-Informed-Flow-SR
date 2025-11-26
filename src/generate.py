from phi.torch.flow import *
import numpy as np
import os
import warnings
# --- GIF SPECIFIC IMPORTS ---
from phi import vis
from phi.vis import plot
import matplotlib.pyplot as plt
import imageio
# ----------------------------

# --- SETUP ---
warnings.filterwarnings('ignore')
math.set_global_precision(64) 

H, W = 128, 128
ds_factor = 4 
BOUNDS = Box(x=100, y=100)

# Inflow Setup
center_pos = tensor([32, 5], channel(vector="x,y"))
sphere_geo = Sphere(center=center_pos, radius=3)
inflow_mask = CenteredGrid(sphere_geo, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
INFLOW = inflow_mask * 0.2

# Initialize Fields
velocity = StaggeredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
smoke    = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
pressure = None 

# Data Storage
hr_data = []
lr_data = []
# GIF STORAGE: Store the visualization data (e.g., density or speed)
gif_frames = []

print(f"🚀 Starting Simulation (Aligned with Tutorial)...")

# --- DEFINING THE STEP FUNCTION (JIT COMPILED) ---
@math.jit_compile
def step_function(s, v, p, dt):
    # 1. Advection
    s = advect.semi_lagrangian(s, v, dt) + INFLOW
    v = advect.semi_lagrangian(v, v, dt)

    # 2. Buoyancy
    buoyancy_force = s * (0, 0.5) @ v
    v = v + buoyancy_force

    # 3. Pressure Solve
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3, max_iterations=1000, x0=p)
    v, p = fluid.make_incompressible(v, solve=solver)
    
    return s, v, p

# --- MAIN LOOP ---
dt = 0.2

for i in range(200):
    try:
        smoke, velocity, pressure = step_function(smoke, velocity, pressure, dt)
        
        # Data Processing for NumPy Arrays
        vel_centered = velocity.at_centers()
        v_hr = vel_centered.values.numpy('y,x,vector')
        
        # Sanity Check for NaN
        if np.isnan(v_hr).any():
            print(f"❌ NaN at frame {i}. Stopping.")
            break

        v_lr = v_hr[::ds_factor, ::ds_factor, :] 

        hr_data.append(v_hr)
        lr_data.append(v_lr)

        # --- GIF PROCESSING (Extract Speed) ---
        # We visualize the speed magnitude, which is the visual equivalent of 'smoke' rising
        speed_magnitude = math.sqrt(v_hr[..., 0]**2 + v_hr[..., 1]**2)
        gif_frames.append(speed_magnitude)
        # ------------------------------------

        if i % 10 == 0:
            print(f"Frame {i}/200 generated.")
            
    except Exception as e:
        print(f"⚠️ Error at frame {i}: {e}")
        continue

# ---------------------------------------------
## FINAL SAVING PHASE
# ---------------------------------------------

# Create the data directory (assuming script is in root/src)
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# Stack lists into arrays: (Frames, H, W, 2)
hr_array = np.array(hr_data)
lr_array = np.array(lr_data)

# 1. Save NumPy Datasets
print("💾 Saving NumPy datasets...")
np.save(f"{DATA_DIR}/train_hr.npy", hr_array)
np.save(f"{DATA_DIR}/train_lr.npy", lr_array)

def save_simulation_gif(vector_data, filename, max_speed_ref=None):
    """
    vector_data: Shape (Frames, H, W, 2)
    filename: Output filename
    max_speed_ref: Optional max value for color normalization (to keep scales consistent)
    """
    print(f"🎥 Rendering {filename}...")
    
    # Calculate Speed Magnitude: sqrt(u^2 + v^2)
    # Shape becomes (Frames, H, W)
    speed_data = np.sqrt(vector_data[..., 0]**2 + vector_data[..., 1]**2)
    
    # Determine Color Scale
    # We use the provided reference max (usually from HR) so colors match across GIFs
    if max_speed_ref is None:
        max_val = np.max(speed_data)
    else:
        max_val = max_speed_ref
        
    # Avoid division by zero
    if max_val == 0: max_val = 1.0
        
    # Normalize to 0.0 - 1.0
    norm_data = speed_data / max_val
    
    # Apply Colormap (Magma: Black -> Purple -> Orange -> White)
    cmap = plt.get_cmap('magma')
    colored_data = cmap(norm_data)
    
    # Convert to RGB Integers (0-255)
    rgb_data = (colored_data[..., :3] * 255).astype(np.uint8)
    
    # Save GIF
    save_path = f"{DATA_DIR}/{filename}"
    imageio.mimsave(save_path, rgb_data, fps=15, loop=0)
    print(f"✅ Saved to {save_path}")

# 3. Generate and Save Both GIFs
if len(hr_array) > 0:
    # Calculate global max speed from HR data to ensure consistent coloring
    # This implies that "Orange" in LR means the same speed as "Orange" in HR
    global_max_speed = np.max(np.sqrt(hr_array[..., 0]**2 + hr_array[..., 1]**2))
    
    # Save High Res GIF
    save_simulation_gif(hr_array, "simulation_hr.gif", max_speed_ref=global_max_speed)
    
    # Save Low Res GIF
    save_simulation_gif(lr_array, "simulation_lr.gif", max_speed_ref=global_max_speed)

else:
    print("⚠️ No data generated.")

print("✅ Done! All files saved.")