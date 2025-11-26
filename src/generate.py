from phi.torch.flow import *
import numpy as np
import os
import warnings

# --- SETUP ---
warnings.filterwarnings('ignore')
math.set_global_precision(64) # Tutorial recommends 64-bit for accuracy

# Open boundaries (smoke can exit)
H, W = 128, 128
ds_factor = 4 # downscale factor for low-res simulation
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

print(f"🚀 Starting Simulation (Aligned with Tutorial)...")

# --- DEFINING THE STEP FUNCTION (JIT COMPILED) ---
# This compiles the physics logic into a fast, optimized function
@math.jit_compile
def step_function(s, v, p, dt):
    # 1. Advection
    s = advect.semi_lagrangian(s, v, dt) + INFLOW
    v = advect.semi_lagrangian(v, v, dt)

    # 2. Buoyancy
    buoyancy_force = s * (0, 0.5) @ v
    v = v + buoyancy_force

    # 3. Pressure Solve (The Fix)
    # We create the Solve object INSIDE the loop to pass 'x0=p' (Warm Start)
    # 'CG' is Conjugate Gradient (standard for fluids)
    # x0=p tells it to start guessing from the last frame's pressure
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3, max_iterations=1000, x0=p)
    
    v, p = fluid.make_incompressible(v, solve=solver)
    
    return s, v, p

# --- MAIN LOOP ---
dt = 0.2

for i in range(200):
    try:
        # Run the compiled step
        smoke, velocity, pressure = step_function(smoke, velocity, pressure, dt)
        
        # Data Processing
        vel_centered = velocity.at_centers()
        v_hr = vel_centered.values.numpy('y,x,vector')
        
        # Sanity Check for NaN
        if np.isnan(v_hr).any():
            print(f"❌ NaN at frame {i}. Stopping.")
            break

        v_lr = v_hr[::ds_factor, ::ds_factor, :] 

        hr_data.append(v_hr)
        lr_data.append(v_lr)
        
        if i % 10 == 0:
            print(f"Frame {i}/200 generated.")
            
    except Exception as e:
        print(f"⚠️ Error at frame {i}: {e}")
        # If it crashes, we try to continue without updating (or break)
        # Usually JIT compilation prevents random fluctuations
        continue

print("💾 Saving dataset...")
os.makedirs("../data", exist_ok=True)
np.save("../data/train_hr.npy", np.array(hr_data))
np.save("../data/train_lr.npy", np.array(lr_data))

print("✅ Done! Data saved to 'data/' folder.")