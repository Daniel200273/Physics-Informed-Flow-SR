from phi.torch.flow import *
import numpy as np
import os
import warnings
import random
import torch
import torch.nn.functional as F

# --- 1. CONFIGURATION ---
warnings.filterwarnings('ignore')
math.set_global_precision(32)

H, W = 256, 256     # High Res Target
DS_FACTOR = 8       # 256 / 8 = 32 (Low Res Input)
FRAMES = 200        # Frames per sim
DT = 0.2            # Time step
BOUNDS = Box(x=100, y=100)

DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- 2. PHYSICS ENGINE (Simple & Stable) ---
@math.jit_compile
def step_function(s, v, p, dt, inflow_s, inflow_v):
    # 1. Advection
    s = advect.semi_lagrangian(s, v, dt) + inflow_s
    v = advect.semi_lagrangian(v, v, dt) + inflow_v 

    # 2. Buoyancy
    # I lowered this from 0.5 to 0.1 so horizontal jets fly straight
    # instead of immediately curving up.
    buoyancy_force = s * (0, 0.1) @ v
    v = v + buoyancy_force

    # 3. Pressure Solve
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3, max_iterations=1000, x0=p)
    v, p = fluid.make_incompressible(v, solve=solver)
    
    return s, v, p

# --- 3. DYNAMIC SCENARIO GENERATOR ---

def get_inflow_for_frame(sim_idx, t):
    # Init Empty Grids
    inflow_s = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    inflow_v = StaggeredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    
    # --- GROUP 1: SIMPLE PLUMES (Sim 0-4) ---
    # Classic "Smoke rising from a chimney"
    if 0 <= sim_idx < 5:
        # Vary X position slightly per sim so they aren't identical
        cx = 50 + (sim_idx * 10) 
        sphere = Sphere(center=tensor([cx, 15], channel(vector="x,y")), radius=8)
        
        inflow_s = CenteredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * 0.5
        inflow_v = StaggeredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * (0, 2.5)

    # --- GROUP 2: PERPENDICULAR COLLISIONS (Sim 5-9) ---
    # One jet Right, One jet Up -> Crash in center
    elif 5 <= sim_idx < 10:
        # Jet 1: Left side, firing Right
        s1 = Sphere(center=tensor([20, 50], channel(vector="x,y")), radius=6)
        v1 = StaggeredGrid(s1, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * (3.0, 0)
        
        # Jet 2: Bottom side, firing Up
        s2 = Sphere(center=tensor([50, 20], channel(vector="x,y")), radius=6)
        v2 = StaggeredGrid(s2, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * (0, 3.0)
        
        inflow_s = (CenteredGrid(s1, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) + 
                    CenteredGrid(s2, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)) * 0.4
        inflow_v = v1 + v2

    # --- GROUP 3: VARIABLE SPINNERS (Sim 10-14) ---
    # Spinning nozzle with DIFFERENT SIZES
    elif 10 <= sim_idx < 15:
        # Rotation Logic
        angle = t * 0.15
        vx = np.cos(angle) * 3.0
        vy = np.sin(angle) * 3.0
        
        # DIFFERENT SIZE logic:
        # Sim 10=Small(4), Sim 14=Large(12)
        r = 4 + (sim_idx - 10) * 2 
        
        sphere = Sphere(center=tensor([50, 50], channel(vector="x,y")), radius=r)
        
        inflow_s = CenteredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * 0.5
        inflow_v = StaggeredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * (vx, vy)

    # --- GROUP 4: MOVING SOURCES (Sim 15-19) ---
    # Source moves Left <-> Right
    else:
        # Ping-pong movement
        # Moves from x=20 to x=80
        curr_x = 20 + abs((t % 100) - 50) * 1.2
        
        sphere = Sphere(center=tensor([curr_x, 30], channel(vector="x,y")), radius=6)
        
        inflow_s = CenteredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * 0.5
        inflow_v = StaggeredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS) * (0, 2.0)

    return inflow_s, inflow_v


# --- 4. MAIN LOOP ---
print(f"🚀 Starting Generation (256x256, 200 Frames)")
global_max_val = 0.0

for sim_idx in range(20):
    
    # Label for filename
    if sim_idx < 5: name = "simple_plume"
    elif sim_idx < 10: name = "perp_collision"
    elif sim_idx < 15: name = "spinner"
    else: name = "mover"
    
    filename = f"sim_{sim_idx:02d}_{name}"
    print(f"Processing {filename}...", end="", flush=True)
    
    # Reset
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    smoke    = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    pressure = None
    
    hr_frames = []

    for t in range(FRAMES):
        # Dynamic Inflow
        in_s, in_v = get_inflow_for_frame(sim_idx, t)
        
        # Physics Step
        smoke, velocity, pressure = step_function(smoke, velocity, pressure, DT, in_s, in_v)
        
        # Store Data
        v_np = velocity.at_centers().values.numpy('y,x,vector') 
        hr_frames.append(v_np)

    # --- SAVE ---
    # 1. To Tensor
    hr_tensor = torch.tensor(np.array(hr_frames), dtype=torch.float32)
    
    # 2. Permute (Time, Channels, H, W)
    hr_tensor = hr_tensor.permute(0, 3, 1, 2)
    
    # 3. Stats
    current_max = torch.max(torch.abs(hr_tensor))
    if current_max > global_max_val: global_max_val = current_max
    
    # 4. AvgPool Downsample (256 -> 32)
    lr_tensor = F.avg_pool2d(hr_tensor, kernel_size=DS_FACTOR, stride=DS_FACTOR)
    
    np.save(f"{DATA_DIR}/{filename}_hr.npy", hr_tensor.numpy())
    np.save(f"{DATA_DIR}/{filename}_lr.npy", lr_tensor.numpy())
    
    print(f" OK (Max={current_max:.2f})")

print(f"\n✅ DONE! Global Max Velocity: {global_max_val:.6f}")