from phi.torch.flow import *
import numpy as np
import os
import warnings
import random

# --- SETUP ---
warnings.filterwarnings('ignore')
math.set_global_precision(64) 

H, W = 128, 128
ds_factor = 4 
BOUNDS = Box(x=100, y=100) # Physical size of the box
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
NUM_SIMULATIONS = 5 
FRAMES_PER_SIM = 100 

# --- JIT COMPILED STEP FUNCTION ---
@math.jit_compile
def step_function(s, v, p, dt, inflow_s, inflow_v):
    # 1. Advection
    s = advect.semi_lagrangian(s, v, dt) + inflow_s
    v = advect.semi_lagrangian(v, v, dt) + inflow_v 

    # 2. Buoyancy (Heat rising effect - optional, but adds realism)
    buoyancy_force = s * (0, 0.5) @ v
    v = v + buoyancy_force

    # 3. Pressure Solve (Incompressibility)
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3, max_iterations=1000, x0=p)
    v, p = fluid.make_incompressible(v, solve=solver)
    
    return s, v, p

# --- HELPER: SMART EMITTER ---
def get_random_inward_emitter():
    """
    Generates a source position on the edge and a velocity vector 
    that points towards the center of the grid.
    """
    # 1. Pick a side to spawn on (0: Top, 1: Bottom, 2: Left, 3: Right)
    side = np.random.randint(0, 4)
    margin = 10 # Keep away from the absolute edge to avoid clipping
    
    if side == 0: # Top Edge
        pos_x = np.random.randint(margin, H - margin)
        pos_y = H - margin
    elif side == 1: # Bottom Edge
        pos_x = np.random.randint(margin, H - margin)
        pos_y = margin
    elif side == 2: # Left Edge
        pos_x = margin
        pos_y = np.random.randint(margin, W - margin)
    else: # Right Edge
        pos_x = H - margin
        pos_y = np.random.randint(margin, W - margin)

    # 2. Pick a Target Point near the center (with some noise)
    # Center is roughly (64, 64)
    target_x = (H // 2) + np.random.randint(-20, 20)
    target_y = (W // 2) + np.random.randint(-20, 20)

    # 3. Calculate Vector (Target - Position)
    vec_x = target_x - pos_x
    vec_y = target_y - pos_y
    
    # 4. Normalize Vector (make length 1.0)
    magnitude = np.sqrt(vec_x**2 + vec_y**2)
    if magnitude == 0: magnitude = 1 # Safety
    
    dir_x = vec_x / magnitude
    dir_y = vec_y / magnitude
    
    # 5. Apply Random Speed
    speed = np.random.uniform(3.0, 6.0) # Speed magnitude
    final_vx = dir_x * speed
    final_vy = dir_y * speed
    
    return (pos_x, pos_y), (final_vx, final_vy)


# --- MAIN GENERATION LOOP ---
print(f"🚀 Starting Generation of {NUM_SIMULATIONS} Smart Simulations...")

all_hr_data = []
all_lr_data = []

for sim_idx in range(NUM_SIMULATIONS):
    
    # 1. Randomize Number of Flows (1 to 3)
    num_flows = np.random.randint(1, 4) # Returns 1, 2, or 3
    
    print(f"--- Sim {sim_idx+1}/{NUM_SIMULATIONS}: Generating {num_flows} colliding flows ---")

    # Initialize accumulation grids for this simulation
    combined_smoke_inflow = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    combined_vel_inflow   = StaggeredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

    # 2. Generate and Add Emitters
    for _ in range(num_flows):
        (px, py), (vx, vy) = get_random_inward_emitter()
        
        # Create Geometry
        sphere = Sphere(center=tensor([px, py], channel(vector="x,y")), radius=4)
        
        # Create Smoke Mask
        mask_smoke = CenteredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
        combined_smoke_inflow += (mask_smoke * 0.2) # Add intensity
        
        # Create Velocity Mask
        mask_vel = StaggeredGrid(sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
        # Apply the specific velocity vector for this emitter
        force = mask_vel * tensor([vx, vy], channel(vector="x,y"))
        combined_vel_inflow += force

    # 3. Reset Simulation State
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    smoke    = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    pressure = None
    dt = 0.2
    
    # 4. Run Time Steps
    sim_hr_frames = []
    
    for t in range(FRAMES_PER_SIM):
        smoke, velocity, pressure = step_function(
            smoke, velocity, pressure, dt, 
            combined_smoke_inflow, combined_vel_inflow
        )
        
        # Process Data
        v_hr = velocity.at_centers().values.numpy('y,x,vector') # (128, 128, 2)
        sim_hr_frames.append(v_hr)

    # Stack and Downsample
    sim_hr_array = np.array(sim_hr_frames)
    sim_lr_array = sim_hr_array[:, ::ds_factor, ::ds_factor, :] 
    
    all_hr_data.append(sim_hr_array)
    all_lr_data.append(sim_lr_array)

# --- SAVE ---
final_hr = np.array(all_hr_data)
final_lr = np.array(all_lr_data)

print(f"💾 Saving Dataset...")
np.save(f"{DATA_DIR}/dataset_hr.npy", final_hr)
np.save(f"{DATA_DIR}/dataset_lr.npy", final_lr)
print("✅ Done! Run the visualization script to see the colliding flows.")