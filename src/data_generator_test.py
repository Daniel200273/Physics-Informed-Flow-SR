import os
import argparse
import time
import numpy as np
import warnings
from phi.flow import *

# Suppress internal solver warnings
warnings.filterwarnings("ignore", module="phiml.math._optimize")

# --- Configuration ---
FRAMES = 150
HR_RES = 256
LR_RES = 64
DOMAIN_SIZE = 100
DT_FRAME = 0.8

def get_velocity_vector(magnitude, angle_deg):
    """Calculates (u, v) vector from magnitude and angle in degrees."""
    angle_rad = np.radians(angle_deg)
    u = magnitude * np.cos(angle_rad)
    v = magnitude * np.sin(angle_rad)
    return (u, v)

def step_simulation(v, s, p, emitters, buoyancy_factor, obstacle_geo, dt, frame, injection_cutoff):
    """
    Advances the physics for ONE time-step.
    This function is resolution-agnostic (works for both HR and LR grids).
    """
    
    # 1. Advection
    # Using mac_cormack as requested in your snippet
    s = advect.mac_cormack(s, v, dt)
    v = advect.mac_cormack(v, v, dt)

    # 2. Inflow
    if frame < injection_cutoff:
        for em in emitters:
            # Create masks specifically for THIS grid's resolution
            # em['geo'] is the abstract shape (Sphere), independent of resolution
            
            # Add Smoke
            mask_s = CenteredGrid(em['geo'], s.extrapolation, bounds=s.bounds, resolution=s.resolution)
            s += mask_s * 0.2 * dt
            
            # Add Momentum
            # Resample to v (Staggered) to ensure shape matching
            mask_v = resample(mask_s, to=v)
            v += (mask_v * em['velocity']) * dt

    # 3. Buoyancy
    # Resample s to v to apply force
    buoyancy_force = resample(s, to=v) * buoyancy_factor
    v += buoyancy_force * dt

    # 4. Friction
    v *= 0.995 

    # 5. Obstacles
    solver_obstacles = []
    if obstacle_geo is not None:
        # Visual Masking
        obs_mask = StaggeredGrid(obstacle_geo, v.extrapolation, bounds=v.bounds, resolution=v.resolution)
        v *= (1 - obs_mask)
        solver_obstacles.append(obstacle_geo)

    # 6. Pressure Solve
    v, p = fluid.make_incompressible(
        v, 
        solver_obstacles, 
        Solve('CG', rel_tol=1e-3, abs_tol=1e-3, x0=p, max_iterations=2000)
    )
    
    return v, s, p

def run_simulation(sim_id, n_sims, output_dir):
    
    scenario = np.random.choice(['plume', 'obstacle', 'jet', 'collision'])
    #scenario = np.random.choice(['obstacle', 'obstacle', 'obstacle', 'obstacle'])
    print(f"--- Generating Sim {sim_id+1}/{n_sims} [{scenario}] ---")
    
    # ---------------------------------------------------------
    # 1. Setup Independent Domains
    # ---------------------------------------------------------
    # We define the bounds once, but create two sets of grids.
    # Note: Using extrapolation settings from your provided snippet.
    
    # High Res (256x256)
    v_hr = StaggeredGrid(0, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    s_hr = CenteredGrid(0, extrapolation.ZERO, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    p_hr = CenteredGrid(0, extrapolation.ZERO, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))

    # Low Res (32x32)
    v_lr = StaggeredGrid(0, extrapolation.BOUNDARY, x=LR_RES, y=LR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    s_lr = CenteredGrid(0, extrapolation.ZERO, x=LR_RES, y=LR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    p_lr = CenteredGrid(0, extrapolation.ZERO, x=LR_RES, y=LR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    
    # ---------------------------------------------------------
    # 2. Physics Parameters (Shared)
    # ---------------------------------------------------------
    # We generate random numbers ONCE so both HR and LR get the exact same setup.
    
    min_inj = int(FRAMES * 0.30)
    max_inj = int(FRAMES * 0.50)
    injection_cutoff = np.random.randint(min_inj, max_inj)
    print(f"   > Injection stops at frame {injection_cutoff}")
    
    buoy_y = np.random.uniform(0.2, 0.5) 
    buoy_x = np.random.uniform(-0.1, 0.1) 
    buoyancy_factor = (buoy_x, buoy_y)

    emitters = []
    obstacle_geo = None
    
    if scenario == 'plume':
        geo = Sphere(x=np.random.uniform(30, 70), y=10, radius=6)
        angle = np.random.uniform(50, 130)
        vel = get_velocity_vector(0.5, angle)
        emitters.append({'geo': geo, 'velocity': vel})
        
    elif scenario == 'jet':
        geo = Sphere(x=10, y=np.random.uniform(30, 60), radius=6)
        angle = np.random.uniform(-40, 20)
        vel = get_velocity_vector(1.0, angle)
        buoyancy_factor = (0.0, 0.05)
        emitters.append({'geo': geo, 'velocity': vel})
        
    elif scenario == 'obstacle':
        geo = Sphere(x=np.random.uniform(30, 70), y=10, radius=6)
        vel = (0, 0.5)
        emitters.append({'geo': geo, 'velocity': vel})
        obstacle_geo = Sphere(x=np.random.uniform(40, 60), y=np.random.uniform(40, 60), radius=np.random.uniform(6, 10))

    elif scenario == 'collision':
        geo1 = Sphere(x=15, y=np.random.uniform(40, 60), radius=6)
        angle1 = np.random.uniform(-30, 0)
        vel1 = get_velocity_vector(1.0, angle1)
        emitters.append({'geo': geo1, 'velocity': vel1})

        geo2 = Sphere(x=85, y=np.random.uniform(40, 60), radius=6)
        angle2 = np.random.uniform(180, 210)
        vel2 = get_velocity_vector(1.0, angle2)
        emitters.append({'geo': geo2, 'velocity': vel2})

    hr_frames = []
    lr_frames = []
    
    total_time_hr = 0.0
    total_time_lr = 0.0

    # ---------------------------------------------------------
    # 3. Time Loop
    # ---------------------------------------------------------
    for frame in range(FRAMES):
        current_time = 0.0
        
        while current_time < DT_FRAME:
            
            # --- Synchronized Time Step ---
            # We calculate dt based on the High Res grid (stricter limit)
            # and apply it to both to keep them in sync.
            v_hr_c = v_hr.at_centers()
            max_vel = math.max(math.abs(v_hr_c.values))
            if max_vel == 0: max_vel = 1e-5
            
            # CFL 0.8
            safe_dt = 0.8 / float(max_vel)
            dt = min(safe_dt, DT_FRAME - current_time)
            dt = min(dt, 0.5)
            
            # --- Step High Res ---
            t0 = time.perf_counter()
            v_hr, s_hr, p_hr = step_simulation(
                v_hr, s_hr, p_hr, 
                emitters, buoyancy_factor, obstacle_geo, 
                dt, frame, injection_cutoff
            )
            total_time_hr += (time.perf_counter() - t0)

            # --- Step Low Res ---
            t0 = time.perf_counter()
            v_lr, s_lr, p_lr = step_simulation(
                v_lr, s_lr, p_lr, 
                emitters, buoyancy_factor, obstacle_geo, 
                dt, frame, injection_cutoff
            )
            total_time_lr += (time.perf_counter() - t0)

            current_time += dt

        # -----------------------------------------------------
        # 4. Extraction & Saving
        # -----------------------------------------------------
        
        # Extract HR (256x256)
        v_hr_c = v_hr.at_centers()
        hr_np = v_hr_c.values.numpy('y,x,vector') 
        
        # Extract LR (32x32) - Directly from the LR simulation
        v_lr_c = v_lr.at_centers()
        lr_np = v_lr_c.values.numpy('y,x,vector')

        hr_frames.append(hr_np)
        lr_frames.append(lr_np)
        
        if frame % 50 == 0:
            print(f"  Frame {frame}/{FRAMES}")

    # ---------------------------------------------------------
    # 5. Finalize
    # ---------------------------------------------------------
    print(f"  ⏱️  HR Generation Time: {total_time_hr:.2f}s")
    print(f"  ⏱️  LR Generation Time: {total_time_lr:.2f}s")
    
    hr_stack = np.stack(hr_frames)
    lr_stack = np.stack(lr_frames)
    
    save_path = os.path.join(output_dir, f'sim_{sim_id:02d}_{scenario}.npz')
    np.savez_compressed(save_path, hr=hr_stack, lr=lr_stack)
    print(f"  💾 Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate independent HR/LR fluid simulations.")
    parser.add_argument("--n_sims", type=int, default=1, help="Number of simulations")
    parser.add_argument("--output_dir", type=str, default='../data_test', help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Generating {args.n_sims} simulations in '{args.output_dir}'...")
    
    for i in range(args.n_sims):
        try:
            run_simulation(i, args.n_sims, args.output_dir)
        except Exception as e:
            print(f"  Simulation {i} failed: {e}")
            
