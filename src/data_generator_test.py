import os
import argparse
import numpy as np
import warnings
from phi.flow import *

warnings.filterwarnings("ignore", module="phiml.math._optimize")

# --- Configuration ---
# N_SIMS and OUTPUT_DIR set via command line
FRAMES = 150
HR_RES = 256
LR_RES = 32
DOMAIN_SIZE = 100
DT_FRAME = 0.8

def get_velocity_vector(magnitude, angle_deg):
    """Calculates (u, v) vector from magnitude and angle in degrees."""
    angle_rad = np.radians(angle_deg)
    u = magnitude * np.cos(angle_rad)
    v = magnitude * np.sin(angle_rad)
    return (u, v)

def step_physics(v, s, p, emitters, buoyancy_factor, obstacle_geo, dt, frame, injection_cutoff):
    """
    Advances the physics by one time-step 'dt'.
    This function is resolution-agnostic; it works for both HR and LR grids.
    """
    # 1. Advection (using MacCormack as requested)
    s = advect.mac_cormack(s, v, dt)
    v = advect.mac_cormack(v, v, dt)

    # 2. Inflow
    if frame < injection_cutoff:
        for em in emitters:
            # em['geo'] is the Shape object (resolution independent)
            # We create the mask specifically for THIS grid (v or s)
            
            # Add Smoke (Centered Grid)
            mask_s = CenteredGrid(em['geo'], extrapolation.BOUNDARY, bounds=s.bounds, resolution=s.resolution)
            s += mask_s * 0.2 * dt
            
            # Add Momentum (Staggered Grid)
            # We recreate the mask on the staggered grid to ensure perfect shape matching
            mask_v = StaggeredGrid(em['geo'], extrapolation.BOUNDARY, bounds=v.bounds, resolution=v.resolution)
            v += (mask_v * em['velocity']) * dt

    # 3. Buoyancy
    # Resample s to v's location
    buoyancy_force = resample(s, to=v) * buoyancy_factor
    v += buoyancy_force * dt

    # 4. Friction
    v *= 0.995 

    # 5. Obstacles
    solver_obstacles = []
    if obstacle_geo is not None:
        # Visual Masking
        obs_mask = StaggeredGrid(obstacle_geo, extrapolation.ZERO, bounds=v.bounds, resolution=v.resolution)
        v *= (1 - obs_mask)
        # Solver constraint
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
    print(f"--- Generating Sim {sim_id+1}/{n_sims} [{scenario}] ---")
    
    # ---------------------------------------------------------
    # 1. Setup Dual Domains (HR and LR)
    # ---------------------------------------------------------
    # Both share the same physical bounds (0 to 100), but different resolutions.
    
    # High Res (256x256)
    v_hr = StaggeredGrid(0, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    s_hr = CenteredGrid(0, extrapolation.ZERO, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    p_hr = CenteredGrid(0, extrapolation.ZERO, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))

    # Low Res (32x32)
    v_lr = StaggeredGrid(0, extrapolation.BOUNDARY, x=LR_RES, y=LR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    s_lr = CenteredGrid(0, extrapolation.ZERO, x=LR_RES, y=LR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    p_lr = CenteredGrid(0, extrapolation.ZERO, x=LR_RES, y=LR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    
    # ---------------------------------------------------------
    # 2. Shared Physics Parameters
    # ---------------------------------------------------------
    # We define the physics parameters ONCE so they are identical for both.
    
    # Injection Timing
    min_inj = int(FRAMES * 0.50)
    max_inj = int(FRAMES * 0.70)
    injection_cutoff = np.random.randint(min_inj, max_inj)
    
    # Buoyancy
    buoy_y = np.random.uniform(0.2, 0.5) 
    buoy_x = np.random.uniform(-0.1, 0.1) 
    buoyancy_factor = (buoy_x, buoy_y)

    # Emitters Configuration
    # We store the GEOMETRY (Shape) and Velocity Vector. 
    # We do NOT store the Grid Mask yet, because masks depend on resolution.
    emitters = [] # List of dicts: {'geo': Shape, 'velocity': (u,v)}
    obstacle_geo = None
    
    if scenario == 'plume':
        geo = Sphere(x=np.random.uniform(35, 65), y=10, radius=6)
        angle = np.random.uniform(50, 130)
        vel = get_velocity_vector(magnitude=0.5, angle_deg=angle)
        emitters.append({'geo': geo, 'velocity': vel})
        
    elif scenario == 'jet':
        geo = Sphere(x=10, y=np.random.uniform(30, 60), radius=6)
        angle = np.random.uniform(-40, 20)
        vel = get_velocity_vector(magnitude=1.0, angle_deg=angle)
        # Reduced buoyancy for jets
        buoyancy_factor = (0.0, 0.05)
        emitters.append({'geo': geo, 'velocity': vel})
        
    elif scenario == 'obstacle':
        geo = Sphere(x=np.random.uniform(30, 70), y=10, radius=np.random.uniform(6, 10))
        vel = (0, 0.5)
        emitters.append({'geo': geo, 'velocity': vel})
        obstacle_geo = Sphere(x=np.random.uniform(40, 60), y=np.random.uniform(40, 60), radius=np.random.uniform(6, 10))

    elif scenario == 'collision':
        # Emitter 1
        geo1 = Sphere(x=15, y=np.random.uniform(40, 60), radius=6)
        angle1 = np.random.uniform(-30, 0)
        vel1 = get_velocity_vector(magnitude=1.0, angle_deg=angle1)
        emitters.append({'geo': geo1, 'velocity': vel1})

        # Emitter 2
        geo2 = Sphere(x=85, y=np.random.uniform(40, 60), radius=6)
        angle2 = np.random.uniform(180, 210)
        vel2 = get_velocity_vector(magnitude=1.0, angle_deg=angle2)
        emitters.append({'geo': geo2, 'velocity': vel2})

    # Data Containers
    hr_frames = []
    lr_frames = []

    # ---------------------------------------------------------
    # 3. Synchronized Time Loop
    # ---------------------------------------------------------
    for frame in range(FRAMES):
        current_time = 0.0
        
        while current_time < DT_FRAME:
            
            # --- CFL Check (Synchronized) ---
            # We must find a 'dt' that is safe for BOTH resolutions.
            # Usually, High Res requires smaller steps.
            
            # Check HR
            v_hr_center = v_hr.at_centers()
            max_hr = math.max(math.abs(v_hr_center.values))
            if max_hr == 0: max_hr = 1e-5
            dt_hr = 0.8 / float(max_hr)

            # Check LR
            v_lr_center = v_lr.at_centers()
            max_lr = math.max(math.abs(v_lr_center.values))
            if max_lr == 0: max_lr = 1e-5
            # Note: LR cells are bigger, so safe_dt is usually larger.
            # But we must respect the physical speed limit.
            # cell_size_lr = DOMAIN_SIZE / LR_RES
            dt_lr = (0.8 * (DOMAIN_SIZE / LR_RES)) / float(max_lr)
            
            # Take the strictest time step
            dt = min(dt_hr, dt_lr, DT_FRAME - current_time, 0.5)
            
            # --- Step Physics for HIGH RES ---
            v_hr, s_hr, p_hr = step_physics(
                v_hr, s_hr, p_hr, 
                emitters, buoyancy_factor, obstacle_geo, 
                dt, frame, injection_cutoff
            )

            # --- Step Physics for LOW RES ---
            v_lr, s_lr, p_lr = step_physics(
                v_lr, s_lr, p_lr, 
                emitters, buoyancy_factor, obstacle_geo, 
                dt, frame, injection_cutoff
            )

            current_time += dt

        # -----------------------------------------------------
        # 4. Data Extraction
        # -----------------------------------------------------
        
        # High Res Extraction
        v_hr_c = v_hr.at_centers()
        hr_np = v_hr_c.values.numpy('y,x,vector') # (256, 256, 2)
        
        # Low Res Extraction (Directly from the LR simulation!)
        v_lr_c = v_lr.at_centers()
        lr_np = v_lr_c.values.numpy('y,x,vector') # (32, 32, 2)

        hr_frames.append(hr_np)
        lr_frames.append(lr_np)
        
        if frame % 50 == 0:
            print(f"  Frame {frame}/{FRAMES} | Max HR Vel: {np.max(np.abs(hr_np)):.2f}")

    # ---------------------------------------------------------
    # 5. Saving
    # ---------------------------------------------------------
    hr_stack = np.stack(hr_frames)
    lr_stack = np.stack(lr_frames)
    
    save_path = os.path.join(output_dir, f'sim_{sim_id:02d}_{scenario}.npz')
    np.savez_compressed(save_path, hr=hr_stack, lr=lr_stack)
    print(f"  Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate fluid simulations.")
    parser.add_argument("--n_sims", type=int, default=40, help="Number of simulations")
    parser.add_argument("--output_dir", type=str, default='../data', help="Output directory")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Generating {args.n_sims} Independent Dual-Res simulations in '{args.output_dir}'...")
    
    for i in range(args.n_sims):
        try:
            run_simulation(i, args.n_sims, args.output_dir)
        except Exception as e:
            print(f"  Simulation {i} failed: {e}")