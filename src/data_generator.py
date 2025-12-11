import os
import argparse  # Added for flags
import numpy as np
import warnings
from phi.flow import *

warnings.filterwarnings("ignore", module="phiml.math._optimize")

# --- Configuration (Defaults) ---
N_SIMS = 40
FRAMES = 150
HR_RES = 256
LR_RES = 32
SCALE_FACTOR = HR_RES // LR_RES
DOMAIN_SIZE = 100
DT_FRAME = 0.8

# Default output dir (can be overwritten by flag)
OUTPUT_DIR = '../data'

def get_velocity_vector(magnitude, angle_deg):
    """Calculates (u, v) vector from magnitude and angle in degrees."""
    angle_rad = np.radians(angle_deg)
    u = magnitude * np.cos(angle_rad)
    v = magnitude * np.sin(angle_rad)
    return (u, v)

def run_simulation(sim_id, n_sims, output_dir):
    """
    Sets up and runs a single fluid simulation scenario.
    """
    # Update Globals based on flags
    N_SIMS = n_sims
    OUTPUT_DIR = output_dir
    
    # Create the directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    scenario = np.random.choice(['plume', 'obstacle', 'jet', 'collision'])
    print(f"--- Generating Sim {sim_id+1}/{N_SIMS} [{scenario}] ---")
    
    # 1. Domain & Grid Setup (Closed Boundaries)
    v = StaggeredGrid(0, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    s = CenteredGrid(0, extrapolation.ZERO, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    p = CenteredGrid(0, extrapolation.ZERO, x=HR_RES, y=HR_RES, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    
    # 2. Pulse Injection Logic
    min_inj = int(FRAMES * 0.50)
    max_inj = int(FRAMES * 0.70)
    injection_cutoff = np.random.randint(min_inj, max_inj)
    print(f"   > Injection stops at frame {injection_cutoff}")

    # 3. Physics Initialization
    buoy_y = np.random.uniform(0.2, 0.5) 
    buoy_x = np.random.uniform(-0.1, 0.1) 
    buoyancy_factor = (buoy_x, buoy_y)

    # We store emitter configs in a list to iterate over them
    emitters = []
    obstacle_geo = None
    
    if scenario == 'plume':
        geo = Sphere(x=np.random.uniform(35, 65), y=10, radius=6)
        mask = CenteredGrid(geo, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=v.bounds)
        angle = np.random.uniform(50, 130)
        vel = get_velocity_vector(magnitude=0.5, angle_deg=angle)
        emitters.append({'mask': mask, 'velocity': vel})
        
    elif scenario == 'jet':
        geo = Sphere(x=10, y=np.random.uniform(30, 60), radius=6)
        mask = CenteredGrid(geo, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=v.bounds)
        angle = np.random.uniform(-40, 20)
        vel = get_velocity_vector(magnitude=1.0, angle_deg=angle)
        buoyancy_factor = (0.0, 0.05)
        emitters.append({'mask': mask, 'velocity': vel})
        
    elif scenario == 'obstacle':
        geo = Sphere(x=np.random.uniform(30, 70), y=10, radius=np.random.uniform(6, 10))
        mask = CenteredGrid(geo, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=v.bounds)
        vel = (0, 0.5)
        emitters.append({'mask': mask, 'velocity': vel})
        obstacle_geo = Sphere(x=np.random.uniform(40, 60), y=np.random.uniform(40, 60), radius=np.random.uniform(6, 10))

    elif scenario == 'collision':
        # Emitter 1
        geo1 = Sphere(x=15, y=np.random.uniform(40, 60), radius=6)
        mask1 = CenteredGrid(geo1, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=v.bounds)
        angle1 = np.random.uniform(-30, 0)
        vel1 = get_velocity_vector(magnitude=1.0, angle_deg=angle1)
        emitters.append({'mask': mask1, 'velocity': vel1})

        # Emitter 2
        geo2 = Sphere(x=85, y=np.random.uniform(40, 60), radius=6)
        mask2 = CenteredGrid(geo2, extrapolation.BOUNDARY, x=HR_RES, y=HR_RES, bounds=v.bounds)
        angle2 = np.random.uniform(180, 210)
        vel2 = get_velocity_vector(magnitude=1.0, angle_deg=angle2)
        emitters.append({'mask': mask2, 'velocity': vel2})

    hr_frames = []
    lr_frames = []

    # 4. Main Time Loop
    for frame in range(FRAMES):
        current_time = 0.0
        
        while current_time < DT_FRAME:
            
            # Check Maximum Velocity
            v_centered_check = v.at_centers()
            v_max = math.max(math.abs(v_centered_check.values))
            
            if v_max == 0: v_max = 1e-5
            safe_dt = 0.8 / float(v_max)
            dt = min(safe_dt, DT_FRAME - current_time)
            dt = min(dt, 0.5)
            
            # Physics Step A: Advection
            s = advect.mac_cormack(s, v, dt)
            v = advect.mac_cormack(v, v, dt)

            # Physics Step B: Inflow
            if frame < injection_cutoff:
                for em in emitters:
                    # Add Smoke
                    s += em['mask'] * 0.2 * dt
                    
                    # Add Momentum
                    # Fix: Resample the scalar mask to the staggered velocity grid
                    # BEFORE multiplying by the vector. This aligns the shapes.
                    mask_staggered = resample(em['mask'], to=v)
                    
                    # Apply the specific velocity vector for this emitter
                    v += (mask_staggered * em['velocity']) * dt

            # Physics Step C: Buoyancy
            buoyancy_force = resample(s, to=v) * buoyancy_factor
            v += buoyancy_force * dt

            # Physics Step D: Friction
            v *= 0.995 

            # Physics Step F: Pressure Solve
            solver_obstacles = [obstacle_geo] if obstacle_geo is not None else []
            
            v, p = fluid.make_incompressible(
                v, 
                solver_obstacles, 
                Solve('CG', rel_tol=1e-3, abs_tol=1e-3, x0=p, max_iterations=2000)
            )

            current_time += dt

        # Data Extraction
        v_centered = v.at_centers()
        hr_vel = v_centered.values.numpy('y,x,vector')
        
        h, w, c = hr_vel.shape
        f = SCALE_FACTOR
        lr_vel = hr_vel.reshape(h//f, f, w//f, f, c).mean(axis=(1, 3))

        hr_frames.append(hr_vel)
        lr_frames.append(lr_vel)
        
        if frame % 50 == 0:
            print(f"  Frame {frame}/{FRAMES} | Max Vel: {np.max(np.abs(hr_vel)):.2f}")

    # Saving
    hr_stack = np.stack(hr_frames)
    lr_stack = np.stack(lr_frames)
    
    save_path = os.path.join(OUTPUT_DIR, f'sim_{sim_id:02d}_{scenario}.npz')
    np.savez_compressed(save_path, hr=hr_stack, lr=lr_stack)
    print(f"  Saved {save_path}")

if __name__ == "__main__":
    # Define Arguments
    parser = argparse.ArgumentParser(description="Generate fluid simulations.")
    parser.add_argument("--n_sims", type=int, default=40, help="Number of simulations to generate [default: 40]")
    parser.add_argument("--output_dir", type=str, default='../data', help="Output directory for .npz files [default: ../data]")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting generation of {args.n_sims} simulations in '{args.output_dir}'...")
    for i in range(args.n_sims):
        try:
            run_simulation(i, args.n_sims, args.output_dir)
        except Exception as e:
            print(f"  Simulation {i} failed: {e}")