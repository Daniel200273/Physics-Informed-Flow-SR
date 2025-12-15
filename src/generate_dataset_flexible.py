import os
import argparse
import numpy as np
import time
import warnings
from phi.flow import *

# Suppress internal warnings
warnings.filterwarnings("ignore", module="phiml.math._optimize")

# --- Default Config ---
FRAMES = 150
DOMAIN_SIZE = 100
DT_FRAME = 1.0

def get_velocity_vector(magnitude, angle_deg):
    angle_rad = np.radians(angle_deg)
    u = magnitude * np.cos(angle_rad)
    v = magnitude * np.sin(angle_rad)
    return (u, v)

def run_simulation(sim_id, n_sims, output_dir, resolution, scenario_seed):
    """
    Runs a SINGLE simulation at a specific resolution using a fixed seed.
    Saves Velocity (u,v), Pressure (p), and Smoke (s).
    """
    
    # 1. Set Seed for Reproducibility
    np.random.seed(scenario_seed)
    
    # 2. Select Scenario & Parameters
    scenario = np.random.choice(['plume', 'obstacle', 'jet', 'collision'])
    
    # Setup Grid
    v = StaggeredGrid(0, extrapolation.BOUNDARY, x=resolution, y=resolution, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    s = CenteredGrid(0, extrapolation.ZERO, x=resolution, y=resolution, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    p = CenteredGrid(0, extrapolation.ZERO, x=resolution, y=resolution, bounds=Box(x=DOMAIN_SIZE, y=DOMAIN_SIZE))
    
    # Physics Params
    min_inj = int(FRAMES * 0.30)
    max_inj = int(FRAMES * 0.50)
    injection_cutoff = np.random.randint(min_inj, max_inj)
    
    buoy_y = np.random.uniform(0.2, 0.5) 
    buoy_x = np.random.uniform(-0.1, 0.1) 
    buoyancy_factor = (buoy_x, buoy_y)

    emitters = []
    obstacle_geo = None
    
    # --- SCENARIO SETUP ---
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

    frames_data = []
    
    # 3. Main Loop
    print(f"   > Sim {sim_id}: {scenario} @ {resolution}x{resolution}...")
    
    for frame in range(FRAMES):
        current_time = 0.0
        while current_time < DT_FRAME:
            v_centered = v.at_centers()
            v_max = math.max(math.abs(v_centered.values))
            if v_max == 0: v_max = 1e-5
            
            # CFL condition
            cell_size = DOMAIN_SIZE / resolution
            safe_dt = (0.8 * cell_size) / float(v_max)
            dt = min(safe_dt, DT_FRAME - current_time, 0.5)
            
            # --- PHYSICS ---
            s = advect.mac_cormack(s, v, dt)
            v = advect.mac_cormack(v, v, dt)

            if frame < injection_cutoff:
                for em in emitters:
                    mask_s = CenteredGrid(em['geo'], s.extrapolation, bounds=s.bounds, resolution=s.resolution)
                    s += mask_s * 0.2 * dt
                    mask_v = resample(mask_s, to=v)
                    v += (mask_v * em['velocity']) * dt

            v += resample(s, to=v) * buoyancy_factor * dt
            v *= 0.995 

            solver_obstacles = []
            if obstacle_geo is not None:
                obs_mask = StaggeredGrid(obstacle_geo, v.extrapolation, bounds=v.bounds, resolution=v.resolution)
                v *= (1 - obs_mask)
                solver_obstacles.append(obstacle_geo)

            v, p = fluid.make_incompressible(
                v, solver_obstacles, Solve('CG', rel_tol=1e-3, abs_tol=1e-3, x0=p, max_iterations=2000)
            )
            current_time += dt

        # --- DATA EXTRACTION (4 Channels) ---
        # 1. Velocity (Vector) -> (H, W, 2)
        v_np = v.at_centers().values.numpy('y,x,vector') 
        
        # 2. Pressure (Scalar) -> (H, W, 1)
        p_np = p.values.numpy('y,x,vector') 
        if p_np.ndim == 2: p_np = p_np[..., None]
        
        # 3. Smoke (Scalar) -> (H, W, 1)
        s_np = s.values.numpy('y,x,vector')
        if s_np.ndim == 2: s_np = s_np[..., None]

        # Stack: [u, v, p, s]
        frame_data = np.concatenate([v_np, p_np, s_np], axis=-1)
        frames_data.append(frame_data)

    return np.stack(frames_data), scenario

def generate_dataset(n_sims, output_dir, lr_res, generate_downscaled):
    os.makedirs(output_dir, exist_ok=True)
    HR_RES = 256
    
    print(f"🚀 Generating {n_sims} simulations (4 Channels: u,v,p,s).")
    print(f"   Configs: HR={HR_RES}, LR={lr_res}, Downscaled={generate_downscaled}")
    
    for i in range(n_sims):
        seed = int(time.time()) + i
        
        # 1. Generate High Res (Target)
        hr_data, scenario = run_simulation(i, n_sims, output_dir, HR_RES, seed)
        
        # 2. Generate Low Res Input
        if generate_downscaled:
            print(f"   > Downscaling HR to {lr_res}x{lr_res}...")
            f = HR_RES // lr_res
            t, h, w, c = hr_data.shape
            lr_data = hr_data.reshape(t, h//f, f, w//f, f, c).mean(axis=(2, 4))
            mode = "downscaled"
        else:
            lr_data, _ = run_simulation(i, n_sims, output_dir, lr_res, seed)
            mode = "native"
            
        # 3. Save Pair
        filename = f"sim_{i:03d}_{scenario}_{mode}_{lr_res}.npz"
        save_path = os.path.join(output_dir, filename)
        np.savez_compressed(save_path, hr=hr_data, lr=lr_data)
        print(f"✅ Saved {filename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sims", type=int, default=40)
    parser.add_argument("--out", type=str, default="../data_new")
    parser.add_argument("--lr_res", type=int, default=64, help="Resolution of Input (32 or 64)")
    parser.add_argument("--downscale", action="store_true", help="If set, LR is generated by pooling HR.")
    
    args = parser.parse_args()
    
    generate_dataset(args.n_sims, args.out, args.lr_res, args.downscale)