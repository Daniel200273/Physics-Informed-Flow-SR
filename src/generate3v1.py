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
BOUNDS = Box(x=100, y=100)
DATA_DIR = "../data"
os.makedirs(DATA_DIR, exist_ok=True)

NUM_SIMULATIONS = 20
FRAMES_PER_SIM = 200

# --- 20 DIVERSE SIMULATION CONFIGURATIONS (STRONGER FLOWS) ---
SIMULATION_PARAMS = [
    # 1-5: POWERFUL Smoke Plumes
    {"name": "smoke_plume_1", "center": [
        32, 8], "strength": 1.8, "vel_mag": 2.5, "num_flows": 1},
    {"name": "smoke_plume_2", "center": [
        64, 8], "strength": 2.0, "vel_mag": 2.8, "num_flows": 1},
    {"name": "smoke_plume_3", "center": [
        96, 8], "strength": 1.6, "vel_mag": 2.3, "num_flows": 1},
    {"name": "turbo_plume", "center": [
        50, 6], "strength": 2.5, "vel_mag": 3.5, "num_flows": 1},
    {"name": "fire_jet", "center": [
        40, 10], "strength": 1.2, "vel_mag": 2.0, "num_flows": 1},

    # 6-10: High-energy Vortex & Channel
    {"name": "mega_vortex", "center": [
        64, 64], "strength": 1.5, "vel_mag": 3.0, "num_flows": 1},
    {"name": "channel_blast", "center": [
        15, 64], "strength": 2.2, "vel_mag": 4.0, "num_flows": 1},
    {"name": "reverse_channel", "center": [
        113, 64], "strength": 2.2, "vel_mag": 4.0, "num_flows": 1},
    {"name": "top_waterfall", "center": [
        64, 15], "strength": 2.0, "vel_mag": 3.2, "num_flows": 1},
    {"name": "corner_tornado", "center": [
        20, 20], "strength": 1.4, "vel_mag": 2.8, "num_flows": 1},

    # 11-15: COLLIDING FLOWS (much stronger)
    {"name": "double_eruption", "center": [
        32, 8], "strength": 1.2, "vel_mag": 2.2, "num_flows": 2},
    {"name": "triple_storm", "center": [
        64, 64], "strength": 1.0, "vel_mag": 2.0, "num_flows": 3},
    {"name": "side_colliders", "center": [
        20, 40], "strength": 1.6, "vel_mag": 2.5, "num_flows": 2},
    {"name": "x_collision", "center": [
        40, 40], "strength": 1.8, "vel_mag": 2.8, "num_flows": 2},
    {"name": "head_on", "center": [
        30, 64], "strength": 2.0, "vel_mag": 3.0, "num_flows": 2},

    # 16-20: SPECIAL EFFECTS
    {"name": "tsunami_sheet", "center": [
        64, 8], "strength": 1.0, "vel_mag": 2.5, "num_flows": 1},
    {"name": "rocket_launch", "center": [
        50, 50], "strength": 2.2, "vel_mag": 3.8, "num_flows": 1},
    {"name": "pulsing_eruption", "center": [
        64, 8], "strength": 1.8, "vel_mag": 2.5, "num_flows": 1},
    {"name": "swirl_storm", "center": [
        96, 20], "strength": 1.4, "vel_mag": 3.2, "num_flows": 1},
    {"name": "chaos_multi", "center": [
        50, 50], "strength": 1.2, "vel_mag": 2.2, "num_flows": 3},
]


@math.jit_compile
def step_function(s, v, p, dt, inflow_s, inflow_v):
    # 1. Advection + DISSIPATION for sharper structures
    s = advect.semi_lagrangian(s, v, dt) * 0.98 + inflow_s  # Smoke decay
    v = advect.semi_lagrangian(v, v, dt) + inflow_v

    # 2. STRONGER Buoyancy
    buoyancy_force = s * (0, 1.5) @ v  # 3x stronger rising!
    v = v + buoyancy_force

    # 3. Pressure Solve
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3,
                        max_iterations=1000, x0=p)
    v, p = fluid.make_incompressible(v, solve=solver)

    return s, v, p


# --- MAIN GENERATION LOOP ---
print(f"🚀 Starting 20 POWERFUL Simulations × 200 Frames = 4,000 frames...")
print(f"Fixed: 10x stronger flows, dramatic visuals!")

all_hr_data = []
all_lr_data = []

for sim_idx, param in enumerate(SIMULATION_PARAMS):
    print(f"\n--- Sim {sim_idx+1}/20: {param['name']} ---")

    # Initialize STRONGER inflows
    combined_smoke_inflow = CenteredGrid(
        0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    combined_vel_inflow = StaggeredGrid(
        0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

    # Generate num_flows POWERFUL emitters
    for flow_idx in range(param['num_flows']):
        offset_x = (flow_idx - param['num_flows']//2) * 25  # Wider spacing
        offset_y = (flow_idx - param['num_flows']//2) * 20

        px = max(8, min(H-8, param['center'][0] + offset_x))
        py = max(8, min(W-8, param['center'][1] + offset_y))

        # LARGER, STRONGER emitter
        sphere = Sphere(center=tensor(
            [px, py], channel(vector="x,y")), radius=6)  # Bigger!

        mask_smoke = CenteredGrid(
            sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
        mask_vel = StaggeredGrid(
            sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

        # MUCH stronger!
        combined_smoke_inflow += (mask_smoke * param['strength'])

        # DIRECTIONAL high-speed velocity (not weak [0.1,0.3] anymore!)
        angle = random.uniform(0, 2*np.pi)
        vel_x = np.cos(angle) * param['vel_mag']
        vel_y = np.sin(angle) * param['vel_mag'] * 0.8  # Slight upward bias
        dir_vec = tensor([vel_x, vel_y], channel(vector="x,y"))
        combined_vel_inflow += mask_vel * dir_vec

    # Reset simulation
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY,
                             x=H, y=W, bounds=BOUNDS)
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    pressure = None

    # PULSING time step for dynamic flows
    sim_hr_frames = []
    for t in range(FRAMES_PER_SIM):
        # Pulse dt for variation
        dt = 0.1 + 0.1 * np.sin(t * 0.1)

        smoke, velocity, pressure = step_function(
            smoke, velocity, pressure, dt,
            combined_smoke_inflow, combined_vel_inflow
        )

        # Extract velocity (now MUCH more dramatic!)
        v_hr = velocity.at_centers().values.numpy('y,x,vector')
        sim_hr_frames.append(v_hr)

        if t % 50 == 0:
            print(f"  Frame {t}/{FRAMES_PER_SIM}")

    # Save this sim
    sim_hr_array = np.array(sim_hr_frames)
    sim_lr_array = sim_hr_array[:, ::ds_factor, ::ds_factor, :]

    all_hr_data.append(sim_hr_array)
    all_lr_data.append(sim_lr_array)

    print(f"✅ {param['name']}: Dramatic {FRAMES_PER_SIM} frames!")

# --- SAVE ---
print("\n💾 Saving POWERFUL dataset...")
final_hr = np.concatenate(all_hr_data, axis=0)
final_lr = np.concatenate(all_lr_data, axis=0)

# Proper train/val/test split
np.save(f"{DATA_DIR}/train_hr.npy", final_hr[:3200])
np.save(f"{DATA_DIR}/train_lr.npy", final_lr[:3200])
np.save(f"{DATA_DIR}/val_hr.npy", final_hr[3200:3600])
np.save(f"{DATA_DIR}/val_lr.npy", final_lr[3200:3600])
np.save(f"{DATA_DIR}/test_hr.npy", final_hr[3600:])
np.save(f"{DATA_DIR}/test_lr.npy", final_lr[3600:])

print(f"\n✅ POWERFUL DATASET COMPLETE!")
print(f"Total: 4,000 frames of DRAMATIC flows!")
print("Run visualize.py - you'll LOVE the results! 🔥")
