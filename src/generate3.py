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

# NEW: 20 SIMULATIONS × 200 FRAMES = 4,000 FRAMES TOTAL
NUM_SIMULATIONS = 1
FRAMES_PER_SIM = 200

# --- 20 DIVERSE SIMULATION CONFIGURATIONS ---
SIMULATION_PARAMS = [
    {"name": "bottom_right_swirl", "center": [
        96, 20], "strength": 0.16, "num_flows": 1},
    {"name": "complex_multi", "center": [
        50, 50], "strength": 0.15, "num_flows": 3},
]

# --- JIT COMPILED STEP FUNCTION (unchanged) ---


@math.jit_compile
def step_function(s, v, p, dt, inflow_s, inflow_v):
    # 1. Advection
    s = advect.semi_lagrangian(s, v, dt) + inflow_s
    v = advect.semi_lagrangian(v, v, dt) + inflow_v

    # 2. Buoyancy
    buoyancy_force = s * (0, 0.5) @ v
    v = v + buoyancy_force

    # 3. Pressure Solve (Incompressibility)
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3,
                        max_iterations=1000, x0=p)
    v, p = fluid.make_incompressible(v, solve=solver)

    return s, v, p


# --- MAIN GENERATION LOOP ---
print(f"🚀 Starting 20 Simulations × 200 Frames = 4,000 total frames...")
print(f"Estimated runtime: ~10-12 hours (run overnight)")

all_hr_data = []
all_lr_data = []

for sim_idx, param in enumerate(SIMULATION_PARAMS):
    print(
        f"\n--- Sim {sim_idx+1}/20: {param['name']} ({param['num_flows']} flows) ---")

    # Initialize inflows for this simulation
    combined_smoke_inflow = CenteredGrid(
        0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    combined_vel_inflow = StaggeredGrid(
        0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

    # Generate num_flows emitters
    for flow_idx in range(param['num_flows']):
        # Use fixed centers for reproducibility (or randomize if desired)
        offset_x = (flow_idx - param['num_flows']//2) * 20
        offset_y = (flow_idx - param['num_flows']//2) * 15

        px = param['center'][0] + offset_x
        py = param['center'][1] + offset_y

        # Create emitter
        sphere = Sphere(center=tensor(
            [px, py], channel(vector="x,y")), radius=4)
        mask_smoke = CenteredGrid(
            sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
        mask_vel = StaggeredGrid(
            sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

        combined_smoke_inflow += (mask_smoke * param['strength'])
        # Directed velocity towards center
        dir_vec = tensor([0.1, 0.3], channel(
            vector="x,y"))  # Slight upward bias
        combined_vel_inflow += mask_vel * dir_vec

    # Reset simulation state
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY,
                             x=H, y=W, bounds=BOUNDS)
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    pressure = None
    dt = 0.2

    # Run 200 frames
    sim_hr_frames = []
    for t in range(FRAMES_PER_SIM):
        smoke, velocity, pressure = step_function(
            smoke, velocity, pressure, dt,
            combined_smoke_inflow, combined_vel_inflow
        )

        # Extract velocity data
        v_hr = velocity.at_centers().values.numpy('y,x,vector')
        sim_hr_frames.append(v_hr)

        if t % 50 == 0:
            print(f"  Frame {t}/{FRAMES_PER_SIM}")

    # Stack and downsample
    sim_hr_array = np.array(sim_hr_frames)  # (200, 128, 128, 2)
    sim_lr_array = sim_hr_array[:, ::ds_factor,
                                ::ds_factor, :]  # (200, 32, 32, 2)

    all_hr_data.append(sim_hr_array)
    all_lr_data.append(sim_lr_array)

    print(f"✅ {param['name']}: {FRAMES_PER_SIM} frames saved")

# --- FINAL SAVE ---
print("\n💾 Combining all simulations...")
final_hr = np.concatenate(all_hr_data, axis=0)  # (4000, 128, 128, 2)
final_lr = np.concatenate(all_lr_data, axis=0)  # (4000, 32, 32, 2)

# Save combined datasets
np.save(f"{DATA_DIR}/train_hr.npy", final_hr)      # 80% train
np.save(f"{DATA_DIR}/train_lr.npy", final_lr)
#np.save(f"{DATA_DIR}/val_hr.npy", final_hr[3200:3600])    # 10% val
#np.save(f"{DATA_DIR}/val_lr.npy", final_lr[3200:3600])
#np.save(f"{DATA_DIR}/test_hr.npy", final_hr[3600:])       # 10% test
#np.save(f"{DATA_DIR}/test_lr.npy", final_lr[3600:])

print(f"\n✅ COMPLETE!")
print(f"Total frames generated: {len(final_hr)}")
print(f"Train: {3200} | Val: {400} | Test: {400}")
print(f"Files saved in {DATA_DIR}/")
print("Run visualize.py to check results!")

''' # 1-5: Smoke Plumes (different positions/strengths)
    {"name": "smoke_bottom_left", "center": [
        32, 12], "strength": 0.20, "num_flows": 1},
    {"name": "smoke_bottom_center", "center": [
        64, 12], "strength": 0.22, "num_flows": 1},
    {"name": "smoke_bottom_right", "center": [
        96, 12], "strength": 0.18, "num_flows": 1},
    {"name": "strong_plume", "center": [
        50, 10], "strength": 0.40, "num_flows": 1},
    {"name": "weak_plume", "center": [40, 15],
        "strength": 0.10, "num_flows": 1},

    # 6-10: Vortex & Channel Flows
    {"name": "central_vortex", "center": [
        64, 64], "strength": 0.15, "num_flows": 1},
    {"name": "channel_left", "center": [
        20, 64], "strength": 0.25, "num_flows": 1},
    {"name": "channel_right", "center": [
        108, 64], "strength": 0.25, "num_flows": 1},
    {"name": "top_down_jet", "center": [
        64, 20], "strength": 0.30, "num_flows": 1},
    {"name": "corner_swirl", "center": [
        20, 20], "strength": 0.18, "num_flows": 1},

    # 11-15: Multi-flow collisions (2-3 inflows)
    {"name": "double_plume", "center": [
        32, 12], "strength": 0.15, "num_flows": 2},
    {"name": "triple_collision", "center": [
        64, 64], "strength": 0.12, "num_flows": 3},
    {"name": "side_jets", "center": [20, 40],
        "strength": 0.18, "num_flows": 2},
    {"name": "crossing_flows", "center": [
        40, 40], "strength": 0.20, "num_flows": 2},
    {"name": "opposing_jets", "center": [
        30, 64], "strength": 0.22, "num_flows": 2},

    # 16-20: Advanced variations
    {"name": "wide_sheet", "center": [64, 12],
        "strength": 0.12, "num_flows": 1},
    {"name": "high_buoyancy", "center": [
        50, 50], "strength": 0.25, "num_flows": 1},
    {"name": "pulsing_jet", "center": [
        64, 12], "strength": 0.20, "num_flows": 1},'''
