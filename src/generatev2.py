from phi.torch.flow import *
import numpy as np
import os
import warnings
import random
# NEW: Import the processing function
# NOTE: Ensure data_processor.py is in the same directory!
from data_processor import process_simulation_data
# scipy.ndimage.zoom is now inside data_processor.py

# --- SETUP: Increased Resolution and Frames ---
warnings.filterwarnings('ignore')
math.set_global_precision(64)

H, W = 256, 256             # **HR Resolution (Target Output)**
ds_factor = 4               # LR Resolution is 64x64
BOUNDS = Box(x=100, y=100)
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# 20 SIMULATIONS × 200 FRAMES = 4,000 FRAMES TOTAL
NUM_SIMULATIONS = 20
FRAMES_PER_SIM = 200        # Increased frames per simulation

# --- 20 DIVERSE SIMULATION CONFIGURATIONS (unchanged) ---
SIMULATION_PARAMS = [
    # 1-5: Smoke Plumes
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
    # 11-15: Multi-flow collisions
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
        64, 12], "strength": 0.20, "num_flows": 1},
    {"name": "bottom_right_swirl", "center": [
        96, 20], "strength": 0.16, "num_flows": 1},
    {"name": "complex_multi", "center": [
        50, 50], "strength": 0.15, "num_flows": 3}
]

# --- JIT COMPILED STEP FUNCTION (Buoyancy corrected for stability) ---


@math.jit_compile
def step_function(s, v, p, dt, inflow_s, inflow_v):
    # Convert centered velocity inflow to staggered grid format
    inflow_v_staggered = v.with_values(inflow_v.values)

    # 1. Advection
    s = advect.semi_lagrangian(s, v, dt) + inflow_s
    v = advect.semi_lagrangian(v, v, dt) + inflow_v_staggered

    # 2. Buoyancy (Corrected Boussinesq term: s * g_vector)
    g_vector = math.tensor([0.0, 0.5], channel(vector='x,y'))
    buoyancy_force = s * g_vector
    v = v + buoyancy_force * dt

    # 3. Pressure Solve (Incompressibility)
    solver = math.Solve('CG', abs_tol=1e-3, rel_tol=1e-3,
                        max_iterations=1000, x0=p)
    v, p = fluid.make_incompressible(v, solve=solver)

    return s, v, p


# --- MAIN GENERATION LOOP ---
print("🚀 Starting %d Simulations x %d Frames = %d total frames." %
      (NUM_SIMULATIONS, FRAMES_PER_SIM, NUM_SIMULATIONS * FRAMES_PER_SIM))
print("Target Output Shape: (N, 3, 2, 256, 256) | HR Target: (N, 256, 256, 2)")

all_hr_data = []  # Stores (H, W, 2) targets
all_lr_data = []  # Stores (3, 2, H, W) inputs

for sim_idx, param in enumerate(SIMULATION_PARAMS):
    print("\n--- Sim %d/%d: %s (%d flows) ---" %
          (sim_idx+1, NUM_SIMULATIONS, param['name'], param['num_flows']))

    combined_smoke_inflow = CenteredGrid(
        0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    combined_vel_inflow = CenteredGrid(
        0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

    for flow_idx in range(param['num_flows']):
        offset_x = (flow_idx - param['num_flows']//2) * 20
        offset_y = (flow_idx - param['num_flows']//2) * 15

        px = param['center'][0] + offset_x
        py = param['center'][1] + offset_y

        sphere = Sphere(center=tensor(
            [px, py], channel(vector="x,y")), radius=4)
        mask_smoke = CenteredGrid(
            sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
        mask_vel = CenteredGrid(
            sphere, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)

        combined_smoke_inflow += (mask_smoke * param['strength'])
        # Boosted velocity components for more dynamic flow
        dir_vec = tensor([0.2, 0.5], channel(
            vector="x,y"))
        combined_vel_inflow += mask_vel * dir_vec

    velocity = StaggeredGrid(0, extrapolation.BOUNDARY,
                             x=H, y=W, bounds=BOUNDS)
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=H, y=W, bounds=BOUNDS)
    pressure = None
    dt = 0.1  # Reduced time step for stability

    sim_hr_frames = []
    sim_lr_frames = []
    for t in range(FRAMES_PER_SIM):
        smoke, velocity, pressure = step_function(
            smoke, velocity, pressure, dt,
            combined_smoke_inflow, combined_vel_inflow
        )

        v_hr = velocity.at_centers().values.numpy('y,x,vector')  # (256, 256, 2)
        v_lr = v_hr[::ds_factor, ::ds_factor, :]                # (64, 64, 2)

        sim_hr_frames.append(v_hr)
        sim_lr_frames.append(v_lr)

        if t % 50 == 0:
            print("  Frame %d/%d" % (t, FRAMES_PER_SIM))

    sim_hr_array = np.stack(sim_hr_frames, axis=0)
    sim_lr_array = np.stack(sim_lr_frames, axis=0)

    # --- Call the external processing module ---
    print("  Calling data processor for slicing and upscaling...")
    processed_lr_slices, processed_hr_targets = process_simulation_data(
        sim_hr_array, sim_lr_array, ds_factor
    )

    all_lr_data.extend(processed_lr_slices)
    all_hr_data.extend(processed_hr_targets)

    print("✅ %s: %d datapoints saved" % (param['name'], FRAMES_PER_SIM - 2))

# --- FINAL SAVE ---
print("\n💾 Combining all simulations...")
# final_hr: (Num_Datapoints, 256, 256, 2)
final_hr = np.stack(all_hr_data, axis=0)
# final_lr: (Num_Datapoints, 3, 2, 256, 256)
final_lr = np.stack(all_lr_data, axis=0)

# Calculate split sizes
total_datapoints = len(final_hr)
train_size = int(total_datapoints * 0.8)
val_size = int(total_datapoints * 0.1)
test_size = total_datapoints - train_size - val_size

# Save combined datasets
np.save(f"{DATA_DIR}/train_hr.npy", final_hr[:train_size])
np.save(f"{DATA_DIR}/train_lr.npy", final_lr[:train_size])
np.save(f"{DATA_DIR}/val_hr.npy", final_hr[train_size:train_size + val_size])
np.save(f"{DATA_DIR}/val_lr.npy", final_lr[train_size:train_size + val_size])
np.save(f"{DATA_DIR}/test_hr.npy", final_hr[train_size + val_size:])
np.save(f"{DATA_DIR}/test_lr.npy", final_lr[train_size + val_size:])

print("\n✅ COMPLETE!")
print("Total processed datapoints: %d" % total_datapoints)
print("Input Shape (N, t, c, H, W): %s | Output Shape (N, H, W, c): %s" %
      (final_lr.shape, final_hr.shape))
print("Train: %d | Val: %d | Test: %d" % (train_size, val_size, test_size))
print("Files saved in %s/" % DATA_DIR)
