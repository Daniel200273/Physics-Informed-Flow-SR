import numpy as np
from scipy.ndimage import zoom


def process_simulation_data(sim_hr_array: np.ndarray, sim_lr_array: np.ndarray, ds_factor: int):
    """
    Slices raw simulation data into (t-1, t, t+1) sequences, bilinearly 
    upscales LR frames, and reshapes them for the ResNet-U-Net input format (N, 3, 2, H, W).

    Args:
        sim_hr_array (np.ndarray): (N, H, W, 2) High-Resolution frames.
        sim_lr_array (np.ndarray): (N, H/ds, W/ds, 2) Low-Resolution frames.
        ds_factor (int): Downsampling factor (e.g., 4).

    Returns:
        tuple: (final_lr_inputs, final_hr_targets)
               final_lr_inputs: List of (3, 2, H, W) tensors
               final_hr_targets: List of (H, W, 2) tensors
    """
    FRAMES_PER_SIM = sim_hr_array.shape[0]
    all_lr_data = []
    all_hr_data = []

    # Iterate from t=1 to t=N-2 to allow for t-1 and t+1 frames
    for t in range(1, FRAMES_PER_SIM - 1):
        # HR Target: The velocity field at time t
        hr_target = sim_hr_array[t]

        # LR Slice: (t-1, t, t+1)
        lr_slice = sim_lr_array[t-1:t+2]  # Shape (3, H/ds, W/ds, 2)

        upscaled_frames = []
        for frame_idx in range(3):
            # Bilinear upscaling (order=1)
            # zoom=(ds_factor, ds_factor, 1) keeps the vector channel (2) untouched
            upscaled_v = zoom(lr_slice[frame_idx],
                              zoom=(ds_factor, ds_factor, 1),
                              order=1)  # Shape (H, W, 2)
            upscaled_frames.append(upscaled_v)

        # Stack the 3 upscaled frames: (3, H, W, 2)
        upscaled_slice = np.stack(upscaled_frames, axis=0)

        # Reshape to final input format: (3, 2, H, W)
        # Transpose moves the 'vector' (component) channel from index 3 to index 1
        input_tensor = np.transpose(upscaled_slice, (0, 3, 1, 2))

        all_lr_data.append(input_tensor)
        all_hr_data.append(hr_target)

    # Return lists of processed tensors for later concatenation
    return all_lr_data, all_hr_data
