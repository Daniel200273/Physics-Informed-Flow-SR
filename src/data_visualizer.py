import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def calculate_magnitude(velocity_field):
    """
    Computes the magnitude of the velocity vectors.
    Input shape: (Frames, Height, Width, 2)
    Output shape: (Frames, Height, Width)
    """
    # u_x is index 0, u_y is index 1
    u_x = velocity_field[..., 0]
    u_y = velocity_field[..., 1]
    return np.sqrt(u_x**2 + u_y**2)

def play_simulation(file_path, save_gif=False, auto_advance=True):
    """
    Loads an NPZ file and animates the HR and LR side-by-side.
    """
    print(f"Loading {file_path}...")
    
    try:
        with np.load(file_path) as data:
            if 'hr' not in data or 'lr' not in data:
                print(f"Skipping {file_path}: Keys 'hr' and 'lr' not found.")
                return

            hr_seq = data['hr']
            lr_seq = data['lr']
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Check dimensions
    frames = hr_seq.shape[0]
    print(f"  Data found: {frames} frames.")
    
    # Compute Magnitude
    hr_mag = calculate_magnitude(hr_seq)
    lr_mag = calculate_magnitude(lr_seq)

    # Determine Color Scale
    valid_max = max(np.max(hr_mag), np.max(lr_mag))
    if valid_max == 0: valid_max = 1.0 
    
    vmin = 0
    vmax = valid_max

    # --- Setup Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title(f"Visualizing: {os.path.basename(file_path)}")

    # High Res Plot
    img_hr = ax1.imshow(hr_mag[0], cmap='inferno', vmin=vmin, vmax=vmax, origin='lower')
    ax1.set_title(f"High Res ({hr_seq.shape[1]}x{hr_seq.shape[2]})")
    ax1.axis('off')

    # Low Res Plot
    img_lr = ax2.imshow(lr_mag[0], cmap='inferno', vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')
    ax2.set_title(f"Low Res ({lr_seq.shape[1]}x{lr_seq.shape[2]})")
    ax2.axis('off')

    # Add Frame Counter
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white', 
                         fontsize=12, fontweight='bold')

    def update(frame):
        img_hr.set_data(hr_mag[frame])
        img_lr.set_data(lr_mag[frame])
        time_text.set_text(f"Frame: {frame}/{frames}")
        return img_hr, img_lr, time_text

    # Animation Settings
    interval_ms = 50
    
    # If auto-advancing, don't loop the video (repeat=False) so it ends cleanly
    should_loop = not auto_advance
    
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval_ms, blit=True, repeat=should_loop
    )

    if save_gif:
        out_name = file_path.replace('.npz', '.gif')
        print(f"  Saving animation to {out_name}...")
        ani.save(out_name, writer='pillow', fps=20)
        print("  Done.")
        plt.close(fig)
    else:
        if auto_advance:
            # Calculate duration in seconds + a small buffer (1.0s)
            duration = (frames * interval_ms / 1000.0) + 1.0
            print(f"  Playing for {duration:.1f} seconds...")
            
            # Non-blocking show
            plt.show(block=False)
            
            # Pause allows the GUI event loop to run for 'duration' seconds
            try:
                plt.pause(duration)
            except Exception:
                pass # Handle case where user closes window manually during playback
            
            # Close the window to allow the main loop to proceed
            plt.close(fig)
        else:
            print("  Displaying... (Close window to continue to next file)")
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize fluid simulation .npz files.")
    parser.add_argument('path', nargs='?', default='../data', 
                        help="Path to a directory containing .npz files or a specific .npz file.")
    parser.add_argument('--save', action='store_true', 
                        help="Save the animation as a .gif file instead of showing it.")
    parser.add_argument('--manual', action='store_true',
                        help="Disable auto-advance. Wait for user to close window before playing next file.")
    
    args = parser.parse_args()

    # Determine files to process
    files = []
    if os.path.isfile(args.path):
        files = [args.path]
    elif os.path.isdir(args.path):
        files = sorted(glob.glob(os.path.join(args.path, "*.npz")))
    else:
        files = sorted(glob.glob(args.path))

    if not files:
        print(f"No .npz files found in {args.path}")
        return

    print(f"Found {len(files)} files.")
    
    # Default is auto-advance unless --manual is passed
    auto_advance = not args.manual
    
    for f in files:
        play_simulation(f, save_gif=args.save, auto_advance=auto_advance)

if __name__ == "__main__":
    main()