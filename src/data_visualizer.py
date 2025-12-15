import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def unpack_physics(data_seq):
    """
    Unpacks (Frames, H, W, 4) -> Velocity Mag, Pressure, Smoke
    """
    # 1. Velocity Magnitude (Channels 0, 1)
    u = data_seq[..., 0]
    v = data_seq[..., 1]
    vel_mag = np.sqrt(u**2 + v**2)
    
    # 2. Pressure (Channel 2)
    pressure = data_seq[..., 2]
    
    # 3. Smoke (Channel 3)
    smoke = data_seq[..., 3]
    
    return vel_mag, pressure, smoke

def play_simulation(file_path, save_gif=False, auto_advance=True):
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

    frames = hr_seq.shape[0]
    print(f"  Data found: {frames} frames.")
    
    # Unpack Data
    hr_vel, hr_pres, hr_smoke = unpack_physics(hr_seq)
    lr_vel, lr_pres, lr_smoke = unpack_physics(lr_seq)

    # Determine Global Max for consistent colors
    # Velocity
    v_max = max(np.max(hr_vel), np.max(lr_vel))
    if v_max == 0: v_max = 1.0
    
    # Pressure (Centered around 0 usually, so use symmetric range)
    p_abs = max(np.max(np.abs(hr_pres)), np.max(np.abs(lr_pres)))
    if p_abs == 0: p_abs = 1.0
    
    # Smoke (Usually 0 to 1)
    s_max = max(np.max(hr_smoke), np.max(lr_smoke))
    if s_max == 0: s_max = 1.0

    # --- Setup Plot (2 Rows x 3 Cols) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.canvas.manager.set_window_title(f"Physics Vis: {os.path.basename(file_path)}")
    
    # Row 1: High Res
    ax_hr_v, ax_hr_p, ax_hr_s = axes[0]
    # Row 2: Low Res
    ax_lr_v, ax_lr_p, ax_lr_s = axes[1]

    # --- Initial Images ---
    
    # Velocity (Inferno)
    img_hr_v = ax_hr_v.imshow(hr_vel[0], cmap='inferno', vmin=0, vmax=v_max, origin='lower')
    img_lr_v = ax_lr_v.imshow(lr_vel[0], cmap='inferno', vmin=0, vmax=v_max, origin='lower', interpolation='nearest')
    ax_hr_v.set_title(f"HR Velocity ({hr_seq.shape[1]}x{hr_seq.shape[2]})")
    ax_lr_v.set_title(f"LR Velocity ({lr_seq.shape[1]}x{lr_seq.shape[2]})")

    # Pressure (RdBu - Diverging)
    img_hr_p = ax_hr_p.imshow(hr_pres[0], cmap='RdBu', vmin=-p_abs, vmax=p_abs, origin='lower')
    img_lr_p = ax_lr_p.imshow(lr_pres[0], cmap='RdBu', vmin=-p_abs, vmax=p_abs, origin='lower', interpolation='nearest')
    ax_hr_p.set_title("HR Pressure")
    ax_lr_p.set_title("LR Pressure")

    # Smoke (Magma)
    img_hr_s = ax_hr_s.imshow(hr_smoke[0], cmap='magma', vmin=0, vmax=s_max, origin='lower')
    img_lr_s = ax_lr_s.imshow(lr_smoke[0], cmap='magma', vmin=0, vmax=s_max, origin='lower', interpolation='nearest')
    ax_hr_s.set_title("HR Smoke Density")
    ax_lr_s.set_title("LR Smoke Density")

    # Hide Axes
    for ax in axes.flat:
        ax.axis('off')

    # Frame Text
    time_text = ax_hr_v.text(0.02, 0.95, '', transform=ax_hr_v.transAxes, color='white', 
                             fontsize=12, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5))

    def update(frame):
        # Velocity
        img_hr_v.set_data(hr_vel[frame])
        img_lr_v.set_data(lr_vel[frame])
        
        # Pressure
        img_hr_p.set_data(hr_pres[frame])
        img_lr_p.set_data(lr_pres[frame])
        
        # Smoke
        img_hr_s.set_data(hr_smoke[frame])
        img_lr_s.set_data(lr_smoke[frame])
        
        time_text.set_text(f"Frame: {frame}/{frames}")
        return img_hr_v, img_lr_v, img_hr_p, img_lr_p, img_hr_s, img_lr_s, time_text

    interval_ms = 50
    should_loop = not auto_advance
    
    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=interval_ms, blit=False, repeat=should_loop
    )

    plt.tight_layout()

    if save_gif:
        out_name = file_path.replace('.npz', '_physics.gif')
        print(f"  Saving animation to {out_name}...")
        ani.save(out_name, writer='pillow', fps=20)
        print("  Done.")
        plt.close(fig)
    else:
        if auto_advance:
            duration = (frames * interval_ms / 1000.0) + 1.0
            print(f"  Playing for {duration:.1f} seconds...")
            plt.show(block=False)
            try:
                plt.pause(duration)
            except Exception:
                pass
            plt.close(fig)
        else:
            print("  Displaying... (Close window to continue)")
            plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize 4-channel fluid simulation (Vel, Pres, Smoke).")
    parser.add_argument('path', nargs='?', default='../data', 
                        help="Path to .npz file or directory.")
    parser.add_argument('--save', action='store_true', help="Save .gif instead of showing.")
    parser.add_argument('--manual', action='store_true', help="Disable auto-advance.")
    
    args = parser.parse_args()

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
    auto_advance = not args.manual
    
    for f in files:
        play_simulation(f, save_gif=args.save, auto_advance=auto_advance)

if __name__ == "__main__":
    main()