import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import glob

# --- CONFIG ---
DATA_DIR = "../data"
FPS = 30
PAUSE_BETWEEN_SIMS = 1.0  # Seconds to pause before next sim starts

def get_speed(velocity_field):
    """
    Input shape: (2, H, W)  <- Channels first
    Output shape: (H, W)
    """
    u = velocity_field[0]
    v = velocity_field[1]
    return np.sqrt(u**2 + v**2)

class SimulationPlayer:
    def __init__(self, hr_files):
        self.hr_files = hr_files
        self.current_sim_idx = 0
        self.current_frame = 0
        
        # Setup Figure once
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.im_lr = None
        self.im_hr = None
        self.title_text = self.fig.suptitle("Initializing...", fontsize=16)
        
        # Load first simulation to initialize plots
        self.load_simulation(0)
        
        # Create Colorbar (shared)
        self.cbar = self.fig.colorbar(self.im_hr, ax=[self.ax1, self.ax2], 
                                      orientation='horizontal', fraction=0.05, pad=0.05)
        self.cbar.set_label('Velocity Magnitude')

    def load_simulation(self, idx):
        if idx >= len(self.hr_files):
            print("All simulations played. Looping back to start.")
            self.current_sim_idx = 0
            idx = 0

        hr_path = self.hr_files[idx]
        lr_path = hr_path.replace("_hr.npy", "_lr.npy")
        
        if not os.path.exists(lr_path):
            print(f"Skipping {hr_path} (Missing LR file)")
            self.current_sim_idx += 1
            self.load_simulation(self.current_sim_idx)
            return

        sim_name = os.path.basename(hr_path).replace("_hr.npy", "")
        print(f"Loading Sim {idx + 1}/{len(self.hr_files)}: {sim_name}")

        # Load Data
        self.hr_data = np.load(hr_path)
        self.lr_data = np.load(lr_path)
        self.total_frames = self.hr_data.shape[0]
        self.current_frame = 0
        self.sim_name = sim_name

        # Calculate Max Speed for Normalization
        flat_mag = np.sqrt(self.hr_data[:, 0]**2 + self.hr_data[:, 1]**2)
        max_speed = np.percentile(flat_mag, 99)

        # Initialize or Update Images
        speed_lr = get_speed(self.lr_data[0])
        speed_hr = get_speed(self.hr_data[0])

        if self.im_lr is None:
            # First run initialization
            self.im_lr = self.ax1.imshow(speed_lr, cmap='inferno', origin='lower', vmin=0, vmax=max_speed)
            self.im_hr = self.ax2.imshow(speed_hr, cmap='inferno', origin='lower', vmin=0, vmax=max_speed)
            self.ax1.set_title(f"Input Low Res ({self.lr_data.shape[2]}x{self.lr_data.shape[3]})")
            self.ax2.set_title(f"Target High Res ({self.hr_data.shape[2]}x{self.hr_data.shape[3]})")
        else:
            # Update existing plot settings for new data limits
            self.im_lr.set_clim(0, max_speed)
            self.im_hr.set_clim(0, max_speed)
            self.im_lr.set_data(speed_lr)
            self.im_hr.set_data(speed_hr)
            
        self.title_text.set_text(f'Simulation: {self.sim_name}')

    def update(self, frame):
        # Check if we reached end of current sim
        if self.current_frame >= self.total_frames:
            # Move to next simulation
            self.current_sim_idx += 1
            self.load_simulation(self.current_sim_idx)
            # Add a small pause visually (optional, effectively frames repeating 0)
            return self.im_lr, self.im_hr, self.title_text

        # Get Data for current frame
        s_lr = get_speed(self.lr_data[self.current_frame])
        s_hr = get_speed(self.hr_data[self.current_frame])
        
        # Update Visuals
        self.im_lr.set_data(s_lr)
        self.im_hr.set_data(s_hr)
        
        self.ax1.set_xlabel(f"Frame {self.current_frame}/{self.total_frames}")
        
        self.current_frame += 1
        return self.im_lr, self.im_hr, self.title_text

if __name__ == "__main__":
    # Get list of files
    hr_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_hr.npy")))
    
    if not hr_files:
        print(f"No data found in {DATA_DIR}. Run data_generator.py first.")
        exit()

    print(f"Found {len(hr_files)} simulations. Starting auto-play...")
    
    # Initialize Player Class
    player = SimulationPlayer(hr_files)
    
    # We use a generator or simple infinite Frames because the player manages the index internally
    ani = animation.FuncAnimation(
        player.fig, 
        player.update, 
        interval=1000/FPS, 
        save_count=200 # Needed only if saving
    )
    
    plt.show()