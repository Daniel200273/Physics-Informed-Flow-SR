import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# --- Custom Modules ---
# Assuming your ResUNet class is in model.py
from model import ResUNet   
# As requested: FluidDataset is in data_processor.py
from data_processor import FluidDataset 
# As requested: Global max calculator is in k_finder.py
from k_finder import calculate_global_max 

# --- Configuration ---
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 50
PHYSICS_WEIGHT = 0.1  # Lambda for physics loss
DATA_DIR = '../data'
STATS_FILE = 'normalization_stats.json'

# --- Physics Loss Class ---
class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()

    def forward(self, output, scaling_factor):
        """
        Calculates the Divergence of the velocity field ( Conservation of Mass).
        output: (Batch, 2, H, W) -> Normalized velocity
        scaling_factor: K used to normalize data
        """
        # 1. Un-scale to physical units (m/s)
        u = output * scaling_factor
        
        # 2. Extract u_x and u_y
        u_x = u[:, 0, :, :]
        u_y = u[:, 1, :, :]

        # 3. Finite Differences (Central Difference or Forward Difference)
        # Here we use simple forward difference for speed/simplicity
        du_dx = u_x[:, :, 1:] - u_x[:, :, :-1]
        dv_dy = u_y[:, 1:, :] - u_y[:, :-1, :]

        # Crop to matching shapes (H-1, W-1)
        du_dx_cut = du_dx[:, :-1, :]
        dv_dy_cut = dv_dy[:, :, :-1]

        # 4. Divergence = du/dx + dv/dy (Should be 0)
        divergence = du_dx_cut + dv_dy_cut

        return torch.mean(divergence**2)

# --- Training Logic ---
def train_model(use_physics=False):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on {device}")
    print(f"   Physics-Informed Mode: {'ON' if use_physics else 'OFF'}")

    # 2. Run Pre-processing (Find K)
    # This ensures the normalization constant K is fresh and accurate for the current data
    print(f"⚙️  Running pre-processing on {DATA_DIR}...")
    calculate_global_max(data_dir=DATA_DIR, output_file=STATS_FILE)

    # 3. Initialize Full Dataset
    # This loads the ordered list of all valid frame slices from all files
    full_dataset = FluidDataset(data_dir=DATA_DIR, stats_file=STATS_FILE, cache_data=True)
    
    # 4. Split and Shuffle
    # We assume independent frames for training (standard for image-to-image tasks)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Random split creates the partition indices
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42) # Seed for reproducibility
    )
    
    print(f"📚 Dataset Split: {train_size} Training | {val_size} Validation")

    # 5. Data Loaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 6. Model & Optimizer
    # Input: 6 channels (3 frames * 2 velocities) -> Output: 2 channels (1 frame * 2 velocities)
    model = ResUNet(in_channels=6, out_channels=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Get scaling factor K for physics loss
    scaling_factor = full_dataset.K
    print(f"📏 Scaling Factor K: {scaling_factor}")

    # 7. Loss Functions
    criterion_mse = nn.MSELoss()
    criterion_phys = PhysicsLoss() if use_physics else None

    # --- Epoch Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_phys = 0.0
        
        for batch_idx, (lr_input, hr_target) in enumerate(train_loader):
            lr_input, hr_target = lr_input.to(device), hr_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(lr_input)
            
            # Base Loss (Data fidelity)
            loss_mse = criterion_mse(output, hr_target)
            loss = loss_mse
            
            phys_val = 0.0
            if use_physics:
                # Physics Loss (Physical validity)
                loss_phys = criterion_phys(output, scaling_factor)
                phys_val = loss_phys.item()
                loss = loss_mse + (PHYSICS_WEIGHT * loss_phys)

            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_phys += phys_val

        # --- Validation ---
        val_mse = validate(model, val_loader, criterion_mse, device)
        
        # Logging
        avg_mse = total_mse / len(train_loader)
        avg_phys = total_phys / len(train_loader)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train MSE: {avg_mse:.6f} | "
              f"Phys Loss: {avg_phys:.6f} | "
              f"Val MSE: {val_mse:.6f}")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            mode = "PINN" if use_physics else "Baseline"
            torch.save(model.state_dict(), f"checkpoints/{mode}_epoch_{epoch+1}.pth")

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for lr_input, hr_target in loader:
            lr_input, hr_target = lr_input.to(device), hr_target.to(device)
            output = model(lr_input)
            loss = criterion(output, hr_target)
            total_loss += loss.item()
    return total_loss / len(loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--physics', action='store_true', help='Enable Physics-Informed Loss')
    args = parser.parse_args()
    
    train_model(use_physics=args.physics)