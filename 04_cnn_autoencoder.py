import os
import sys
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast
import torch.nn.functional as F

from config import PLOT_RCPARAMS
# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week4")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update(PLOT_RCPARAMS)

# Training config
BATCH_SIZE = 1024
NUM_WORKERS = 0
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EMBEDDING_DIM = 64
PATCH_SIZE = 7

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch Dataset
# ══════════════════════════════════════════════════════════════════════════════

class HyperspectralPatchDataset(Dataset):
    """
    Dataset for loading pre-extracted hyperspectral patches.
    Patches are pre-loaded into RAM for faster training.
    """
    def __init__(self, patches_file):
        print(f"    Loading {os.path.basename(patches_file)} into RAM...")
        t0 = time.time()
        # Shape: (N, Bands, H, W)
        self.patches = np.load(patches_file)
        # Convert to torch tensor here to save time during __getitem__
        self.patches = torch.tensor(self.patches, dtype=torch.float32)
        print(f"    Loaded {len(self.patches):,} patches in {time.time()-t0:.2f}s "
              f"({self.patches.element_size() * self.patches.nelement() / 1024**2:.1f} MB)")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


# ══════════════════════════════════════════════════════════════════════════════
# CNN Autoencoder Architecture
# ══════════════════════════════════════════════════════════════════════════════

class HyperspectralAutoencoder(nn.Module):
    """
    Spatial-Spectral CNN Autoencoder.
    Compresses a (Bands x H x W) patch into a 64D embedding, then reconstructs it.
    """
    def __init__(self, in_bands, embedding_dim=64, output_activation="sigmoid"):
        super(HyperspectralAutoencoder, self).__init__()
        
        self.in_bands = in_bands
        self.embedding_dim = embedding_dim
        if output_activation not in {"sigmoid", "linear"}:
            raise ValueError("output_activation must be 'sigmoid' or 'linear'")
        self.output_activation = output_activation
        
        # ─── ENCODER ──────────────────────────────────────────────────────────
        # Input: (B, in_bands, H, W)
        self.encoder_conv = nn.Sequential(
        nn.Conv2d(in_bands, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),

        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),

        nn.AdaptiveAvgPool2d((1,1))
    )
        
        # Flatten (B, 256, 1, 1) -> (B, 256) -> (B, embedding_dim)
        self.encoder_fc = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            # Do NOT apply ReLU to the final embedding so we don't zero out negative coords
        )
        
        # ─── DECODER ──────────────────────────────────────────────────────────
        # (B, embedding_dim) -> (B, 256)
        self.decoder_fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True)
        )
        
        # Unflatten (B, 256) -> (B, 256, 1, 1)
        self.decoder_conv = nn.Sequential(
        
            nn.ConvTranspose2d(256, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        
            nn.ConvTranspose2d(64, in_bands, kernel_size=3),
        )

    def encode(self, x):
        """Extract the embedding."""
        x = self.encoder_conv(x)      # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 256)
        x = self.encoder_fc(x)        # (B, 64)
        return x

    def decode(self, z, output_size):

        z = self.decoder_fc(z)
    
        z = z.view(z.size(0), 256, 1, 1)
    
        x_recon = self.decoder_conv(z)
    
        x_recon = F.interpolate(
            x_recon,
            size=output_size,
            mode='bilinear'
        )
        if self.output_activation == "sigmoid":
            x_recon = torch.sigmoid(x_recon)
    
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z, output_size=x.shape[-2:])
        return x_recon, z

# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_autoencoder(dataset_name, prefix, patches_file):
    """Train the autoencoder on the given dataset."""
    
    print(f"\n{'═'*70}")
    print(f"  Training Autoencoder: {dataset_name}")
    print(f"{'═'*70}")
    
    # ─── Load Data ──────────────────────────────────────────────────────────
    dataset = HyperspectralPatchDataset(patches_file)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,     # Speeds up host-to-device transfer
        drop_last=False
    )
    
    # Extract number of bands from first element
    in_bands = dataset[0].shape[0]
    print(f"  Architecture: {in_bands} bands → {EMBEDDING_DIM}D embedding")
    print(f"  Device: {DEVICE.type.upper()} ({torch.cuda.get_device_name() if DEVICE.type=='cuda' else ''})")
    print(f"  Batches per epoch: {len(loader)} (Batch size: {BATCH_SIZE})")
    
    # ─── Model Setup ────────────────────────────────────────────────────────
    model = HyperspectralAutoencoder(in_bands=in_bands, embedding_dim=EMBEDDING_DIM)
    model = model.to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")
    
    # ─── Training ───────────────────────────────────────────────────────────
    print(f"\n  Starting training for {EPOCHS} epochs...\n")
    
    epoch_losses = []
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        # Optional: update learning rate schedule
        # Halve LR at 50% and 75% of epochs
        if epoch == EPOCHS // 2 or epoch == int(EPOCHS * 0.75):
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                print(f"  [LR] Reduced to {param_group['lr']:.1e}")
        
        for batch_idx, patches in enumerate(loader):
            patches = patches.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                reconstructed, _ = model(patches)
                loss = criterion(reconstructed, patches)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * patches.size(0)
            
            # Print progress every ~20% of the epoch
            if batch_idx % max(1, len(loader) // 5) == 0 and batch_idx > 0:
                print(f"    Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx}/{len(loader)}] "
                      f"Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        
        print(f"  → Epoch {epoch:02d} | Avg Loss: {epoch_loss:.6f}")
    
    total_time = time.time() - start_time
    print(f"\n  Training completed in {total_time/60:.1f} minutes.")
    print(f"  Final MSE Loss: {epoch_losses[-1]:.6f}")
    
    # ─── Save Models ────────────────────────────────────────────────────────
    model_path = os.path.join(MODELS_DIR, f"{prefix}_autoencoder.pth")
    encoder_path = os.path.join(MODELS_DIR, f"{prefix}_encoder.pth")
    
    # Save full model
    torch.save(model.state_dict(), model_path)
    
    # Save just the encoder part for easy loading in Week 5
    encoder_state = {k: v for k, v in model.state_dict().items() 
                     if k.startswith('encoder_')}
    torch.save(encoder_state, encoder_path)
    
    print(f"  Saved Full AE to : models/{os.path.basename(model_path)}")
    print(f"  Saved Encoder to : models/{os.path.basename(encoder_path)}")
    
    # ─── Plot Loss Curve ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, 'b-', linewidth=2, label='Train MSE')
    plt.title(f"{dataset_name} — Autoencoder Training Loss", fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Make y-axis logarithmic for better visualization if dropping steeply
    if epoch_losses[0] / epoch_losses[-1] > 10:
        plt.yscale('log')
        
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_training_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved Loss Curve : outputs/week4/{os.path.basename(plot_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():

    if not torch.cuda.is_available():
        print("\n WARNING: CUDA is not available. Training will be slow on CPU.")
    else:
        print(f"\n GPU Active: {torch.cuda.get_device_name(0)}")
    
    for dataset_name, prefix in [("Indian Pines", "ip"), ("Pavia University", "pu")]:
        patches_file = os.path.join(PROCESSED_DIR, f"{prefix}_all_patches.npy")
        if os.path.exists(patches_file):
            train_autoencoder(dataset_name, prefix, patches_file)
        else:
            print(f"\n  [!] Error: {patches_file} not found. Run Week 2 pipeline first.")

    print("\n" + "═"*70)
    print("  Week 4 complete. Encoders are ready for embedding extraction.")
    print("═"*70)


if __name__ == "__main__":
    main()
