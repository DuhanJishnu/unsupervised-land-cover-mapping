"""
=============================================================================
Week 4b — CNN Autoencoder Design & Training (EnMAP)
=============================================================================

Trains the CNN Autoencoder purely unsupervised on the sampled EnMAP patches
stored in `processed_data/enmap_train_patches.npy`.

Optimized for RTX 4070 (8GB VRAM) and i9 (32GB RAM).
    - Uses mixed precision (AMP) for faster training.

Outputs saved to:
    - models/enmap_encoder.pth -> Trained weights

"""

import os
import sys
import time
import importlib.util
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from config import PLOT_RCPARAMS

# Reuse the model architecture from week 4
try:
    spec = importlib.util.spec_from_file_location(
        "cnn_autoencoder",
        os.path.join(BASE_DIR, "04_cnn_autoencoder.py")
    )
    cnn_module = importlib.util.module_from_spec(spec)
    sys.modules["cnn_autoencoder"] = cnn_module
    spec.loader.exec_module(cnn_module)
    HyperspectralAutoencoder = cnn_module.HyperspectralAutoencoder
    HyperspectralPatchDataset = cnn_module.HyperspectralPatchDataset
except (ImportError, AttributeError, FileNotFoundError):
    print("Error: Could not import 04_cnn_autoencoder.py from the project root.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week4b")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update(PLOT_RCPARAMS)

BATCH_SIZE = 1024
NUM_WORKERS = 8       
EPOCHS = 40           
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EMBEDDING_DIM = 64

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def train_enmap_autoencoder():
    print(f"\n{'═'*70}")
    print(f"  Training Autoencoder: EnMAP")
    print(f"{'═'*70}")
    
    patches_file = os.path.join(PROCESSED_DIR, "enmap_train_patches_3.npy")
    if not os.path.exists(patches_file):
        print(f"\n  [!] Error: {patches_file} not found. Run 02b_enmap_preprocessing.py first.")
        return

    # ─── Load Data ──────────────────────────────────────────────────────────
    dataset = HyperspectralPatchDataset(patches_file)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True,     
        drop_last=False
    )
    
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
    model_path = os.path.join(MODELS_DIR, f"enmap_autoencoder.pth")
    encoder_path = os.path.join(MODELS_DIR, f"enmap_encoder.pth")
    
    torch.save(model.state_dict(), model_path)
    
    encoder_state = {k: v for k, v in model.state_dict().items() 
                     if k.startswith('encoder_')}
    torch.save(encoder_state, encoder_path)
    
    print(f"  Saved Full AE to : models/{os.path.basename(model_path)}")
    print(f"  Saved Encoder to : models/{os.path.basename(encoder_path)}")
    
    # ─── Plot Loss Curve ────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), epoch_losses, 'b-', linewidth=2, label='Train MSE')
    plt.title(f"EnMAP — Autoencoder Training Loss", fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    if epoch_losses[0] / epoch_losses[-1] > 10:
        plt.yscale('log')
        
    plot_path = os.path.join(OUTPUT_DIR, f"enmap_training_loss.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved Loss Curve : outputs/week4b/{os.path.basename(plot_path)}")


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 4b — CNN Autoencoder Training (EnMAP)                    ║")
    print("║  Hardware profile: RTX 4070 (8GB) + i9 (32GB RAM)              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    if not torch.cuda.is_available():
        print("\n  [!] WARNING: CUDA is not available. Training will be slow on CPU.")
    else:
        print(f"\n  [✓] GPU Active: {torch.cuda.get_device_name(0)}")
    
    train_enmap_autoencoder()

    print("\n" + "═"*70)
    print("  Week 4b complete. Encoder is ready for embedding extraction.")
    print("═"*70)

if __name__ == "__main__":
    main()
