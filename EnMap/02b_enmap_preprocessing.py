"""
=============================================================================
Week 2b - EnMAP Preprocessing & Patch Sampling
=============================================================================

Scans the `datas/` directory for EnMAP L2A Hyperspectral images (.TIF).
Since extracting all patches would exceed RAM (~1.4M patches per image),
this script randomly samples N center pixels per image, extracts their 
7x7 spatial neighborhoods, and consolidates them into a single .npy
dataset for unsupervised Autoencoder training.

"""

import os
import sys
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BASE_DIR)

from preprocessing import normalize_bands

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(BASE_DIR, "datas")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week2b")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Patch configuration
PATCH_SIZE = 7
SAMPLES_PER_IMAGE = 25000  # 8 images * 25k = 200k total training patches (~8.5 GB RAM)
RANDOM_SEED = 42

def load_enmap_tif(filepath):
    """
    Load an EnMAP SPECTRAL_IMAGE.TIF as (H, W, B) float32 and mask nodata.
    EnMAP L2A background fill is -32768.
    """
    data = tifffile.imread(filepath)

    if data.ndim == 3 and data.shape[0] > 100 and data.shape[1] > 500:
        data = data.transpose(1, 2, 0)

    data = data.astype(np.float32)
    data[data <= -10000] = np.nan
    return data


def extract_sampled_patches(data, patch_size, num_samples, seed=42, valid_mask=None):
    """
    Extract randomly sampled spatial patches with zero-padding.
    data: (H, W, B)
    """
    H, W, B = data.shape
    margin = patch_size // 2
    
    # Zero-pad the spatial dimensions
    padded = np.pad(
        data,
        ((margin, margin), (margin, margin), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    rng = np.random.default_rng(seed)
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=bool)

    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        raise ValueError("No valid center pixels available for patch sampling.")

    sample_size = min(num_samples, len(rows))
    sample_idx = rng.choice(len(rows), size=sample_size, replace=False)
    rows = rows[sample_idx]
    cols = cols[sample_idx]
    
    # Pre-allocate array for PyTorch: (N, Bands, H, W)
    patches = np.zeros((sample_size, B, patch_size, patch_size), dtype=np.float32)
    coords = np.zeros((sample_size, 2), dtype=np.int32)
    
    for i in range(sample_size):
        r, c = rows[i], cols[i]
        # Padded array is shifted by margin, so padded[r:r+ps] corresponds to centering at original r
        patch = padded[r:r+patch_size, c:c+patch_size, :]
        patches[i] = patch.transpose(2, 0, 1)  # (B, 7, 7)
        coords[i] = [r, c]
        
    return patches, coords

def process_enmap_data():
    print("="*70)
    print(" STARTING ENMAP PREPROCESSING & SAMPLING")
    print("="*70)
    
    # Find all SPECTRAL_IMAGE.TIF files
    search_pattern = os.path.join(DATA_DIR, "**", "*-SPECTRAL_IMAGE.TIF")
    tif_files = sorted(glob.glob(search_pattern, recursive=True))
    
    if not tif_files:
        print(f"[!] No EnMAP .TIF files found in {DATA_DIR}")
        return
        
    print(f"Found {len(tif_files)} EnMAP images.")
    print(f"Targeting {SAMPLES_PER_IMAGE:,} patches per image.")
    
    all_patches = []
    
    for i, filepath in enumerate(tif_files):
        img_name = os.path.basename(os.path.dirname(filepath))
        print(f"\n[{i+1}/{len(tif_files)}] Processing {img_name}...")
        
        # Load EnMAP image and mask nodata
        print("  -> Loading TIF...")
        data = load_enmap_tif(filepath)
        
        print(f"  -> Original shape: {data.shape} | dtype: {data.dtype}")

        valid_centers = np.isfinite(data).any(axis=2)
        complete_centers = np.isfinite(data).all(axis=2)
        n_valid = int(valid_centers.sum())
        n_complete = int(complete_centers.sum())
        print(f"  -> Valid center pixels: {n_valid:,} / {data.shape[0] * data.shape[1]:,}")
        print(f"  -> Fully observed centers: {n_complete:,}")
        if n_valid == 0:
            print("  -> Skipping scene: no valid pixels after nodata masking.")
            continue
        
        # Normalization (Z-score)
        print("  -> Normalizing valid pixels (Z-score)...")
        data = normalize_bands(data, method="zscore").astype(np.float32)
        # After z-score, 0 is the per-band mean, so it is a neutral fill value for nodata.
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Random Sampling
        print(f"  -> Sampling up to {SAMPLES_PER_IMAGE:,} patches (7x7) from valid centers...")
        patches, _ = extract_sampled_patches(
            data,
            patch_size=PATCH_SIZE,
            num_samples=SAMPLES_PER_IMAGE,
            seed=RANDOM_SEED+i,
            valid_mask=valid_centers,
        )
        
        all_patches.append(patches)
        print(f"  -> Extracted batch shape: {patches.shape} ({patches.nbytes / 1024**2:.1f} MB)")
        
        # Clear memory
        del data
        
    # Consolidate
    if not all_patches:
        print("\n[!] No EnMAP patches were extracted. Check nodata masking and input scenes.")
        return

    print("\n" + "-"*50)
    print("Consolidating all sampled patches...")
    final_patches = np.concatenate(all_patches, axis=0)
    
    print(f"Final dataset shape: {final_patches.shape}")
    print(f"Total memory size: {final_patches.nbytes / 1024**3:.2f} GB")
    
    # Save to disk
    out_file = os.path.join(PROCESSED_DIR, "enmap_train_patches.npy")
    print(f"Saving to: {out_file}")
    np.save(out_file, final_patches)
    
    print("="*70)
    print(" ENMAP PREPROCESSING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    process_enmap_data()
