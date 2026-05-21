"""Preprocessing demonstration and diagnostic plots for benchmark datasets."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import INDIAN_PINES_CLASSES, PAVIA_UNIVERSITY_CLASSES, PLOT_RCPARAMS

# Import the preprocessing module
from preprocessing import (
    PreprocessingConfig, load_dataset, preprocess_dataset,
    save_patches
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week2")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

plt.rcParams.update(PLOT_RCPARAMS)


# ══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_normalization_comparison(raw_data, norm_data, gt, class_names, dataset_label, filename):
    """
    Compare spectral profiles before and after normalization (3 random classes).
    """
    n_classes = len(class_names)
    # Pick 3 classes with most samples
    unique, counts = np.unique(gt, return_counts=True)
    mask = unique > 0
    cls_ids = unique[mask]
    cls_counts = counts[mask]
    top3_idx = np.argsort(cls_counts)[-3:]
    top3_classes = cls_ids[top3_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"{dataset_label} — Before vs After Z-Score Normalization", fontweight='bold')
    
    for col, cls_id in enumerate(top3_classes):
        mask_cls = gt == cls_id
        raw_pixels = raw_data[mask_cls]
        norm_pixels = norm_data[mask_cls]
        
        # Before normalization
        mean_raw = raw_pixels.mean(axis=0)
        std_raw = raw_pixels.std(axis=0)
        bands = np.arange(len(mean_raw))
        axes[0, col].plot(bands, mean_raw, color='#e6194b', linewidth=1.2)
        axes[0, col].fill_between(bands, mean_raw - std_raw, mean_raw + std_raw,
                                   color='#e6194b', alpha=0.15)
        axes[0, col].set_title(f"Raw — {class_names[cls_id]}")
        axes[0, col].set_xlabel("Band Index")
        axes[0, col].set_ylabel("Reflectance (DN)")
        axes[0, col].grid(True, alpha=0.3, linestyle='--')
        
        # After normalization
        mean_norm = norm_pixels.mean(axis=0)
        std_norm = norm_pixels.std(axis=0)
        axes[1, col].plot(bands, mean_norm, color='#4363d8', linewidth=1.2)
        axes[1, col].fill_between(bands, mean_norm - std_norm, mean_norm + std_norm,
                                   color='#4363d8', alpha=0.15)
        axes[1, col].set_title(f"Normalized — {class_names[cls_id]}")
        axes[1, col].set_xlabel("Band Index")
        axes[1, col].set_ylabel("Z-Score")
        axes[1, col].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_patch_examples(patches, labels, class_names, dataset_label, filename, n_examples=5):
    """
    Show sample patches: display a grid of patches for different classes.
    Shows the first 3 bands as an RGB composite of the patch.
    """
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels > 0]  # skip background
    
    n_classes_show = min(6, len(unique_labels))
    selected_classes = unique_labels[:n_classes_show]
    
    fig, axes = plt.subplots(n_classes_show, n_examples, 
                              figsize=(n_examples * 2, n_classes_show * 2))
    fig.suptitle(f"{dataset_label} — Sample {patches.shape[2]}×{patches.shape[3]} Patches",
                 fontweight='bold', fontsize=13)
    
    for row_idx, cls_id in enumerate(selected_classes):
        cls_mask = labels == cls_id
        cls_patches = patches[cls_mask]
        
        for col_idx in range(n_examples):
            if col_idx < len(cls_patches):
                # Use first 3 spectral bands as pseudo-RGB
                patch = cls_patches[col_idx]  # (B, ps, ps)
                rgb = patch[:3].transpose(1, 2, 0)  # (ps, ps, 3)
                # Normalize for display
                for ch in range(3):
                    vmin, vmax = rgb[:, :, ch].min(), rgb[:, :, ch].max()
                    if vmax - vmin > 1e-10:
                        rgb[:, :, ch] = (rgb[:, :, ch] - vmin) / (vmax - vmin)
                
                ax = axes[row_idx, col_idx] if n_classes_show > 1 else axes[col_idx]
                ax.imshow(rgb)
                ax.set_xticks([])
                ax.set_yticks([])
                if col_idx == 0:
                    ax.set_ylabel(f"C{cls_id}", fontsize=9, rotation=0, labelpad=20)
            else:
                ax = axes[row_idx, col_idx] if n_classes_show > 1 else axes[col_idx]
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_pca_variance(pca_model, dataset_label, filename):
    """
    Plot PCA scree plot (variance explained per component + cumulative).
    """
    var_ratio = pca_model.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    n = len(var_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{dataset_label} — PCA Variance Analysis", fontweight='bold')
    
    # Individual variance
    ax1.bar(range(1, n + 1), var_ratio * 100, color='#4363d8', alpha=0.7, edgecolor='#333')
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.set_title("Scree Plot")
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Cumulative variance
    ax2.plot(range(1, n + 1), cum_var * 100, 'o-', color='#e6194b', markersize=4, linewidth=1.5)
    ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    ax2.axhline(y=99, color='gray', linestyle=':', alpha=0.5, label='99% threshold')
    
    # Find components needed for 95% and 99%
    n95 = np.argmax(cum_var >= 0.95) + 1
    n99 = np.argmax(cum_var >= 0.99) + 1
    ax2.axvline(x=n95, color='#3cb44b', linestyle='--', alpha=0.5)
    ax2.axvline(x=n99, color='#f58231', linestyle='--', alpha=0.5)
    ax2.annotate(f'{n95} PCs → 95%', xy=(n95, 95), fontsize=8, color='#3cb44b')
    ax2.annotate(f'{n99} PCs → 99%', xy=(n99, 99), fontsize=8, color='#f58231')
    
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_band_removal_comparison(raw_data, clean_data, retained_bands, dataset_label, filename):
    """
    Visualize which bands were removed and show before/after mean spectra.
    """
    B_orig = raw_data.shape[2]
    B_clean = clean_data.shape[2]
    removed = sorted(set(range(B_orig)) - set(retained_bands))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f"{dataset_label} — Noise Band Removal", fontweight='bold')
    
    # Mean spectrum with removed bands highlighted
    mean_spec = raw_data.reshape(-1, B_orig).mean(axis=0)
    ax1.plot(range(B_orig), mean_spec, color='#4363d8', linewidth=1.0, label='All bands')
    if removed:
        for b in removed:
            ax1.axvspan(b - 0.5, b + 0.5, color='red', alpha=0.15)
        ax1.axvspan(removed[0] - 0.5, removed[0] + 0.5, color='red', alpha=0.15, 
                     label=f'{len(removed)} removed bands')
    ax1.set_xlabel("Original Band Index")
    ax1.set_ylabel("Mean Reflectance")
    ax1.set_title(f"Before: {B_orig} bands (red = removed)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # After removal
    mean_clean = clean_data.reshape(-1, B_clean).mean(axis=0)
    ax2.plot(range(B_clean), mean_clean, color='#3cb44b', linewidth=1.0)
    ax2.set_xlabel("Retained Band Index")
    ax2.set_ylabel("Mean Reflectance")
    ax2.set_title(f"After: {B_clean} bands retained")
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def process_one_dataset(name, class_names, prefix, dataset_label):
    """Process a single dataset through the full pipeline."""
    
    print(f"\n{'─' * 60}")
    print(f"  Processing: {dataset_label}")
    print(f"{'─' * 60}")
    
    # Load raw data
    print("  Loading dataset...")
    data, gt = load_dataset(name)
    print(f"  Loaded: {data.shape} | GT: {gt.shape}")
    
    # Run full pipeline
    config = PreprocessingConfig()
    print(f"\n  Running preprocessing pipeline (patch_size={config.patch_size})...")
    result = preprocess_dataset(data, gt, name, config)
    
    # ─── Save patches ────────────────────────────────────────────────────
    print(f"\n  Saving patches to {PROCESSED_DIR}...")
    save_patches(
        result['patches_all'], result['labels_all'], result['coords_all'],
        PROCESSED_DIR, f"{prefix}_all"
    )
    save_patches(
        result['patches_labeled'], result['labels_labeled'], result['coords_labeled'],
        PROCESSED_DIR, f"{prefix}_labeled"
    )
    
    # ─── Print Summary ───────────────────────────────────────────────────
    print(f"\n  {'─' * 50}")
    print(f"  PREPROCESSING SUMMARY — {dataset_label.upper()}")
    print(f"  {'─' * 50}")
    print(f"  Original shape     : {data.shape}")
    print(f"  After normalization: {result['normalized'].shape}")
    print(f"  After band removal : {result['clean'].shape}")
    print(f"  Bands removed      : {data.shape[2] - result['clean'].shape[2]}")
    print(f"  Bands retained     : {result['clean'].shape[2]}")
    print(f"  Patch shape        : {result['patches_all'].shape}")
    print(f"  Total patches      : {result['patches_all'].shape[0]:,}")
    print(f"  Labeled patches    : {result['patches_labeled'].shape[0]:,}")
    print(f"  PCA components     : {config.pca_components}")
    print(f"  PCA variance       : {result['pca_model'].explained_variance_ratio_.sum():.4f}")
    
    mem_patches = result['patches_all'].nbytes / (1024 ** 2)
    print(f"  Patch memory       : {mem_patches:.1f} MB")
    print(f"  {'─' * 50}")
    
    # ─── Generate Visualizations ─────────────────────────────────────────
    print(f"\n  Generating visualizations...")
    
    plot_normalization_comparison(
        data, result['normalized'], gt, class_names,
        dataset_label, f"{prefix}_normalization_comparison.png"
    )
    
    # Band removal (only meaningful for Indian Pines)
    noisy_bands = config.ip_noisy_bands if name == "indian_pines" else config.pu_noisy_bands
    if noisy_bands:
        from preprocessing import normalize_bands, remove_noisy_bands
        norm_temp = normalize_bands(data, method=config.normalization)
        clean_temp, retained = remove_noisy_bands(norm_temp, noisy_bands)
        plot_band_removal_comparison(
            norm_temp, clean_temp, retained,
            dataset_label, f"{prefix}_band_removal.png"
        )
    
    plot_patch_examples(
        result['patches_labeled'], result['labels_labeled'], class_names,
        dataset_label, f"{prefix}_patch_examples.png"
    )
    
    plot_pca_variance(
        result['pca_model'], dataset_label, f"{prefix}_pca_variance.png"
    )
    
    return result


def main():
    
    # ─── Pavia University ────────────────────────────────────────────────
    pu_result = process_one_dataset(
        "pavia_university", PAVIA_UNIVERSITY_CLASSES, "pu", "Pavia University"
    )
    
    # ─── Final Summary ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Week 2 Complete")
    print("═" * 60)
    
    # List saved files
    print(f"\n  Processed data files ({PROCESSED_DIR}):")
    for f in sorted(os.listdir(PROCESSED_DIR)):
        size_mb = os.path.getsize(os.path.join(PROCESSED_DIR, f)) / (1024 ** 2)
        print(f"    • {f} ({size_mb:.1f} MB)")
    
    print(f"\n  Visualization files ({OUTPUT_DIR}):")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    • {f} ({size_kb:.0f} KB)")
    
    print("═" * 60)


if __name__ == "__main__":
    main()
