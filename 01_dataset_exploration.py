"""Dataset exploration and visualization for hyperspectral benchmarks."""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt     
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
from tabulate import tabulate

from config import (
    INDIAN_PINES_CLASSES,
    IP_COLORS,
    PAVIA_UNIVERSITY_CLASSES,
    PLOT_RCPARAMS,
    PU_COLORS,
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "week1")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "mat_files")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update(PLOT_RCPARAMS)


# ══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(name):
    """
    Load a hyperspectral dataset from .mat files.
    
    Parameters
    ----------
    name : str
        Either 'indian_pines' or 'pavia_university'
    
    Returns
    -------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral image cube.
    gt : np.ndarray, shape (H, W)
        Ground truth label map (0 = background/unlabeled).
    """
    if name == "indian_pines":
        data_mat = sio.loadmat(os.path.join(DATA_DIR, "Indian_pines_corrected.mat"))
        gt_mat = sio.loadmat(os.path.join(DATA_DIR, "Indian_pines_gt.mat"))
        # Identify the data key (not starting with __)
        data_key = [k for k in data_mat.keys() if not k.startswith("__")][0]
        gt_key = [k for k in gt_mat.keys() if not k.startswith("__")][0]
        data = data_mat[data_key].astype(np.float64)
        gt = gt_mat[gt_key].astype(np.int32)
    elif name == "pavia_university":
        data_mat = sio.loadmat(os.path.join(DATA_DIR, "PaviaU.mat"))
        gt_mat = sio.loadmat(os.path.join(DATA_DIR, "PaviaU_gt.mat"))
        data_key = [k for k in data_mat.keys() if not k.startswith("__")][0]
        gt_key = [k for k in gt_mat.keys() if not k.startswith("__")][0]
        data = data_mat[data_key].astype(np.float64)
        gt = gt_mat[gt_key].astype(np.int32)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    return data, gt


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Metadata & Statistics
# ══════════════════════════════════════════════════════════════════════════════

def print_dataset_info(data, gt, name, class_names):
    """Print comprehensive metadata about a dataset."""
    H, W, B = data.shape
    num_classes = len(class_names) - 1  # excluding background
    labeled_pixels = np.count_nonzero(gt)
    total_pixels = H * W
    memory_mb = data.nbytes / (1024 ** 2)
    
    print("\n" + "=" * 70)
    print(f"  DATASET: {name.upper().replace('_', ' ')}")
    print("=" * 70)
    print(f"  Spatial dimensions  : {H} × {W} pixels")
    print(f"  Spectral bands      : {B}")
    print(f"  Data type           : {data.dtype}")
    print(f"  Value range         : [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Memory              : {memory_mb:.1f} MB")
    print(f"  Total pixels        : {total_pixels:,}")
    print(f"  Labeled pixels      : {labeled_pixels:,} ({100*labeled_pixels/total_pixels:.1f}%)")
    print(f"  Background pixels   : {total_pixels - labeled_pixels:,} ({100*(total_pixels-labeled_pixels)/total_pixels:.1f}%)")
    print(f"  Number of classes   : {num_classes}")
    print("-" * 70)
    
    # Per-class distribution
    unique, counts = np.unique(gt, return_counts=True)
    table_rows = []
    for cls_id, count in zip(unique, counts):
        if cls_id == 0:
            continue
        pct = 100 * count / labeled_pixels
        table_rows.append([
            cls_id,
            class_names[cls_id],
            f"{count:,}",
            f"{pct:.2f}%"
        ])
    
    print(tabulate(
        table_rows,
        headers=["ID", "Class Name", "Pixels", "% of Labeled"],
        tablefmt="simple_outline"
    ))
    
    # Class imbalance ratio
    class_counts = counts[unique != 0]
    imbalance_ratio = class_counts.max() / class_counts.min()
    print(f"\n  Max/Min class imbalance ratio: {imbalance_ratio:.1f}x")
    print(f"  Largest class : {class_names[unique[np.argmax(counts[unique != 0]) + 1]]} ({class_counts.max():,})")
    print(f"  Smallest class: {class_names[unique[np.argmin(counts[unique != 0]) + 1]]} ({class_counts.min():,})")
    print("=" * 70 + "\n")
    
    return unique, counts


# ══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_false_color_composite(data, bands, title, filename):
    """
    Create and save a false-color RGB composite from selected bands.
    
    Parameters
    ----------
    data : np.ndarray (H, W, B)
    bands : tuple of 3 ints (R, G, B band indices)
    title : str
    filename : str
    """
    rgb = data[:, :, list(bands)].copy()
    # Per-channel stretch to [0, 1] for display
    for i in range(3):
        ch = rgb[:, :, i]
        p2, p98 = np.percentile(ch, [2, 98])
        rgb[:, :, i] = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.tick_params(direction='in')
    
    # Band annotation
    band_text = f"R: Band {bands[0]}  |  G: Band {bands[1]}  |  B: Band {bands[2]}"
    ax.text(0.5, -0.08, band_text, transform=ax.transAxes, ha='center',
            fontsize=9, style='italic', color='gray')
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_ground_truth_map(gt, class_names, colors, title, filename):
    """
    Create and save a ground truth label map with legend.
    """
    n_classes = len(class_names)
    cmap = ListedColormap(colors[:n_classes])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(gt, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation='nearest')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    # Create legend patches (skip background for cleaner legend)
    patches = [
        mpatches.Patch(color=colors[i], label=f"{i}: {class_names[i]}")
        for i in range(1, n_classes)
    ]
    ax.legend(
        handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0, fontsize=8, frameon=True,
        fancybox=True, shadow=True, title="Classes", title_fontsize=9
    )
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_spectral_signatures(data, gt, class_names, colors, title, filename):
    """
    Plot mean ± std spectral signature for each class.
    """
    n_classes = len(class_names)
    bands = np.arange(data.shape[2])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for cls_id in range(1, n_classes):
        mask = gt == cls_id
        if mask.sum() == 0:
            continue
        pixels = data[mask]  # shape: (N_pixels, B)
        mean_sig = pixels.mean(axis=0)
        std_sig = pixels.std(axis=0)
        
        ax.plot(bands, mean_sig, color=colors[cls_id], label=class_names[cls_id],
                linewidth=1.2, alpha=0.9)
        ax.fill_between(bands, mean_sig - std_sig, mean_sig + std_sig,
                        color=colors[cls_id], alpha=0.08)
    
    ax.set_xlabel("Band Index")
    ax.set_ylabel("Reflectance (Digital Number)")
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=7, frameon=True, fancybox=True, ncol=1)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(bands) - 1)
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_class_distribution(gt, class_names, colors, title, filename):
    """
    Plot horizontal bar chart showing per-class pixel count.
    """
    unique, counts = np.unique(gt, return_counts=True)
    
    # Filter out background
    mask = unique != 0
    cls_ids = unique[mask]
    cls_counts = counts[mask]
    
    # Sort by count (ascending for horizontal bar)
    sort_idx = np.argsort(cls_counts)
    cls_ids = cls_ids[sort_idx]
    cls_counts = cls_counts[sort_idx]
    
    labels = [f"{cid}: {class_names[cid]}" for cid in cls_ids]
    bar_colors = [colors[cid] for cid in cls_ids]
    
    fig, ax = plt.subplots(figsize=(10, max(5, len(cls_ids) * 0.45)))
    bars = ax.barh(labels, cls_counts, color=bar_colors, edgecolor='#333', linewidth=0.5)
    
    # Add count annotations
    for bar, count in zip(bars, cls_counts):
        ax.text(bar.get_width() + max(cls_counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,}", va='center', fontsize=8, color='#333')
    
    ax.set_xlabel("Number of Pixels")
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(0, max(cls_counts) * 1.15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_band_correlation(data, title, filename, sample_size=5000):
    """
    Plot inter-band correlation heatmap using a random pixel subset.
    """
    H, W, B = data.shape
    pixels = data.reshape(-1, B)
    
    # Subsample for speed
    if len(pixels) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[idx]
    
    corr = np.corrcoef(pixels.T)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xlabel("Band Index")
    ax.set_ylabel("Band Index")
    ax.set_title(title, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation")
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_combined_overview(data, gt, class_names, colors, bands_rgb, dataset_label, filename):
    """
    Create a 2×2 overview figure combining key visualizations.
    """
    n_classes = len(class_names)
    cmap = ListedColormap(colors[:n_classes])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    fig.suptitle(f"{dataset_label} — Dataset Overview", fontsize=15, fontweight='bold', y=0.98)
    
    # (0,0) False-color composite
    rgb = data[:, :, list(bands_rgb)].copy()
    for i in range(3):
        ch = rgb[:, :, i]
        p2, p98 = np.percentile(ch, [2, 98])
        rgb[:, :, i] = np.clip((ch - p2) / (p98 - p2 + 1e-8), 0, 1)
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"False-Color Composite (B{bands_rgb[0]}, B{bands_rgb[1]}, B{bands_rgb[2]})")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")
    
    # (0,1) Ground truth
    axes[0, 1].imshow(gt, cmap=cmap, vmin=0, vmax=n_classes - 1, interpolation='nearest')
    axes[0, 1].set_title("Ground Truth Map")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")
    
    # (1,0) Spectral signatures
    band_arr = np.arange(data.shape[2])
    for cls_id in range(1, n_classes):
        mask = gt == cls_id
        if mask.sum() == 0:
            continue
        mean_sig = data[mask].mean(axis=0)
        axes[1, 0].plot(band_arr, mean_sig, color=colors[cls_id],
                        label=class_names[cls_id], linewidth=1.0, alpha=0.85)
    axes[1, 0].set_xlabel("Band Index")
    axes[1, 0].set_ylabel("Reflectance (DN)")
    axes[1, 0].set_title("Mean Spectral Signatures")
    axes[1, 0].legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.7)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # (1,1) Class distribution
    unique, counts = np.unique(gt, return_counts=True)
    mask_bg = unique != 0
    cls_ids = unique[mask_bg]
    cls_counts = counts[mask_bg]
    sort_idx = np.argsort(cls_counts)[::-1]
    cls_ids = cls_ids[sort_idx]
    cls_counts = cls_counts[sort_idx]
    x_labels = [f"{cid}" for cid in cls_ids]
    bar_colors = [colors[cid] for cid in cls_ids]
    axes[1, 1].bar(x_labels, cls_counts, color=bar_colors, edgecolor='#333', linewidth=0.5)
    axes[1, 1].set_xlabel("Class ID")
    axes[1, 1].set_ylabel("Pixel Count")
    axes[1, 1].set_title("Class Distribution")
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 1 — Dataset Exploration & Visualization                  ║")
    print("║  Unsupervised Hyperspectral Land Cover Mapping                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # ─── Load Datasets ───────────────────────────────────────────────────
    print("Loading datasets...")
    ip_data, ip_gt = load_dataset("indian_pines")
    pu_data, pu_gt = load_dataset("pavia_university")
    print("  ✓ Indian Pines loaded")
    print("  ✓ Pavia University loaded")
    
    # ─── Dataset Metadata ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  DATASET METADATA & STATISTICS")
    print("─" * 70)
    print_dataset_info(ip_data, ip_gt, "Indian Pines", INDIAN_PINES_CLASSES)
    print_dataset_info(pu_data, pu_gt, "Pavia University", PAVIA_UNIVERSITY_CLASSES)
    
    # ─── Indian Pines Visualizations ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("  GENERATING INDIAN PINES VISUALIZATIONS")
    print("─" * 70)
    
    # False-color: Near-IR, Red, Green → bands 29, 19, 9 (approx)
    plot_false_color_composite(
        ip_data, bands=(29, 19, 9),
        title="Indian Pines — False-Color Composite (NIR–R–G)",
        filename="ip_false_color.png"
    )
    
    plot_ground_truth_map(
        ip_gt, INDIAN_PINES_CLASSES, IP_COLORS,
        title="Indian Pines — Ground Truth Map (16 Classes)",
        filename="ip_ground_truth.png"
    )
    
    plot_spectral_signatures(
        ip_data, ip_gt, INDIAN_PINES_CLASSES, IP_COLORS,
        title="Indian Pines — Mean Spectral Signatures (±1 Std. Dev.)",
        filename="ip_spectral_signatures.png"
    )
    
    plot_class_distribution(
        ip_gt, INDIAN_PINES_CLASSES, IP_COLORS,
        title="Indian Pines — Class Distribution",
        filename="ip_class_distribution.png"
    )
    
    plot_band_correlation(
        ip_data,
        title="Indian Pines — Inter-Band Correlation Matrix",
        filename="ip_band_correlation.png"
    )
    
    plot_combined_overview(
        ip_data, ip_gt, INDIAN_PINES_CLASSES, IP_COLORS,
        bands_rgb=(29, 19, 9), dataset_label="Indian Pines",
        filename="ip_overview.png"
    )
    
    # ─── Pavia University Visualizations ─────────────────────────────────
    print("\n" + "─" * 70)
    print("  GENERATING PAVIA UNIVERSITY VISUALIZATIONS")
    print("─" * 70)
    
    # False-color: bands 56, 33, 12
    plot_false_color_composite(
        pu_data, bands=(56, 33, 12),
        title="Pavia University — False-Color Composite",
        filename="pu_false_color.png"
    )
    
    plot_ground_truth_map(
        pu_gt, PAVIA_UNIVERSITY_CLASSES, PU_COLORS,
        title="Pavia University — Ground Truth Map (9 Classes)",
        filename="pu_ground_truth.png"
    )
    
    plot_spectral_signatures(
        pu_data, pu_gt, PAVIA_UNIVERSITY_CLASSES, PU_COLORS,
        title="Pavia University — Mean Spectral Signatures (±1 Std. Dev.)",
        filename="pu_spectral_signatures.png"
    )
    
    plot_class_distribution(
        pu_gt, PAVIA_UNIVERSITY_CLASSES, PU_COLORS,
        title="Pavia University — Class Distribution",
        filename="pu_class_distribution.png"
    )
    
    plot_band_correlation(
        pu_data,
        title="Pavia University — Inter-Band Correlation Matrix",
        filename="pu_band_correlation.png"
    )
    
    plot_combined_overview(
        pu_data, pu_gt, PAVIA_UNIVERSITY_CLASSES, PU_COLORS,
        bands_rgb=(56, 33, 12), dataset_label="Pavia University",
        filename="pu_overview.png"
    )
    
    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("  Week 1 Complete — All figures saved to:")
    print(f"    {OUTPUT_DIR}")
    
    saved_files = sorted(os.listdir(OUTPUT_DIR))
    print(f"\n  Generated {len(saved_files)} files:")
    for f in saved_files:
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    • {f} ({size_kb:.0f} KB)")
    print("═" * 70)


if __name__ == "__main__":
    main()
