"""Spatial smoothing and map refinement for raw cluster maps."""

import os
import sys
import time
import warnings
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy.ndimage import generic_filter
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from tabulate import tabulate

import torch
from torch.utils.data import DataLoader

from config import (
    INDIAN_PINES_CLASSES,
    IP_COLORS,
    PAVIA_UNIVERSITY_CLASSES,
    PLOT_RCPARAMS,
    PU_COLORS,
)

# Project modules
from preprocessing import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "cnn_autoencoder",
    os.path.join(BASE_DIR, "04_cnn_autoencoder.py"),
)
cnn_module = importlib.util.module_from_spec(spec)
sys.modules["cnn_autoencoder"] = cnn_module
spec.loader.exec_module(cnn_module)
from cnn_autoencoder import HyperspectralPatchDataset, HyperspectralAutoencoder

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEEK6_DIR = os.path.join(BASE_DIR, "outputs", "week6")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week7")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_SIZE = 5  # 5x5 majority filter

plt.rcParams.update(PLOT_RCPARAMS)

# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def extract_cnn_embeddings(prefix):
    """Computes CNN embeddings on the fly for metric evaluation."""
    patches_file = os.path.join(PROCESSED_DIR, f"{prefix}_all_patches.npy")
    encoder_file = os.path.join(MODELS_DIR, f"{prefix}_encoder.pth")
    
    dataset = HyperspectralPatchDataset(patches_file)
    in_bands = dataset[0].shape[0]
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)
    
    model = HyperspectralAutoencoder(in_bands=in_bands, embedding_dim=64)
    model.load_state_dict(torch.load(encoder_file, map_location=DEVICE), strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        from torch.amp import autocast
        for patches in loader:
            patches = patches.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                emb = model.encode(patches)
            embeddings.append(emb.cpu().numpy())
            
    return np.concatenate(embeddings, axis=0)


def fast_mode(window):
    """Returns the most frequent item in a 1D flattened window using bincount."""
    return np.bincount(window.astype(np.int32)).argmax()


def compute_metrics(emb, cl_labels, gt_labels):
    metrics = {}
    
    # Silhouette
    n = len(emb)
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 10000, replace=False)
        metrics['silhouette'] = silhouette_score(emb[idx], cl_labels[idx])
    else:
        metrics['silhouette'] = silhouette_score(emb, cl_labels)
        
    metrics['dbi'] = davies_bouldin_score(emb, cl_labels)
    
    labeled_mask = gt_labels > 0
    if labeled_mask.sum() > 0:
        metrics['ari'] = adjusted_rand_score(gt_labels[labeled_mask], cl_labels[labeled_mask])
    else:
        metrics['ari'] = float('nan')
        
    return metrics


def process_dataset_smoothing(dataset_name, dataset_code, prefix, n_classes, class_names, colors):
    print(f"\n{'═'*60}")
    print(f"  {dataset_name} — Spatial Smoothing")
    print(f"{'═'*60}")
    
    raw_map_file = os.path.join(WEEK6_DIR, f"{prefix}_raw_cnn_map.npy")
    if not os.path.exists(raw_map_file):
        print(f"  [!] Missing raw map: {raw_map_file}. Run Week 6 first.")
        return None
        
    # ─── Load Data ───────────────────────────────────────────────────────────
    raw_map_2d = np.load(raw_map_file)
    _, gt_img = load_dataset(dataset_code)
    gt_flat = gt_img.ravel()
    
    print("  [1] Fetching Embeddings for metric comparison...")
    t0 = time.time()
    cnn_emb = extract_cnn_embeddings(prefix)
    print(f"      Done in {time.time()-t0:.1f}s")
    
    # Base CNN metrics (before smoothing)
    raw_flat = raw_map_2d.ravel()
    base_metrics = compute_metrics(cnn_emb, raw_flat, gt_flat)
    
    # ─── Apply Majority Filter ───────────────────────────────────────────────
    print(f"\n  [2] Applying {WINDOW_SIZE}x{WINDOW_SIZE} Majority (Mode) Filter...")
    t0 = time.time()
    
    # Pad mode 'reflect' helps keep borders sensible during filtering
    smoothed_map_2d = generic_filter(raw_map_2d, fast_mode, size=WINDOW_SIZE, mode='reflect')
    print(f"      Done in {time.time()-t0:.1f}s")
    
    smoothed_flat = smoothed_map_2d.ravel()
    smooth_metrics = compute_metrics(cnn_emb, smoothed_flat, gt_flat)
    
    # ─── Comparison Output ───────────────────────────────────────────────────
    print("\n  [3] Comparing Metrics (Raw vs Smoothed):")
    table = [
        ["Raw CNN Map", f"{base_metrics['silhouette']:.4f}", f"{base_metrics['dbi']:.4f}", f"{base_metrics['ari']:.4f}"],
        ["Smoothed Map", f"{smooth_metrics['silhouette']:.4f}", f"{smooth_metrics['dbi']:.4f}", f"{smooth_metrics['ari']:.4f}"]
    ]
    print(tabulate(table, headers=["Method", "Silhouette", "DBI", "ARI"], tablefmt="double_outline"))
    
    # Save comparison plot
    print("\n  [4] Generating Before/After Map Plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{dataset_name} — Spatial Map Refinement", fontweight='bold', fontsize=15)
    
    cmap_gt = ListedColormap(colors[:len(class_names)])
    cmap_pred = ListedColormap(colors[:n_classes + 1])
    
    axes[0].imshow(gt_img, cmap=cmap_gt, vmin=0, vmax=len(class_names)-1, interpolation='nearest')
    axes[0].set_title("Ground Truth")
    axes[1].imshow(raw_map_2d, cmap=cmap_pred, interpolation='nearest')
    axes[1].set_title("Raw CNN Cluster Map")
    axes[2].imshow(smoothed_map_2d, cmap=cmap_pred, interpolation='nearest')
    axes[2].set_title(f"Smoothed ({WINDOW_SIZE}x{WINDOW_SIZE} Filter)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_smoothing_comparison.png"))
    plt.close()
    
    # Save smoothed map for the final step
    np.save(os.path.join(OUTPUT_DIR, f"{prefix}_smoothed_cnn_map.npy"), smoothed_map_2d)
    
    return {
        'raw_metrics': base_metrics,
        'smooth_metrics': smooth_metrics
    }


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 7 — Spatial Smoothing & Map Refinement                  ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Indian Pines
    ip_res = process_dataset_smoothing(
        "Indian Pines", "indian_pines", "ip", 16, INDIAN_PINES_CLASSES, IP_COLORS
    )
    if ip_res: results["Indian Pines"] = ip_res
    
    # Pavia University
    pu_res = process_dataset_smoothing(
        "Pavia University", "pavia_university", "pu", 9, PAVIA_UNIVERSITY_CLASSES, PU_COLORS
    )
    if pu_res: results["Pavia University"] = pu_res
    
    if not results: return
    
    # Build final comparative CSV summarizing Week 7
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "smoothing_results.csv")
    headers = ["Dataset", "Stage", "Silhouette", "DBI", "ARI"]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for dname, res in results.items():
            writer.writerow([dname, "Raw CNN Map", res['raw_metrics']['silhouette'], res['raw_metrics']['dbi'], res['raw_metrics']['ari']])
            writer.writerow([dname, "Smoothed CNN Map", res['smooth_metrics']['silhouette'], res['smooth_metrics']['dbi'], res['smooth_metrics']['ari']])
            
    print(f"\n  ✓ Week 7 processing complete. CSV saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
