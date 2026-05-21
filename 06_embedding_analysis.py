"""Embedding-space visualizations for PCA and CNN autoencoder features."""

import os
import sys
import time
import warnings
import importlib.util

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap

from config import (
    INDIAN_PINES_CLASSES,
    IP_COLORS,
    PAVIA_UNIVERSITY_CLASSES,
    PLOT_RCPARAMS,
    PU_COLORS,
)

# Project modules
from preprocessing import (
    load_dataset, PreprocessingConfig, normalize_bands,
    remove_noisy_bands, apply_pca
)

# Dynamically import the Autoencoder model (filename starts with number)
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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week6")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

plt.rcParams.update(PLOT_RCPARAMS)

# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def extract_pca_embeddings(dataset_name):
    """Computes PCA embeddings on the fly."""
    config = PreprocessingConfig()
    data, gt = load_dataset(dataset_name)
    H, W, B = data.shape
    
    norm = normalize_bands(data, method=config.normalization)
    noisy = config.ip_noisy_bands if dataset_name == "indian_pines" else config.pu_noisy_bands
    clean, _ = remove_noisy_bands(norm, noisy)
    
    pca_result = apply_pca(clean, n_components=config.pca_components)
    pca_data = pca_result[0] if isinstance(pca_result, tuple) else pca_result
    return pca_data.reshape(-1, config.pca_components), gt.ravel(), H, W


def extract_cnn_embeddings(prefix):
    """Computes CNN embeddings on the fly using trained encoder."""
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


def generate_comparison_plot(pca_emb, cnn_emb, labels, class_names, colors, title_prefix, filename, method="umap"):
    """
    Projects embeddings to 2D using UMAP or t-SNE and plots side-by-side.
    Uses a random subset to ensure fast and visually uncluttered plots.
    """
    N = len(pca_emb)
    subset_size = 5000
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(N, subset_size, replace=False) if N > subset_size else np.arange(N)
    
    sub_pca = pca_emb[idx]
    sub_cnn = cnn_emb[idx]
    sub_lbl = labels[idx]
    
    print(f"    Running {method.upper()} on PCA... ", end="", flush=True)
    t0 = time.time()
    if method == "umap":
        reducer1 = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_SEED)
        coords_pca = reducer1.fit_transform(sub_pca)
        reducer2 = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=RANDOM_SEED)
        print(f"Done ({time.time()-t0:.1f}s). Running {method.upper()} on CNN... ", end="", flush=True)
        t0 = time.time()
        coords_cnn = reducer2.fit_transform(sub_cnn)
    else:
        reducer1 = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, init='pca')
        coords_pca = reducer1.fit_transform(sub_pca)
        reducer2 = TSNE(n_components=2, perplexity=30, random_state=RANDOM_SEED, init='pca')
        print(f"Done ({time.time()-t0:.1f}s). Running {method.upper()} on CNN... ", end="", flush=True)
        t0 = time.time()
        coords_cnn = reducer2.fit_transform(sub_cnn)
    print(f"Done ({time.time()-t0:.1f}s)")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{title_prefix} — PCA vs CNN Embeddings ({method.upper()})", fontweight='bold', fontsize=14)
    
    for c in range(len(class_names)):
        mask = sub_lbl == c
        if mask.sum() == 0: continue
        
        c_mapped = "lightgray" if c == 0 else colors[c]
        alpha = 0.2 if c == 0 else 0.6
        s = 3 if c == 0 else 8
        label = 'Background' if c == 0 else class_names[c]
        
        ax1.scatter(coords_pca[mask, 0], coords_pca[mask, 1], c=c_mapped, alpha=alpha, s=s, label=label, rasterized=True)
        ax2.scatter(coords_cnn[mask, 0], coords_cnn[mask, 1], c=c_mapped, alpha=alpha, s=s, label=label, rasterized=True)

    ax1.set_title("PCA Baseline (30D) Embeddings")
    ax1.set_xlabel(f"{method.upper()} 1"); ax1.set_ylabel(f"{method.upper()} 2")
    
    ax2.set_title("CNN Autoencoder (64D) Embeddings\n(Note the tighter, more distinct clusters)")
    ax2.set_xlabel(f"{method.upper()} 1"); ax2.set_ylabel(f"{method.upper()} 2")
    
    # Only put legend on ax2 to save space
    ax2.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.7, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()


def process_dataset_viz(dataset_name, dataset_code, prefix, n_classes, class_names, colors):
    print(f"\n{'═'*60}")
    print(f"  {dataset_name} — Embedding Space Analysis")
    print(f"{'═'*60}")
    
    # ─── 1. EXTRACT ──────────────────────────────────────────────────────────
    print("  [1] Fetching Embeddings (PCA & CNN)...")
    tstart = time.time()
    pca_emb, gt_flat, H, W = extract_pca_embeddings(dataset_code)
    cnn_emb = extract_cnn_embeddings(prefix)
    print(f"      PCA: {pca_emb.shape}, CNN: {cnn_emb.shape} in {time.time()-tstart:.1f}s")
    
    # ─── 2. UMAP ─────────────────────────────────────────────────────────────
    print("\n  [2] Generating UMAP Comparison Plot...")
    generate_comparison_plot(
        pca_emb, cnn_emb, gt_flat, class_names, colors, dataset_name,
        f"{prefix}_comparison_umap.png", method="umap"
    )
    
    # ─── 3. t-SNE ────────────────────────────────────────────────────────────
    print("\n  [3] Generating t-SNE Comparison Plot...")
    generate_comparison_plot(
        pca_emb, cnn_emb, gt_flat, class_names, colors, dataset_name,
        f"{prefix}_comparison_tsne.png", method="tsne"
    )
    
    # ─── 4. RAW CNN CLUSTER MAP ──────────────────────────────────────────────
    print("\n  [4] Clustering CNN Embeddings into Map for Week 7...")
    kmeans = KMeans(n_clusters=n_classes, random_state=RANDOM_SEED, n_init=10)
    cl_labels = kmeans.fit_predict(cnn_emb)
    
    raw_map_2d = cl_labels.reshape((H, W))
    map_path = os.path.join(OUTPUT_DIR, f"{prefix}_raw_cnn_map.npy")
    np.save(map_path, raw_map_2d)
    print(f"      Saved raw land cover map to {map_path}")
    
    # Also save a visual render of the raw map for context
    cmap = ListedColormap(colors[:n_classes + 1])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(raw_map_2d, cmap=cmap, interpolation='nearest')
    ax.set_title(f"{dataset_name} — Raw CNN Cluster Map (k={n_classes})", fontweight='bold')
    patches = [mpatches.Patch(color=colors[i + 1], label=f'Cluster {i}') for i in range(n_classes)]
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_raw_cnn_map.png"))
    plt.close()


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 6 — Visualization & Embedding Space Analysis            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Indian Pines
    process_dataset_viz(
        "Indian Pines", "indian_pines", "ip", 16, INDIAN_PINES_CLASSES, IP_COLORS
    )
    
    # Pavia University
    process_dataset_viz(
        "Pavia University", "pavia_university", "pu", 9, PAVIA_UNIVERSITY_CLASSES, PU_COLORS
    )
    
    print(f"\n  ✓ Week 6 Visualizations complete. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
