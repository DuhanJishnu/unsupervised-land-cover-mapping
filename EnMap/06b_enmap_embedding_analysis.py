"""
=============================================================================
Week 6b — Visualization & Embedding Space Analysis (EnMAP)
=============================================================================

Visually demonstrates the embedding spaces using patches from EnMAP dataset.
Since EnMAP does not have ground-truth labels in this pipeline, we use 
KMeans (k=8) cluster labels on the CNN embeddings as the pseudo ground-truth
to color the points in UMAP and t-SNE. This shows how PCA scattered the points
versus how tightly the CNN Autoencoder grouped them.

Also saves a "Raw Land Cover Map". Because enmap_train_patches_3.npy contains 
25,000 sampled patches without original spatial coordinates, we arrange the 
resulting 25,000 labels into a 100 x 250 grid to visualize the cluster distribution.

Outputs saved to:
    - outputs/week6b/ -> UMAP/t-SNE comparison figures
    - outputs/week6b/ -> Raw cluster maps (.npy & .png) for Week 7

"""

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import PLOT_RCPARAMS, setup_stdout

setup_stdout()

PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week6b")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

plt.rcParams.update(PLOT_RCPARAMS)

# Dynamically import the Autoencoder model
spec = importlib.util.spec_from_file_location("cnn_autoencoder", os.path.join(BASE_DIR, "04_cnn_autoencoder.py"))
cnn_module = importlib.util.module_from_spec(spec)
sys.modules["cnn_autoencoder"] = cnn_module
spec.loader.exec_module(cnn_module)
from cnn_autoencoder import HyperspectralPatchDataset, HyperspectralAutoencoder

# 8 thematic clusters for EnMAP
ENMAP_COLORS = [
    "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
    "#e6194b", "#911eb4", "#42d4f4", "#bfef45",
]
ENMAP_CLASSES = [
    "Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3",
    "Cluster 4", "Cluster 5", "Cluster 6", "Cluster 7"
]

# ══════════════════════════════════════════════════════════════════════════════
# Core Functions
# ══════════════════════════════════════════════════════════════════════════════

def extract_pca_embeddings(patches_file, n_components=30):
    """
    Computes PCA embeddings on the fly for EnMAP patches.
    We take the center pixel of each patch, scale it, and apply PCA.
    """
    patches = np.load(patches_file)  # (N, Bands, 7, 7)
    
    # Extract center pixel (index 3 for 7x7 patch)
    center_pixels = patches[:, :, 3, 3]  # (N, Bands)
    
    print("      Scaling center pixels for PCA...")
    scaler = StandardScaler()
    scaled_pixels = scaler.fit_transform(center_pixels)
    
    print(f"      Running PCA ({n_components} components)...")
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca_result = pca.fit_transform(scaled_pixels)
    
    return pca_result

def extract_cnn_embeddings(patches_file, encoder_file):
    """Computes CNN embeddings on the fly using trained encoder."""
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
    idx = rng.choice(N, min(subset_size, N), replace=False)
    
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
        
        args = {'alpha': 0.6, 's': 8, 'rasterized': True}
        c_mapped = "lightgray" if c == -1 else colors[c % len(colors)]
        alpha = 0.2 if c == -1 else 0.6
        s = 3 if c == -1 else 8
        label = 'Noise' if c == -1 else class_names[c]
        
        ax1.scatter(coords_pca[mask, 0], coords_pca[mask, 1], c=c_mapped, alpha=alpha, s=s, label=label, rasterized=True)
        ax2.scatter(coords_cnn[mask, 0], coords_cnn[mask, 1], c=c_mapped, alpha=alpha, s=s, label=label, rasterized=True)

    ax1.set_title("PCA Baseline (30D) Embeddings\n(Colored by CNN KMeans Clusters)")
    ax1.set_xlabel(f"{method.upper()} 1"); ax1.set_ylabel(f"{method.upper()} 2")
    
    ax2.set_title("CNN Autoencoder (64D) Embeddings\n(Note the tighter, more distinct clusters)")
    ax2.set_xlabel(f"{method.upper()} 1"); ax2.set_ylabel(f"{method.upper()} 2")
    
    # Only put legend on ax2 to save space
    ax2.legend(fontsize=8, loc='upper right', ncol=2, framealpha=0.7, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def process_enmap_viz():
    print(f"\n{'═'*60}")
    print(f"  EnMAP — Embedding Space Analysis")
    print(f"{'═'*60}")
    
    patches_file = os.path.join(PROCESSED_DIR, "enmap_train_patches_3.npy")
    encoder_file = os.path.join(MODELS_DIR, "enmap_encoder.pth")
    
    if not os.path.exists(patches_file):
        print(f"  [!] Missing patches file: {patches_file}")
        return
        
    if not os.path.exists(encoder_file):
        print(f"  [!] Missing encoder model: {encoder_file}")
        return
    
    # ─── 1. EXTRACT ──────────────────────────────────────────────────────────
    print("  [1] Fetching Embeddings (PCA & CNN)...")
    tstart = time.time()
    pca_emb = extract_pca_embeddings(patches_file)
    cnn_emb = extract_cnn_embeddings(patches_file, encoder_file)
    print(f"      PCA: {pca_emb.shape}, CNN: {cnn_emb.shape} in {time.time()-tstart:.1f}s")
    
    # ─── 2. KMEANS FOR PSEUDO-LABELS ─────────────────────────────────────────
    print("\n  [2] Clustering CNN Embeddings (KMeans k=8) for pseudo-labels...")
    kmeans = KMeans(n_clusters=8, random_state=RANDOM_SEED, n_init=10)
    cl_labels = kmeans.fit_predict(cnn_emb)
    
    # ─── 3. UMAP ─────────────────────────────────────────────────────────────
    print("\n  [3] Generating UMAP Comparison Plot...")
    generate_comparison_plot(
        pca_emb, cnn_emb, cl_labels, ENMAP_CLASSES, ENMAP_COLORS, "EnMAP Dataset",
        "enmap_comparison_umap.png", method="umap"
    )
    
    # ─── 4. t-SNE ────────────────────────────────────────────────────────────
    print("\n  [4] Generating t-SNE Comparison Plot...")
    generate_comparison_plot(
        pca_emb, cnn_emb, cl_labels, ENMAP_CLASSES, ENMAP_COLORS, "EnMAP Dataset",
        "enmap_comparison_tsne.png", method="tsne"
    )
    
    # ─── 5. RAW CNN CLUSTER MAP ──────────────────────────────────────────────
    print("\n  [5] Creating 'Raw' EnMAP Cluster Map...")
    N = cl_labels.shape[0]
    
    # We attempt to factorize N to make a reasonably shaped 2D image
    # For exactly 25,000, 100 x 250 works well
    # For other sizes, we will pad to a perfect square
    if N == 25000:
        H, W = 100, 250
        raw_map_2d = cl_labels.reshape((H, W))
        print(f"      Arranged {N} points into ({H}x{W}) synthetic mosaic.")
    else:
        side = int(np.ceil(np.sqrt(N)))
        padded_labels = np.full(side * side, -1, dtype=int) # -1 code for 'empty'
        padded_labels[:N] = cl_labels
        raw_map_2d = padded_labels.reshape((side, side))
        print(f"      Arranged {N} points into ({side}x{side}) padded synthetic mosaic.")
        
    map_path = os.path.join(OUTPUT_DIR, "enmap_raw_cnn_map.npy")
    np.save(map_path, raw_map_2d)
    print(f"      Saved raw land cover map to {map_path}")
    
    # Visual Render
    cmap = ListedColormap(["lightgray"] + ENMAP_COLORS)
    # Re-normalize data so -1 goes to lightgray, 0..7 go to defined colors
    mapped_data = raw_map_2d + 1 
    
    fig, ax = plt.subplots(figsize=(10, 4) if N == 25000 else (8, 8))
    ax.imshow(mapped_data, cmap=cmap, interpolation='nearest', vmin=0, vmax=len(ENMAP_COLORS))
    ax.set_title(f"EnMAP — Synthetic Raw CNN Map (from patches, k=8)", fontweight='bold')
    
    patches = [mpatches.Patch(color=ENMAP_COLORS[i], label=f'Cluster {i}') for i in range(8)]
    if -1 in raw_map_2d:
         patches.append(mpatches.Patch(color='lightgray', label='Padding'))
         
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "enmap_raw_cnn_map.png"))
    plt.close()


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 6b — Visualization & Embedding Space Analysis (EnMAP)    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    process_enmap_viz()
    
    print(f"\n  ✓ Week 6b Visualizations complete. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
