"""
=============================================================================
Week 5b — EnMAP CNN Embeddings & Clustering
=============================================================================

Extracts 64D embeddings from the trained EnMAP CNN autoencoder and performs
unsupervised clustering with six methods.

Key difference vs Week 5 (IP/PU): EnMAP has NO ground-truth labels.
  → ARI is omitted. Only Silhouette Score and Davies-Bouldin Index are used.
  → N_CLUSTERS = 8 (realistic for satellite land cover: water, urban,
    cropland, forest, grassland, wetland, barren, cloud/snow)

Inputs:
    processed_data/enmap_train_patches.npy  → (N, Bands, 7, 7) patches
    models/enmap_encoder.pth               → trained encoder weights

Outputs saved to: outputs/week5b/

"""

import os
import sys
import time
import warnings
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

import hdbscan
from sklearn.cluster import (
    KMeans, SpectralClustering, AgglomerativeClustering
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
import umap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import PLOT_RCPARAMS, setup_stdout

setup_stdout()

# Load CNN autoencoder module dynamically (filename starts with digit)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cnn_autoencoder",
    os.path.join(BASE_DIR, "04_cnn_autoencoder.py")
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
MODELS_DIR    = os.path.join(BASE_DIR, "models")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "week5b")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2048   # Larger batch fine for inference

# Number of clusters for parametric methods (KMeans, Spectral, HC, GMM)
# 8 is a reasonable choice for real land cover:
# water, urban, cropland, forest, grassland, wetland, barren, cloud/snow
N_CLUSTERS = 8

SPECTRAL_SUBSET_SIZE   = 5000   # Subset for spectral clustering (memory)
HIERARCHICAL_SUBSET    = 20000  # Subset for Ward linkage (speed)
MEANSHIFT_SUBSET       = 10000  # Subset for GPU mean-shift iterations
SILHOUETTE_SAMPLE      = 10000  # Max sample for silhouette (speed)

RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
plt.rcParams.update(PLOT_RCPARAMS)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Extract Embeddings
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(patches_file, encoder_path):
    """
    Pass all patches through the trained encoder.
    Returns np.ndarray of shape (N, embedding_dim).
    """
    print(f"  Loading patches: {os.path.basename(patches_file)}")
    dataset  = HyperspectralPatchDataset(patches_file)
    in_bands = dataset[0].shape[0]
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)

    model = HyperspectralAutoencoder(in_bands=in_bands, embedding_dim=64)
    model.load_state_dict(
        torch.load(encoder_path, map_location=DEVICE), strict=False
    )
    model = model.to(DEVICE)
    model.eval()

    embeddings = []
    print("  Computing embeddings via forward pass...", end=" ", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for patches in loader:
            patches = patches.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                emb = model.encode(patches)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Done ({time.time()-t0:.1f}s) → shape {embeddings.shape}")
    return embeddings


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Clustering Methods
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(emb, k):
    print(f"  KMeans (k={k})...", end=" ", flush=True)
    t0 = time.time()
    km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(emb)
    print(f"Done ({time.time()-t0:.1f}s)  →  {len(np.unique(labels))} clusters")
    return labels


def run_spectral(emb, k, subset_size):
    N   = len(emb)
    rng = np.random.default_rng(RANDOM_SEED)
    if N > subset_size:
        idx = rng.choice(N, subset_size, replace=False)
        sub = emb[idx]
        print(f"  Spectral Clustering (k={k}, n={subset_size}/{N})...", end=" ", flush=True)
    else:
        idx = np.arange(N)
        sub = emb
        print(f"  Spectral Clustering (k={k}, n={N})...", end=" ", flush=True)

    t0 = time.time()
    sc = SpectralClustering(n_clusters=k, random_state=RANDOM_SEED,
                             affinity='nearest_neighbors', assign_labels='kmeans',
                             n_init=5)
    sub_labels = sc.fit_predict(sub)
    
    if N > subset_size:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(sub, sub_labels)
        labels = knn.predict(emb)
    else:
        labels = sub_labels

    print(f"Done ({time.time()-t0:.1f}s)  →  {len(np.unique(labels))} clusters")
    return labels, idx


def run_hdbscan(emb):
    """HDBSCAN with UMAP pre-reduction (better for 64D embeddings)."""
    print("  HDBSCAN with UMAP...", end=" ", flush=True)
    emb_f = emb.astype(np.float32)
    scaler = StandardScaler()
    emb_s = scaler.fit_transform(emb_f)
    
    t0 = time.time()
    reducer = umap.UMAP(n_components=5, n_neighbors=50, min_dist=0.1, random_state=RANDOM_SEED)
    emb_umap = reducer.fit_transform(emb_s)

    min_size = max(200, len(emb) // 100)  # e.g. 250 for 25k points
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size, min_samples=50,
        metric='euclidean', core_dist_n_jobs=-1
    )
    labels = clusterer.fit_predict(emb_umap)
    n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
    n_ns   = int((labels == -1).sum())
    print(f"Done ({time.time()-t0:.1f}s)  →  {n_cl} clusters, {n_ns} noise pts "
          f"({100*n_ns/len(labels):.1f}%)")
    return labels


def run_hierarchical(emb, k, subset_size):
    N   = len(emb)
    rng = np.random.default_rng(RANDOM_SEED)
    if N > subset_size:
        idx = rng.choice(N, subset_size, replace=False)
        sub = emb[idx]
        print(f"  Hierarchical Ward (k={k}, n={subset_size}/{N})...",
              end=" ", flush=True)
    else:
        idx = np.arange(N)
        sub = emb
        print(f"  Hierarchical Ward (k={k}, n={N})...", end=" ", flush=True)

    t0 = time.time()
    hc = AgglomerativeClustering(n_clusters=k, linkage='ward')
    sub_labels = hc.fit_predict(sub)

    if N > subset_size:
        nc = NearestCentroid()
        nc.fit(sub, sub_labels)
        labels = nc.predict(emb)
    else:
        labels = sub_labels

    print(f"Done ({time.time()-t0:.1f}s)  →  {len(np.unique(labels))} clusters")
    return labels


def run_gmm(emb, k):
    print(f"  GMM (k={k})...", end=" ", flush=True)
    t0 = time.time()
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_SEED,
                           covariance_type='full', max_iter=300, n_init=3)
    labels = gmm.fit_predict(emb)
    print(f"Done ({time.time()-t0:.1f}s)  →  {len(np.unique(labels))} clusters")
    return labels


def run_meanshift(emb, subset_size):
    """GPU-accelerated Mean Shift (mirrors 05_cnn_clustering.py logic)."""
    N   = len(emb)
    rng = np.random.default_rng(RANDOM_SEED)
    idx_sub = rng.choice(N, min(subset_size, N), replace=False)
    subset  = emb[idx_sub]

    # Estimate bandwidth from pairwise distances on a micro-sample
    sample = subset[:min(2000, len(subset))]
    dists  = torch.cdist(
        torch.tensor(sample, dtype=torch.float32),
        torch.tensor(sample, dtype=torch.float32)
    )
    mean_dist = dists.mean().item()
    bw = 0.2 * mean_dist
    bw = max(bw, 1e-3)

    print(f"  Mean Shift GPU (bw={bw:.3f}, n={len(subset)})...",
          end=" ", flush=True)
    t0 = time.time()

    X      = torch.tensor(subset, dtype=torch.float32, device=DEVICE)
    bw_sq  = bw * bw
    max_iter = 50
    tol    = 1e-3 * bw
    BATCH  = 2048

    for iteration in range(max_iter):
        new_X = torch.zeros_like(X)
        for start in range(0, len(X), BATCH):
            end  = min(start + BATCH, len(X))
            batch = X[start:end]
            d2   = torch.cdist(batch, X, p=2.0).pow(2)
            w    = torch.exp(-0.5 * d2 / bw_sq)
            new_X[start:end] = (w.unsqueeze(-1) * X.unsqueeze(0)).sum(1) / \
                                w.sum(1, keepdim=True)
        shift = torch.norm(new_X - X, dim=1).mean().item()
        X     = new_X
        if shift < tol:
            break

    # Merge converged modes
    centers    = X.clone()
    labels_gpu = torch.arange(len(X), device=DEVICE)
    for i in range(len(centers)):
        if labels_gpu[i] != i:
            continue
        d = torch.norm(centers - centers[i], dim=1)
        labels_gpu[d < bw] = i

    unique_labels = labels_gpu.unique()
    label_map = torch.zeros(int(labels_gpu.max().item()) + 1,
                            dtype=torch.long, device=DEVICE)
    for new_id, old_id in enumerate(unique_labels):
        label_map[old_id] = new_id
    sub_labels = label_map[labels_gpu]

    n_cl = len(unique_labels)
    cluster_centers = torch.zeros(n_cl, X.shape[1], device=DEVICE)
    for c in range(n_cl):
        mask = sub_labels == c
        if mask.any():
            cluster_centers[c] = X[mask].mean(dim=0)

    # Assign ALL embeddings to nearest centre
    all_data  = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
    all_labels = []
    for start in range(0, len(all_data), BATCH):
        end = min(start + BATCH, len(all_data))
        d   = torch.cdist(all_data[start:end], cluster_centers, p=2.0)
        all_labels.append(d.argmin(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    print(f"Done ({time.time()-t0:.1f}s)  →  {n_cl} clusters ({iteration+1} iters)")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Metrics (no ARI — no ground truth)
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(emb, labels):
    """
    Compute unsupervised clustering metrics.
    - Silhouette Score  (higher is better, max 1.0)
    - Davies-Bouldin Index  (lower is better)
    ARI is NOT computed — EnMAP has no ground-truth labels.
    """
    # Filter noise points (HDBSCAN label == -1)
    valid = labels >= 0
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {'silhouette': float('nan'), 'dbi': float('nan')}

    emb_v, lab_v = emb[valid], labels[valid]

    # Silhouette (subsample for speed)
    n = len(emb_v)
    if n > SILHOUETTE_SAMPLE:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, SILHOUETTE_SAMPLE, replace=False)
        sil = silhouette_score(emb_v[idx], lab_v[idx])
    else:
        sil = silhouette_score(emb_v, lab_v)

    dbi = davies_bouldin_score(emb_v, lab_v)
    return {'silhouette': sil, 'dbi': dbi}


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Visualizations
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne(emb, labels_dict, filename):
    """
    t-SNE projection of embeddings coloured by each clustering method.
    labels_dict: {'KMeans': array, 'HDBSCAN': array, ...}
    """
    N = len(emb)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(N, min(5000, N), replace=False)
    sub_emb = emb[idx]

    print("  Running t-SNE (5000 samples, 64D → 2D)...", end=" ", flush=True)
    t0 = time.time()
    tsne   = TSNE(n_components=2, random_state=RANDOM_SEED,
                  perplexity=30, n_iter=1000, init='pca')
    coords = tsne.fit_transform(sub_emb)
    print(f"Done ({time.time()-t0:.1f}s)")

    n_methods = len(labels_dict)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.ravel() if nrows > 1 else [axes] if ncols == 1 else list(axes)

    for ax_idx, (method_name, all_labels) in enumerate(labels_dict.items()):
        ax   = axes[ax_idx]
        sub_labels = all_labels[idx]
        unique_cls = np.unique(sub_labels)
        cmap = plt.cm.tab20

        for c in unique_cls:
            mask = sub_labels == c
            color = 'lightgray' if c == -1 else cmap(c % 20 / 20)
            alpha = 0.2 if c == -1 else 0.6
            label_str = 'Noise' if c == -1 else f'C{c}'
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=[color], s=6, alpha=alpha,
                       label=label_str, rasterized=True)

        n_cl = len(unique_cls) - (1 if -1 in unique_cls else 0)
        ax.set_title(f"{method_name}  ({n_cl} clusters)", fontweight='bold')
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.6,
                  markerscale=2)

    # Hide unused axes
    for ax_idx in range(n_methods, len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle("EnMAP — t-SNE of CNN Embeddings (64D → 2D)",
                 fontweight='bold', fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


def plot_metrics_comparison(results_dict, filename):
    """
    Bar chart comparing Silhouette and DBI across all clustering methods.
    """
    methods = list(results_dict.keys())
    sil_vals = [results_dict[m]['silhouette'] for m in methods]
    dbi_vals = [results_dict[m]['dbi'] for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("EnMAP — Clustering Method Comparison", fontweight='bold', fontsize=14)

    x = np.arange(len(methods))
    palette = plt.cm.Set2(np.linspace(0, 0.8, len(methods)))

    # Silhouette (higher = better)
    bars = ax1.bar(x, sil_vals, color=palette, edgecolor='#333', linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha='right')
    ax1.set_ylabel("Silhouette Score")
    ax1.set_title("Silhouette Score (↑ higher = better)")
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, sil_vals):
        if not np.isnan(val):
            ax1.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    # DBI (lower = better)
    bars2 = ax2.bar(x, dbi_vals, color=palette, edgecolor='#333', linewidth=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha='right')
    ax2.set_ylabel("Davies-Bouldin Index")
    ax2.set_title("Davies-Bouldin Index (↓ lower = better)")
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars2, dbi_vals):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.02,
                     f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


def plot_cluster_size_distribution(labels_dict, filename):
    """Bar plots of cluster size distribution for each method."""
    n_methods = len(labels_dict)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4*nrows))
    axes_flat = axes.ravel() if hasattr(axes, 'ravel') else [axes]
    fig.suptitle("EnMAP — Cluster Size Distributions",
                 fontweight='bold', fontsize=14)

    for ax_idx, (name, labels) in enumerate(labels_dict.items()):
        ax = axes_flat[ax_idx]
        unique, counts = np.unique(labels, return_counts=True)

        bar_colors = ['#cccccc' if u == -1 else plt.cm.tab20(u % 20 / 20)
                      for u in unique]
        bar_labels  = [f'Noise' if u == -1 else f'C{u}' for u in unique]

        ax.bar(bar_labels, counts, color=bar_colors,
               edgecolor='#333', linewidth=0.5)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Patch Count")
        ax.set_title(name, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    for ax_idx in range(n_methods, len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 5b — EnMAP CNN Embeddings & Clustering                   ║")
    print("║  Unsupervised Hyperspectral Land Cover Mapping                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    if torch.cuda.is_available():
        print(f"\n  [✓] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n  [!] WARNING: CUDA not available — running on CPU (slower).")

    patches_file = os.path.join(PROCESSED_DIR, "enmap_train_patches.npy")
    encoder_file = os.path.join(MODELS_DIR, "enmap_encoder.pth")

    # Existence checks
    if not os.path.exists(patches_file):
        print(f"\n  [!] Patches file not found: {patches_file}")
        print("      Run 02b_enmap_preprocessing.py first.")
        return
    if not os.path.exists(encoder_file):
        print(f"\n  [!] Encoder not found: {encoder_file}")
        print("      Run 04b_enmap_autoencoder.py first.")
        return

    print(f"\n  Patches : {patches_file.split(os.sep)[-1]}  "
          f"({os.path.getsize(patches_file)/1024**3:.2f} GB)")
    print(f"  Encoder : {encoder_file.split(os.sep)[-1]}")

    # ── Step 1: Extract embeddings ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  [1] Extracting CNN Embeddings...")
    print(f"{'─'*60}")
    embeddings = extract_embeddings(patches_file, encoder_file)
    N, D = embeddings.shape
    print(f"  Total embeddings: {N:,} × {D}D")

    # ── Step 2: Clustering ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [2] Clustering  (N_CLUSTERS={N_CLUSTERS} for parametric methods)")
    print(f"{'─'*60}")

    all_labels = {}
    results    = {}

    # — KMeans —
    labels = run_kmeans(embeddings, N_CLUSTERS)
    all_labels['KMeans']  = labels
    results['KMeans']     = compute_metrics(embeddings, labels)

    # — Spectral —
    labels, _ = run_spectral(embeddings, N_CLUSTERS, SPECTRAL_SUBSET_SIZE)
    all_labels['Spectral']  = labels
    results['Spectral']     = compute_metrics(embeddings, labels)

    # — HDBSCAN —
    labels = run_hdbscan(embeddings)
    all_labels['HDBSCAN'] = labels
    results['HDBSCAN']    = compute_metrics(embeddings, labels)

    # — Hierarchical —
    labels = run_hierarchical(embeddings, N_CLUSTERS, HIERARCHICAL_SUBSET)
    all_labels['Hierarchical'] = labels
    results['Hierarchical']    = compute_metrics(embeddings, labels)

    # — GMM —
    labels = run_gmm(embeddings, N_CLUSTERS)
    all_labels['GMM']  = labels
    results['GMM']     = compute_metrics(embeddings, labels)

    # — Mean Shift —
    labels = run_meanshift(embeddings, MEANSHIFT_SUBSET)
    all_labels['MeanShift'] = labels
    results['MeanShift']    = compute_metrics(embeddings, labels)

    # ── Step 3: Print metrics table ────────────────────────────────────────
    print(f"\n\n{'═'*65}")
    print("   FINAL RESULTS — EnMAP CNN Autoencoder (64D Embeddings)")
    print(f"{'═'*65}")
    print("   Note: ARI omitted (no ground-truth labels available)\n")

    table_rows = []
    for method, m in results.items():
        n_unique = len(np.unique(all_labels[method]))
        n_noise  = int((all_labels[method] == -1).sum())
        sil_str  = f"{m['silhouette']:.4f}" if not np.isnan(m['silhouette']) else "N/A"
        dbi_str  = f"{m['dbi']:.4f}"        if not np.isnan(m['dbi'])        else "N/A"
        table_rows.append([method, n_unique, n_noise, sil_str, dbi_str])

    headers = ["Method", "#Clusters", "#Noise pts",
               "Silhouette ↑", "Davies-Bouldin ↓"]
    print(tabulate(table_rows, headers=headers, tablefmt="double_outline"))

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "enmap_clustering_results.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Embedding"] + headers)
        for row in table_rows:
            writer.writerow(["EnMAP", "CNN AE (64D)"] + row)
    print(f"\n  ✓ CSV saved: {csv_path}")

    # ── Step 4: Visualizations ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  [3] Generating Visualizations...")
    print(f"{'─'*60}")

    # t-SNE coloured by each method
    plot_tsne(embeddings, all_labels, "enmap_tsne_all_methods_3.png")

    # Metrics bar comparison
    plot_metrics_comparison(results, "enmap_metrics_comparison_3.png")

    # Cluster size distributions
    plot_cluster_size_distribution(all_labels, "enmap_cluster_sizes_3.png")

    # ── Summary ───────────────────────────────────────────────────────────
    saved = sorted(os.listdir(OUTPUT_DIR))
    print(f"\n{'═'*65}")
    print("  Week 5b Complete. All outputs saved to:")
    print(f"    {OUTPUT_DIR}")
    print(f"\n  {len(saved)} files generated:")
    for f in saved:
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    • {f}  ({sz:.0f} KB)")
    print("═"*65)


if __name__ == "__main__":
    main()
