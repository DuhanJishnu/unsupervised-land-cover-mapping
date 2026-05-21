"""PCA embeddings, clustering baselines, and evaluation metrics."""

import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (avoids Tkinter threading crash)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, adjusted_rand_score
)
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestCentroid
from tabulate import tabulate

from config import (
    INDIAN_PINES_CLASSES,
    IP_COLORS,
    PAVIA_UNIVERSITY_CLASSES,
    PLOT_RCPARAMS,
    PU_COLORS,
)

# Import from project modules
from preprocessing import (
    load_dataset, PreprocessingConfig, normalize_bands,
    remove_noisy_bands, apply_pca
)

# Third-party clustering libraries emit convergence and graph warnings that are
# expected during parameter sweeps; metrics are still recorded for comparison.
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update(PLOT_RCPARAMS)

# Spectral clustering subset size (full data is too slow for spectral clustering)
SPECTRAL_SUBSET_SIZE = 5000
RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# Clustering Functions
# ══════════════════════════════════════════════════════════════════════════════

def run_kmeans(embeddings, n_clusters, seed=42):
    """
    Run KMeans clustering.
    
    Parameters
    ----------
    embeddings : np.ndarray, shape (N, D)
    n_clusters : int
    
    Returns
    -------
    labels : np.ndarray, shape (N,)
    """
    print(f"    Running KMeans (k={n_clusters})...", end=" ", flush=True)
    t0 = time.time()
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=seed,
        n_init=10, max_iter=300
    )
    labels = kmeans.fit_predict(embeddings)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s)")
    return labels


def run_spectral_clustering(embeddings, n_clusters, subset_size=5000, seed=42):
    """
    Run Spectral Clustering on a random subset (full data is too slow).
    
    Returns labels for the SUBSET only, plus the subset indices.
    """
    N = len(embeddings)
    rng = np.random.default_rng(seed)
    
    if N > subset_size:
        idx = rng.choice(N, subset_size, replace=False)
        subset = embeddings[idx]
        print(f"    Running Spectral Clustering on {subset_size}/{N} subset...", end=" ", flush=True)
    else:
        idx = np.arange(N)
        subset = embeddings
        print(f"    Running Spectral Clustering on all {N} pixels...", end=" ", flush=True)
    
    t0 = time.time()
    sc = SpectralClustering(
        n_clusters=n_clusters, random_state=seed,
        affinity='nearest_neighbors', n_neighbors=15,
        assign_labels='kmeans', n_init=5
    )
    labels = sc.fit_predict(subset)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s)")
    
    return labels, idx


def run_dbscan(embeddings, min_samples=10, prefix="", output_dir=""):
    """
    Run DBSCAN with automatic eps selection via k-distance graph.
    1. Normalize embeddings (StandardScaler)
    2. Compute k-distance graph (k = min_samples)
    3. Auto-select eps from elbow point
    4. Save k-distance plot
    5. Run DBSCAN on normalized data
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    # ─── 1. Normalize ────────────────────────────────────────────────────
    print(f"    Normalizing embeddings (StandardScaler)...", end=" ", flush=True)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    print("Done")

    # ─── 2. K-Distance Graph ─────────────────────────────────────────────
    k = min_samples
    N = len(emb_scaled)
    SAMPLE = min(N, 10000)  # subsample for speed
    rng = np.random.default_rng(42)
    if N > SAMPLE:
        idx_sub = rng.choice(N, SAMPLE, replace=False)
        sub = emb_scaled[idx_sub]
    else:
        sub = emb_scaled

    print(f"    Computing k-distance graph (k={k}) on {len(sub)} pts...", end=" ", flush=True)
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(sub)
    distances, _ = nn.kneighbors(sub)
    k_distances = np.sort(distances[:, -1])[::-1]  # sorted descending
    print(f"Done ({time.time()-t0:.1f}s)")

    # ─── 3. Auto-select eps via elbow (max 2nd derivative) ───────────────
    # Smooth the curve slightly to avoid noise in 2nd derivative
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(k_distances, size=max(len(k_distances)//100, 5))
    second_deriv = np.diff(smoothed, n=2)
    elbow_idx = np.argmax(np.abs(second_deriv)) + 1  # +1 for diff offset
    eps_auto = float(k_distances[elbow_idx])
    # Sanity: clamp eps to reasonable range
    eps_auto = max(eps_auto, 0.1)
    print(f"    Auto-selected eps = {eps_auto:.4f} (elbow at index {elbow_idx})")

    # ─── 4. Save k-distance plot ─────────────────────────────────────────
    if output_dir:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(len(k_distances)), k_distances, color='#4363d8', linewidth=1.2)
        ax.axhline(y=eps_auto, color='#e6194b', linestyle='--', linewidth=1.5,
                   label=f'eps = {eps_auto:.4f}')
        ax.axvline(x=elbow_idx, color='gray', linestyle=':', alpha=0.5)
        ax.scatter([elbow_idx], [eps_auto], color='#e6194b', s=80, zorder=5, edgecolors='black')
        ax.set_xlabel("Points (sorted by distance)", fontsize=12)
        ax.set_ylabel(f"{k}-th Nearest Neighbor Distance", fontsize=12)
        ax.set_title(f"{prefix.upper()} — k-Distance Graph (k={k})", fontweight='bold', fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        fname = f"{prefix}_kdistance_graph.png"
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()
        print(f"    ✓ Saved: {fname}")

    # ─── 5. Run DBSCAN with auto eps ────────────────────────────────────
    print(f"    Running DBSCAN (eps={eps_auto:.4f}, min_samples={min_samples})...", end=" ", flush=True)
    t0 = time.time()
    db = DBSCAN(eps=eps_auto, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(emb_scaled)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s) → {n_clusters} clusters, {n_noise} noise pts ({100*n_noise/N:.1f}%)")
    return labels


def run_hierarchical(embeddings, n_clusters, subset_size=20000, seed=42):
    """Run Agglomerative (Ward) Hierarchical Clustering."""
    N = len(embeddings)
    rng = np.random.default_rng(seed)
    if N > subset_size:
        idx = rng.choice(N, subset_size, replace=False)
        subset = embeddings[idx]
        print(f"    Running Hierarchical Clustering on {subset_size}/{N} subset...", end=" ", flush=True)
    else:
        idx = np.arange(N)
        subset = embeddings
        print(f"    Running Hierarchical Clustering on all {N} pixels...", end=" ", flush=True)
    t0 = time.time()
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    sub_labels = hc.fit_predict(subset)
    if N > subset_size:
        nc = NearestCentroid()
        nc.fit(subset, sub_labels)
        labels = nc.predict(embeddings)
    else:
        labels = sub_labels
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s)")
    return labels


def run_gmm(embeddings, n_clusters, seed=42):
    """Run Gaussian Mixture Model clustering."""
    print(f"    Running GMM (k={n_clusters})...", end=" ", flush=True)
    t0 = time.time()
    gmm = GaussianMixture(n_components=n_clusters, random_state=seed,
                          covariance_type='full', max_iter=200, n_init=3)
    labels = gmm.fit_predict(embeddings)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s)")
    return labels


def run_meanshift(embeddings, seed=42):
    """Run Mean Shift clustering using GPU-accelerated PyTorch."""
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    N = len(embeddings)
    SUBSET = 10000
    rng = np.random.default_rng(seed)
    if N > SUBSET:
        idx_sub = rng.choice(N, SUBSET, replace=False)
        subset = embeddings[idx_sub]
    else:
        subset = embeddings

    # Estimate bandwidth on CPU (fast, small sample)
    bw = estimate_bandwidth(subset, quantile=0.1, random_state=seed,
                            n_samples=min(1000, len(subset)))
    if bw == 0:
        bw = 1.0

    print(f"    Running MeanShift GPU (bw={bw:.2f}, device={device}) on {len(subset)} pts...", end=" ", flush=True)
    t0 = time.time()

    # GPU Mean Shift iteration
    X = torch.tensor(subset, dtype=torch.float32, device=device)
    bw_sq = bw * bw
    max_iter = 50
    tol = 1e-3 * bw
    BATCH = 2048  # Process points in batches to avoid OOM

    for iteration in range(max_iter):
        new_X = torch.zeros_like(X)
        for start in range(0, len(X), BATCH):
            end = min(start + BATCH, len(X))
            batch = X[start:end]  # (B, D)
            # Compute squared distances to ALL points
            dists_sq = torch.cdist(batch, X, p=2.0).pow(2)  # (B, N_sub)
            # Gaussian kernel weights
            weights = torch.exp(-0.5 * dists_sq / bw_sq)  # (B, N_sub)
            # Weighted mean → new position
            new_X[start:end] = (weights.unsqueeze(-1) * X.unsqueeze(0)).sum(dim=1) / weights.sum(dim=1, keepdim=True)
        shift = torch.norm(new_X - X, dim=1).mean().item()
        X = new_X
        if shift < tol:
            break

    # Merge converged modes: points within bandwidth are same cluster
    centers = X.clone()
    labels_gpu = torch.arange(len(X), device=device)
    for i in range(len(centers)):
        if labels_gpu[i] != i:
            continue
        dists = torch.norm(centers - centers[i], dim=1)
        merge_mask = dists < bw
        labels_gpu[merge_mask] = i

    # Compact labels
    unique_labels = labels_gpu.unique()
    label_map = torch.zeros(labels_gpu.max() + 1, dtype=torch.long, device=device)
    for new_id, old_id in enumerate(unique_labels):
        label_map[old_id] = new_id
    sub_labels = label_map[labels_gpu]

    # Get cluster centers
    n_clusters = len(unique_labels)
    cluster_centers = torch.zeros(n_clusters, X.shape[1], device=device)
    for c in range(n_clusters):
        cluster_centers[c] = X[sub_labels == c].mean(dim=0)

    # Assign ALL points (full data) to nearest cluster center
    all_data = torch.tensor(embeddings, dtype=torch.float32, device=device)
    all_labels = []
    for start in range(0, len(all_data), BATCH):
        end = min(start + BATCH, len(all_data))
        dists = torch.cdist(all_data[start:end], cluster_centers, p=2.0)
        all_labels.append(dists.argmin(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s) → {n_clusters} clusters ({iteration+1} iters)")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation Functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(embeddings, cluster_labels, gt_labels):
    """
    Compute clustering evaluation metrics.
    Handles DBSCAN noise labels (label == -1) by filtering them out.
    """
    metrics = {}
    
    # Filter noise labels for DBSCAN
    valid = cluster_labels >= 0
    if valid.sum() < 2 or len(np.unique(cluster_labels[valid])) < 2:
        return {'silhouette': float('nan'), 'dbi': float('nan'), 'ari': float('nan')}
    
    emb_v = embeddings[valid]
    cl_v = cluster_labels[valid]
    gt_v = gt_labels[valid]
    
    # Silhouette Score [-1, 1] (higher = better)
    n_samples = len(emb_v)
    if n_samples > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_samples, 10000, replace=False)
        metrics['silhouette'] = silhouette_score(
            emb_v[idx], cl_v[idx], sample_size=None
        )
    else:
        metrics['silhouette'] = silhouette_score(emb_v, cl_v)
    
    # Davies-Bouldin Index [0, inf) (lower = better)
    metrics['dbi'] = davies_bouldin_score(emb_v, cl_v)
    
    # Adjusted Rand Index [-1, 1] (higher = better)
    labeled_mask = gt_v > 0
    if labeled_mask.sum() > 0:
        metrics['ari'] = adjusted_rand_score(
            gt_v[labeled_mask], cl_v[labeled_mask]
        )
    else:
        metrics['ari'] = float('nan')
    
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Visualization Functions
# ══════════════════════════════════════════════════════════════════════════════

def plot_tsne(embeddings, gt_labels, cluster_labels, class_names, dataset_label, filename,
              subset_size=5000):
    """
    Plot t-SNE: side-by-side colored by ground truth and by cluster assignment.
    """
    N = len(embeddings)
    rng = np.random.default_rng(42)
    
    # Subsample for t-SNE speed
    if N > subset_size:
        idx = rng.choice(N, subset_size, replace=False)
    else:
        idx = np.arange(N)
    
    sub_emb = embeddings[idx]
    sub_gt = gt_labels[idx]
    sub_cl = cluster_labels[idx]
    
    print(f"    Running t-SNE on {len(idx)} points...", end=" ", flush=True)
    t0 = time.time()
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca')
    coords = tsne.fit_transform(sub_emb)
    elapsed = time.time() - t0
    print(f"Done ({elapsed:.1f}s)")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{dataset_label} — t-SNE of PCA Embeddings (30D → 2D)", 
                 fontweight='bold', fontsize=14)
    
    # Color by ground truth
    n_classes = len(class_names)
    for cls_id in range(n_classes):
        mask = sub_gt == cls_id
        if mask.sum() == 0:
            continue
        if cls_id == 0:
            ax1.scatter(coords[mask, 0], coords[mask, 1], c='lightgray', s=3,
                       alpha=0.2, label='Background', rasterized=True)
        else:
            ax1.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                       label=class_names[cls_id], rasterized=True)
    ax1.set_title("Colored by Ground Truth")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.7, markerscale=2)
    
    # Color by cluster assignment
    unique_clusters = np.unique(sub_cl)
    for cl in unique_clusters:
        mask = sub_cl == cl
        ax2.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   label=f'Cluster {cl}', rasterized=True)
    ax2.set_title("Colored by KMeans Cluster")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.7, markerscale=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_cluster_map(cluster_labels_2d, n_clusters, colors, title, filename):
    """
    Visualize cluster assignments as a spatial map.
    """
    cmap = ListedColormap(colors[:n_clusters + 1])
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cluster_labels_2d, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    patches = [mpatches.Patch(color=colors[i + 1], label=f'Cluster {i}')
               for i in range(n_clusters)]
    ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=8, frameon=True)
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


def plot_pca_scree(pca_model, dataset_label, filename):
    """Plot a detailed PCA scree plot with cumulative variance."""
    var = pca_model.explained_variance_ratio_
    cum = np.cumsum(var)
    n = len(var)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar(range(1, n + 1), var * 100, color='#4363d8', alpha=0.6, 
           label='Individual', edgecolor='#333', linewidth=0.3)
    ax2 = ax.twinx()
    ax2.plot(range(1, n + 1), cum * 100, 'o-', color='#e6194b', markersize=4,
             linewidth=1.5, label='Cumulative')
    ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.4)
    ax2.set_ylabel("Cumulative Variance (%)", color='#e6194b')
    
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)", color='#4363d8')
    ax.set_title(f"{dataset_label} — PCA Scree Plot", fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=9)
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"  ✓ Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline per Dataset
# ══════════════════════════════════════════════════════════════════════════════

def run_baseline(name, class_names, colors, prefix, dataset_label):
    """Run the full PCA baseline pipeline for one dataset."""
    
    n_classes = len(class_names) - 1  # exclude background
    config = PreprocessingConfig()
    
    print(f"\n{'═' * 60}")
    print(f"  {dataset_label} — PCA Baseline")
    print(f"{'═' * 60}")
    
    # ─── Load & Preprocess ───────────────────────────────────────────────
    print("\n  [1] Loading dataset...")
    data, gt = load_dataset(name)
    H, W, B = data.shape
    print(f"      Shape: {data.shape}, GT classes: {n_classes}")
    
    print("  [2] Normalizing...")
    normalized = normalize_bands(data, method=config.normalization)
    
    noisy = config.ip_noisy_bands if name == "indian_pines" else config.pu_noisy_bands
    print(f"  [3] Removing {len(noisy)} noisy bands...")
    clean, retained = remove_noisy_bands(normalized, noisy)
    print(f"      {B} → {clean.shape[2]} bands")
    
    # ─── PCA Embeddings ──────────────────────────────────────────────────
    print(f"  [4] Computing PCA ({config.pca_components} components)...")
    pca_data, pca_model = apply_pca(clean, n_components=config.pca_components, return_model=True)
    var_explained = pca_model.explained_variance_ratio_.sum()
    print(f"      Variance explained: {var_explained:.4f} ({var_explained*100:.1f}%)")
    
    # Flatten for clustering: (H*W, 30)
    pca_flat = pca_data.reshape(-1, config.pca_components)
    gt_flat = gt.ravel()
    
    # ─── PCA Scree Plot ──────────────────────────────────────────────────
    print("\n  [5] Generating PCA scree plot...")
    plot_pca_scree(pca_model, dataset_label, f"{prefix}_pca_scree.png")
    
    # ─── KMeans Clustering ───────────────────────────────────────────────
    print(f"\n  [6] KMeans Clustering (k={n_classes})...")
    km_labels = run_kmeans(pca_flat, n_clusters=n_classes, seed=RANDOM_SEED)
    
    # Compute metrics
    km_metrics = compute_metrics(pca_flat, km_labels, gt_flat)
    print(f"      Silhouette : {km_metrics['silhouette']:.4f}")
    print(f"      DBI        : {km_metrics['dbi']:.4f}")
    print(f"      ARI        : {km_metrics['ari']:.4f}")
    
    # Cluster assignment map
    km_labels_2d = km_labels.reshape(H, W)
    plot_cluster_map(
        km_labels_2d, n_classes, colors,
        f"{dataset_label} — KMeans Cluster Map (PCA, k={n_classes})",
        f"{prefix}_kmeans_cluster_map.png"
    )
    
    # ─── Spectral Clustering ─────────────────────────────────────────────
    print(f"\n  [7] Spectral Clustering (k={n_classes}, subset={SPECTRAL_SUBSET_SIZE})...")
    sc_labels_sub, sc_idx = run_spectral_clustering(
        pca_flat, n_clusters=n_classes, subset_size=SPECTRAL_SUBSET_SIZE, seed=RANDOM_SEED
    )
    
    sc_metrics = compute_metrics(
        pca_flat[sc_idx], sc_labels_sub, gt_flat[sc_idx]
    )
    print(f"      Silhouette : {sc_metrics['silhouette']:.4f}")
    print(f"      DBI        : {sc_metrics['dbi']:.4f}")
    print(f"      ARI        : {sc_metrics['ari']:.4f}")
    
    # ─── DBSCAN ─────────────────────────────────────────────────────────
    print(f"\n  [8] DBSCAN Clustering...")
    db_labels = run_dbscan(pca_flat, min_samples=10, prefix=prefix, output_dir=OUTPUT_DIR)
    db_metrics = compute_metrics(pca_flat, db_labels, gt_flat)
    print(f"      Silhouette : {db_metrics['silhouette']:.4f}")
    print(f"      DBI        : {db_metrics['dbi']:.4f}")
    print(f"      ARI        : {db_metrics['ari']:.4f}")
    
    # ─── Hierarchical Clustering ────────────────────────────────────────
    print(f"\n  [9] Hierarchical Clustering (k={n_classes})...")
    hc_labels = run_hierarchical(pca_flat, n_clusters=n_classes, seed=RANDOM_SEED)
    hc_metrics = compute_metrics(pca_flat, hc_labels, gt_flat)
    print(f"      Silhouette : {hc_metrics['silhouette']:.4f}")
    print(f"      DBI        : {hc_metrics['dbi']:.4f}")
    print(f"      ARI        : {hc_metrics['ari']:.4f}")
    
    # ─── GMM ────────────────────────────────────────────────────────────
    print(f"\n  [10] Gaussian Mixture Model (k={n_classes})...")
    gmm_labels = run_gmm(pca_flat, n_clusters=n_classes, seed=RANDOM_SEED)
    gmm_metrics = compute_metrics(pca_flat, gmm_labels, gt_flat)
    print(f"      Silhouette : {gmm_metrics['silhouette']:.4f}")
    print(f"      DBI        : {gmm_metrics['dbi']:.4f}")
    print(f"      ARI        : {gmm_metrics['ari']:.4f}")
    
    # ─── Mean Shift ─────────────────────────────────────────────────────
    print(f"\n  [11] Mean Shift Clustering...")
    ms_labels = run_meanshift(pca_flat, seed=RANDOM_SEED)
    ms_metrics = compute_metrics(pca_flat, ms_labels, gt_flat)
    print(f"      Silhouette : {ms_metrics['silhouette']:.4f}")
    print(f"      DBI        : {ms_metrics['dbi']:.4f}")
    print(f"      ARI        : {ms_metrics['ari']:.4f}")
    
    # ─── t-SNE Visualization ───────────────────────────────────────────
    print(f"\n  [12] t-SNE Visualization...")
    plot_tsne(
        pca_flat, gt_flat, km_labels, class_names, dataset_label,
        f"{prefix}_tsne_pca.png", subset_size=5000
    )
    
    # ─── Ground Truth vs KMeans side-by-side ─────────────────────────────
    print(f"\n  [13] Ground Truth vs Cluster comparison...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f"{dataset_label} — Ground Truth vs PCA + KMeans", fontweight='bold', fontsize=14)
    
    gt_cmap = ListedColormap(colors[:len(class_names)])
    ax1.imshow(gt, cmap=gt_cmap, vmin=0, vmax=len(class_names)-1, interpolation='nearest')
    ax1.set_title("Ground Truth")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")
    
    km_cmap = ListedColormap(colors[:n_classes + 1])
    ax2.imshow(km_labels_2d, cmap=km_cmap, interpolation='nearest')
    ax2.set_title(f"KMeans Clusters (k={n_classes})")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_gt_vs_kmeans.png"))
    plt.close()
    print(f"  ✓ Saved: {prefix}_gt_vs_kmeans.png")
    
    return {
        'kmeans': km_metrics,
        'spectral': sc_metrics,
        'dbscan': db_metrics,
        'hierarchical': hc_metrics,
        'gmm': gmm_metrics,
        'meanshift': ms_metrics,
        'pca_var': var_explained
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 3 — PCA Baseline: Embeddings + Clustering               ║")
    print("║  Unsupervised Hyperspectral Land Cover Mapping                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    all_results = {}
    
    # ─── Pavia University ────────────────────────────────────────────────
    pu_results = run_baseline(
        "pavia_university", PAVIA_UNIVERSITY_CLASSES, PU_COLORS, "pu", "Pavia University"
    )
    all_results['Pavia University'] = pu_results
    
    # ══════════════════════════════════════════════════════════════════════
    # Final Results Table
    # ══════════════════════════════════════════════════════════════════════
    
    print("\n\n" + "═" * 70)
    print("   BASELINE RESULTS — PCA Embeddings (30 Components)")
    print("═" * 70)
    
    table_rows = []
    method_labels = [
        ('kmeans', 'KMeans'), ('spectral', 'Spectral*'),
        ('dbscan', 'DBSCAN'), ('hierarchical', 'Hierarchical'),
        ('gmm', 'GMM'), ('meanshift', 'MeanShift')
    ]
    for dataset_name, results in all_results.items():
        for key, label in method_labels:
            m = results[key]
            table_rows.append([
                dataset_name,
                "PCA (30D)",
                label,
                f"{m['silhouette']:.4f}",
                f"{m['dbi']:.4f}",
                f"{m['ari']:.4f}",
            ])
    
    headers = ["Dataset", "Embedding", "Clustering", "Silhouette(hi)", "DBI(lo)", "ARI(hi)"]
    print(tabulate(table_rows, headers=headers, tablefmt="double_outline"))
    print("\n  * Spectral Clustering evaluated on random subset of "
          f"{SPECTRAL_SUBSET_SIZE} pixels")
    print(f"\n  PCA Variance Explained:")
    for name, results in all_results.items():
        print(f"    {name}: {results['pca_var']*100:.1f}%")
    
    # Save results table to CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "pca_baseline_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)
    print(f"\n  ✓ Results saved to: {csv_path}")
    
    # List all output files
    print(f"\n  All output files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    • {f} ({size_kb:.0f} KB)")
    
    print("\n" + "═" * 70)
    print("  Week 3 complete. These baseline results will be compared against")
    print("  CNN Autoencoder embeddings in Week 5.")
    print("═" * 70)


if __name__ == "__main__":
    main()
