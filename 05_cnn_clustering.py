import os
import sys
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (avoids Tkinter threading crash)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tabulate import tabulate

import torch
from torch.utils.data import DataLoader

import hdbscan
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestCentroid

# Project modules
from preprocessing import load_dataset
import importlib.util

from config import INDIAN_PINES_CLASSES, PAVIA_UNIVERSITY_CLASSES, setup_stdout

setup_stdout()

# Load the 04_cnn_autoencoder module dynamically since it starts with a number
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
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week5")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2048  # Inference can handle larger batches

SPECTRAL_SUBSET_SIZE = 5000
RANDOM_SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions (Re-used heavily from Week 3 logic)
# ══════════════════════════════════════════════════════════════════════════════

def extract_embeddings(patches_file, encoder_path):
    """Pass all patches through the trained encoder to get 64D vectors."""
    print(f"    Loading patches: {os.path.basename(patches_file)}")
    dataset = HyperspectralPatchDataset(patches_file)
    in_bands = dataset[0].shape[0]  # Determine correct bands after noise removal
    
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=0, pin_memory=True
    )
    
    # Init encoder part of the autoencoder
    model = HyperspectralAutoencoder(in_bands=in_bands, embedding_dim=64)
    # Load ONLY encoder weights (strict=False)
    model.load_state_dict(torch.load(encoder_path, map_location=DEVICE), strict=False)
    model = model.to(DEVICE)
    model.eval()
    
    embeddings = []
    print("    Computing embeddings via forward pass...", end=" ", flush=True)
    t0 = time.time()
    
    with torch.no_grad():
        from torch.amp import autocast
        for patches in loader:
            patches = patches.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                emb = model.encode(patches)
            embeddings.append(emb.cpu().numpy())
            
    embeddings = np.concatenate(embeddings, axis=0)
    print(f"Done ({time.time()-t0:.1f}s)")
    return embeddings


def run_kmeans(embeddings, n_clusters):
    print(f"    Running KMeans (k={n_clusters})...", end=" ", flush=True)
    t0 = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    print(f"Done ({time.time()-t0:.1f}s)")
    return labels

def run_spectral_sub(embeddings, n_clusters):
    N = len(embeddings)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(N, SPECTRAL_SUBSET_SIZE, replace=False) if N > SPECTRAL_SUBSET_SIZE else np.arange(N)
    subset = embeddings[idx]
    
    print(f"    Running Spectral Clustering on {len(idx)} subset...", end=" ", flush=True)
    t0 = time.time()
    sc = SpectralClustering(
        n_clusters=n_clusters, random_state=RANDOM_SEED,
        affinity='nearest_neighbors', assign_labels='kmeans', n_init=5
    )
    labels = sc.fit_predict(subset)
    print(f"Done ({time.time()-t0:.1f}s)")
    return labels, idx

def run_hdbscan(embeddings, min_cluster_size=20, prefix="", output_dir=""):
    """
    Run HDBSCAN clustering (better for embeddings).
    No eps tuning required.
    """
    from sklearn.preprocessing import StandardScaler

    # ─── 1. Normalize ────────────────────────────────────────────────────
    print(f"    Normalizing embeddings (StandardScaler)...", end=" ", flush=True)
    embeddings = embeddings.astype(np.float32)
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    print("Done")

    # ─── 2. (Optional) PCA — VERY IMPORTANT for high-D ───────────────────
    from sklearn.decomposition import PCA
    emb_scaled = PCA(n_components=20).fit_transform(emb_scaled)

    # ─── 3. Run HDBSCAN ─────────────────────────────────────────────────
    print(f"    Running HDBSCAN (min_cluster_size={min_cluster_size})...", end=" ", flush=True)
    t0 = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,          # auto
        metric='euclidean',
        core_dist_n_jobs=-1
    )

    labels = clusterer.fit_predict(emb_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    N = len(labels)

    print(f"Done ({time.time()-t0:.1f}s) → {n_clusters} clusters, {n_noise} noise pts ({100*n_noise/N:.1f}%)")

    return labels


def run_hierarchical(embeddings, n_clusters):
    """Run Agglomerative (Ward) Hierarchical Clustering."""
    N = len(embeddings)
    SUBSET = 20000  # Ward linkage can be slow on full data
    rng = np.random.default_rng(RANDOM_SEED)
    if N > SUBSET:
        idx = rng.choice(N, SUBSET, replace=False)
        subset = embeddings[idx]
        print(f"    Running Hierarchical Clustering on {SUBSET}/{N} subset...", end=" ", flush=True)
    else:
        idx = np.arange(N)
        subset = embeddings
        print(f"    Running Hierarchical Clustering on all {N} pixels...", end=" ", flush=True)
    t0 = time.time()
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    sub_labels = hc.fit_predict(subset)
    if N > SUBSET:
        # Assign remaining points via nearest centroid
        nc = NearestCentroid()
        nc.fit(subset, sub_labels)
        labels = nc.predict(embeddings)
    else:
        labels = sub_labels
    print(f"Done ({time.time()-t0:.1f}s)")
    return labels


def run_gmm(embeddings, n_clusters):
    """Run Gaussian Mixture Model clustering."""
    print(f"    Running GMM (k={n_clusters})...", end=" ", flush=True)
    t0 = time.time()
    gmm = GaussianMixture(n_components=n_clusters, random_state=RANDOM_SEED,
                          covariance_type='full', max_iter=200, n_init=3)
    labels = gmm.fit_predict(embeddings)
    print(f"Done ({time.time()-t0:.1f}s)")
    return labels


def run_meanshift(embeddings):
    """Run Mean Shift clustering using GPU-accelerated PyTorch."""
    N = len(embeddings)
    SUBSET = 10000
    rng = np.random.default_rng(RANDOM_SEED)
    if N > SUBSET:
        idx_sub = rng.choice(N, SUBSET, replace=False)
        subset = embeddings[idx_sub]
    else:
        subset = embeddings

    sample = subset[:2000]
    dists = torch.cdist(
        torch.tensor(sample, dtype=torch.float32),
        torch.tensor(sample, dtype=torch.float32)
    )

    mean_dist = dists.mean().item()
    bw = 0.2 * mean_dist

    print(f"    Running MeanShift GPU (bw={bw:.2f}, device={DEVICE}) on {len(subset)} pts...", end=" ", flush=True)
    t0 = time.time()

    # GPU Mean Shift iteration
    X = torch.tensor(subset, dtype=torch.float32, device=DEVICE)
    bw_sq = bw * bw
    max_iter = 50
    tol = 1e-3 * bw
    BATCH = 2048

    for iteration in range(max_iter):
        new_X = torch.zeros_like(X)
        for start in range(0, len(X), BATCH):
            end = min(start + BATCH, len(X))
            batch = X[start:end]
            dists_sq = torch.cdist(batch, X, p=2.0).pow(2)
            weights = torch.exp(-0.5 * dists_sq / bw_sq)
            new_X[start:end] = (weights.unsqueeze(-1) * X.unsqueeze(0)).sum(dim=1) / weights.sum(dim=1, keepdim=True)
        shift = torch.norm(new_X - X, dim=1).mean().item()
        X = new_X
        if shift < tol:
            break

    # Merge converged modes
    centers = X.clone()
    labels_gpu = torch.arange(len(X), device=DEVICE)
    for i in range(len(centers)):
        if labels_gpu[i] != i:
            continue
        dists = torch.norm(centers - centers[i], dim=1)
        merge_mask = dists < bw
        labels_gpu[merge_mask] = i

    # Compact labels
    unique_labels = labels_gpu.unique()
    label_map = torch.zeros(labels_gpu.max() + 1, dtype=torch.long, device=DEVICE)
    for new_id, old_id in enumerate(unique_labels):
        label_map[old_id] = new_id
    sub_labels = label_map[labels_gpu]

    n_clusters = len(unique_labels)
    cluster_centers = torch.zeros(n_clusters, X.shape[1], device=DEVICE)
    for c in range(n_clusters):
        cluster_centers[c] = X[sub_labels == c].mean(dim=0)

    # Assign ALL points to nearest cluster center
    all_data = torch.tensor(embeddings, dtype=torch.float32, device=DEVICE)
    all_labels = []
    for start in range(0, len(all_data), BATCH):
        end = min(start + BATCH, len(all_data))
        dists = torch.cdist(all_data[start:end], cluster_centers, p=2.0)
        all_labels.append(dists.argmin(dim=1).cpu().numpy())

    labels = np.concatenate(all_labels)
    print(f"Done ({time.time()-t0:.1f}s) → {n_clusters} clusters ({iteration+1} iters)")
    return labels


def compute_metrics(emb, cl_labels, gt_labels):
    metrics = {}
    
    # Filter noise labels for HDBSCAN (label == -1)
    valid = cl_labels >= 0
    if valid.sum() < 2 or len(np.unique(cl_labels[valid])) < 2:
        return {'silhouette': float('nan'), 'dbi': float('nan'), 'ari': float('nan')}
    
    emb_v, cl_v, gt_v = emb[valid], cl_labels[valid], gt_labels[valid]
    
    # Silhouette
    n = len(emb_v)
    if n > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 10000, replace=False)
        metrics['silhouette'] = silhouette_score(emb_v[idx], cl_v[idx])
    else:
        metrics['silhouette'] = silhouette_score(emb_v, cl_v)
        
    metrics['dbi'] = davies_bouldin_score(emb_v, cl_v)
    
    labeled_mask = gt_v > 0
    if labeled_mask.sum() > 0:
        metrics['ari'] = adjusted_rand_score(gt_v[labeled_mask], cl_v[labeled_mask])
    else:
        metrics['ari'] = float('nan')
        
    return metrics

# ══════════════════════════════════════════════════════════════════════════════
# Main Execution Strategy
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(name, prefix, n_classes, class_names):
    print(f"\n{'═'*60}")
    print(f"  {name} — CNN Embeddings & Clustering")
    print(f"{'═'*60}")
    
    # Paths
    patches_file = os.path.join(PROCESSED_DIR, f"{prefix}_all_patches.npy")
    labels_file = os.path.join(PROCESSED_DIR, f"{prefix}_labels.npy")
    gt_file = os.path.join(PROCESSED_DIR, f"{prefix}_labels.npy") # all_labels.npy has background
    encoder_file = os.path.join(MODELS_DIR, f"{prefix}_encoder.pth")
    
    if not os.path.exists(encoder_file):
        print(f"  [!] Missing encoder: {encoder_file}. Run Week 4 first.")
        return None
        
    # Get ground truth map for full shape and labeled metrics
    dataset_code = "indian_pines" if prefix == "ip" else "pavia_university"
    _, gt_img = load_dataset(dataset_code)
    gt_flat = gt_img.ravel()
    
    # ─── 1. EXTRACT ──────────────────────────────────────────────────────────
    print("\n  [1] Extracting CNN Embeddings...")
    embeddings = extract_embeddings(patches_file, encoder_file)
    print(f"      Embeddings shape: {embeddings.shape}")
    
    # ─── 2. KMEANS ───────────────────────────────────────────────────────────
    print(f"\n  [2] KMeans Clustering...")
    km_labels = run_kmeans(embeddings, n_clusters=n_classes)
    km_metrics = compute_metrics(embeddings, km_labels, gt_flat)
    print(f"      Silhouette : {km_metrics['silhouette']:.4f}")
    print(f"      DBI        : {km_metrics['dbi']:.4f}")
    print(f"      ARI        : {km_metrics['ari']:.4f}")
    
    # ─── 3. SPECTRAL ─────────────────────────────────────────────────────────
    print(f"\n  [3] Spectral Clustering...")
    sc_labels, sc_idx = run_spectral_sub(embeddings, n_clusters=n_classes)
    sc_metrics = compute_metrics(embeddings[sc_idx], sc_labels, gt_flat[sc_idx])
    print(f"      Silhouette : {sc_metrics['silhouette']:.4f}")
    print(f"      DBI        : {sc_metrics['dbi']:.4f}")
    print(f"      ARI        : {sc_metrics['ari']:.4f}")
    
    # ─── 4. HDBSCAN ───────────────────────────────────────────
    print(f"\n  [4] HDBSCAN Clustering...")
    db_labels = run_hdbscan(embeddings, min_cluster_size=20, prefix=prefix, output_dir=OUTPUT_DIR)
    db_metrics = compute_metrics(embeddings, db_labels, gt_flat)
    print(f"      Silhouette : {db_metrics['silhouette']:.4f}")
    print(f"      DBI        : {db_metrics['dbi']:.4f}")
    print(f"      ARI        : {db_metrics['ari']:.4f}")
    
    # ─── 5. HIERARCHICAL ─────────────────────────────────────────────────────
    print(f"\n  [5] Hierarchical Clustering...")
    hc_labels = run_hierarchical(embeddings, n_clusters=n_classes)
    hc_metrics = compute_metrics(embeddings, hc_labels, gt_flat)
    print(f"      Silhouette : {hc_metrics['silhouette']:.4f}")
    print(f"      DBI        : {hc_metrics['dbi']:.4f}")
    print(f"      ARI        : {hc_metrics['ari']:.4f}")
    
    # ─── 6. GMM ──────────────────────────────────────────────────────────────
    print(f"\n  [6] Gaussian Mixture Model Clustering...")
    gmm_labels = run_gmm(embeddings, n_clusters=n_classes)
    gmm_metrics = compute_metrics(embeddings, gmm_labels, gt_flat)
    print(f"      Silhouette : {gmm_metrics['silhouette']:.4f}")
    print(f"      DBI        : {gmm_metrics['dbi']:.4f}")
    print(f"      ARI        : {gmm_metrics['ari']:.4f}")
    
    # ─── 7. MEAN SHIFT ───────────────────────────────────────────────────────
    print(f"\n  [7] Mean Shift Clustering...")
    ms_labels = run_meanshift(embeddings)
    ms_metrics = compute_metrics(embeddings, ms_labels, gt_flat)
    print(f"      Silhouette : {ms_metrics['silhouette']:.4f}")
    print(f"      DBI        : {ms_metrics['dbi']:.4f}")
    print(f"      ARI        : {ms_metrics['ari']:.4f}")
    
    # ─── 8. t-SNE Plot ───────────────────────────────────────────────────────
    print(f"\n  [8] t-SNE projection (saved to outputs/week5/)...")
    from preprocessing import apply_pca # Use TSNE from sklearn
    
    # Subsample for t-SNE
    N = len(embeddings)
    idx = np.random.default_rng(42).choice(N, 5000, replace=False) if N > 5000 else np.arange(N)
    
    sub_emb = embeddings[idx]
    sub_gt = gt_flat[idx]
    sub_km = km_labels[idx]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000, init='pca')
    coords = tsne.fit_transform(sub_emb)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"{name} — t-SNE of CNN Embeddings (64D → 2D)", fontweight='bold', fontsize=14)
    
    # GT Colored
    for c in range(len(class_names)):
        mask = sub_gt == c
        if mask.sum() == 0: continue
        if c == 0:
            ax1.scatter(coords[mask, 0], coords[mask, 1], c='lightgray', s=3, alpha=0.2, label='BG', rasterized=True)
        else:
            ax1.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=class_names[c], rasterized=True)
    ax1.set_title("Colored by Ground Truth")
    ax1.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.7)
    
    # KM Colored
    for c in np.unique(sub_km):
        mask = sub_km == c
        ax2.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, label=f'Cls {c}', rasterized=True)
    ax2.set_title("Colored by CNN+KMeans")
    ax2.legend(fontsize=6, loc='upper right', ncol=2, framealpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{prefix}_cnn_tsne.png"))
    plt.close()
    
    return {
        'kmeans': km_metrics,
        'spectral': sc_metrics,
        'hdbscan': db_metrics,
        'hierarchical': hc_metrics,
        'gmm': gmm_metrics,
        'meanshift': ms_metrics
    }


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 5 — CNN Embeddings & Clustering                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    results = {}
    
    # Indian Pines
    ip_res = process_dataset("Indian Pines", "ip", 16, INDIAN_PINES_CLASSES)
    if ip_res: results["Indian Pines"] = ip_res
    
    # Pavia University
    pu_res = process_dataset("Pavia University", "pu", 9, PAVIA_UNIVERSITY_CLASSES)
    if pu_res: results["Pavia University"] = pu_res
    
    if not results: return
    
    # ─── Compare & Output Table ──────────────────────────────────────────────
    print("\n\n" + "═"*75)
    print("   FINAL EMBEDDING RESULTS — CNN Autoencoder (64 Components)")
    print("═"*75)
    
    table_rows = []
    method_labels = [
        ('kmeans', 'KMeans'), ('spectral', 'Spectral*'),
        ('hdbscan', 'HDBSCAN'), ('hierarchical', 'Hierarchical'),
        ('gmm', 'GMM'), ('meanshift', 'MeanShift')
    ]
    for dname, r in results.items():
        for key, label in method_labels:
            m = r[key]
            table_rows.append([dname, "CNN AE (64D)", label,
                              f"{m['silhouette']:.4f}", f"{m['dbi']:.4f}", f"{m['ari']:.4f}"])
                          
    headers = ["Dataset", "Embedding", "Clustering", "Silhouette(hi)", "DBI(lo)", "ARI(hi)"]
    print(tabulate(table_rows, headers=headers, tablefmt="double_outline"))
    
    # Save CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, "cnn_clustering_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)
        
    print(f"\n  ✓ Results saved to: {csv_path}")

if __name__ == "__main__":
    main()
