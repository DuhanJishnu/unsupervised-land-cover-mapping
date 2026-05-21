"""Embedding extraction, clustering, and comparison for advanced models."""

import os
import sys
import csv
import time
import argparse
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from tabulate import tabulate

import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.manifold import TSNE

# ── Load project modules ──────────────────────────────────────────────────────
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import (
    INDIAN_PINES_CLASSES,
    PAVIA_UNIVERSITY_CLASSES,
    PLOT_RCPARAMS,
    setup_stdout,
)

setup_stdout()

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE_DIR = PROJECT_ROOT

cnn_mod  = _load_module("cnn_autoencoder",      os.path.join(BASE_DIR, "04_cnn_autoencoder.py"))
har_mod  = _load_module("hyperattnres_model",    os.path.join(SCRIPT_DIR, "06_hyperattnres_model.py"))
std_mod  = _load_module("standard_transformer",  os.path.join(SCRIPT_DIR, "07_standard_transformer_ae.py"))

from preprocessing import load_dataset

HyperspectralPatchDataset     = cnn_mod.HyperspectralPatchDataset
HyperspectralAutoencoder      = cnn_mod.HyperspectralAutoencoder
build_hyperattnres            = har_mod.build_hyperattnres
build_standard_transformer_ae = std_mod.build_standard_transformer_ae
CFG = har_mod.CFG

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "week7")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 2048
RANDOM_SEED = CFG["random_seed"]

plt.rcParams.update(PLOT_RCPARAMS)

# Cluster color palette (distinct colors)
CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabebe",
    "#469990", "#e6beff", "#9A6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1",
]

# ══════════════════════════════════════════════════════════════════════════════
# Embedding extraction
# ══════════════════════════════════════════════════════════════════════════════

def extract_and_cache(patches_file, model, prefix, model_tag, force_recompute=False):
    """
    Extract embeddings from model; cache to disk so re-runs are instant.
    Returns np.ndarray shape (N, 64).
    """
    cache_path = os.path.join(OUTPUT_DIR, f"{prefix}_{model_tag}_embeddings.npy")

    if os.path.exists(cache_path) and not force_recompute:
        print(f"    Loading cached embeddings: {os.path.basename(cache_path)}")
        return np.load(cache_path)

    print(f"    Extracting embeddings from {model_tag}...", end=" ", flush=True)
    t0 = time.time()

    dataset = HyperspectralPatchDataset(patches_file)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=True)

    model = model.to(DEVICE)
    model.eval()
    embeddings = []

    with torch.no_grad():
        from torch.amp import autocast
        for patches in loader:
            patches = patches.to(DEVICE, non_blocking=True)
            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                emb, _ = model.encode(patches)
            embeddings.append(emb.cpu().float().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    np.save(cache_path, embeddings)
    print(f"Done ({time.time()-t0:.1f}s) | shape={embeddings.shape}")
    print(f"    Saved: {os.path.basename(cache_path)}")
    return embeddings


def load_cnn_ae_model(prefix, in_bands):
    """Load the CNN Autoencoder from Week 4 weights."""
    path = os.path.join(MODELS_DIR, f"{prefix}_autoencoder.pth")
    if not os.path.exists(path):
        print(f"    [!] CNN-AE weights not found: {path}")
        return None
    model = HyperspectralAutoencoder(in_bands=in_bands, embedding_dim=64)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def load_transformer_ae_model(prefix, in_bands):
    """Load the Standard Transformer AE from Week 8 weights."""
    path = os.path.join(MODELS_DIR, f"{prefix}_transformer_ae.pth")
    if not os.path.exists(path):
        print(f"    [!] Transformer AE weights not found: {path}")
        return None
    model = build_standard_transformer_ae(in_bands=in_bands)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def load_hyperattnres_model(prefix, in_bands, n_blocks=4):
    """Load HyperAttnRes from Week 8 weights."""
    tag  = f"hyperattnres_N{n_blocks}"
    path = os.path.join(MODELS_DIR, f"{prefix}_{tag}.pth")
    if not os.path.exists(path):
        print(f"    [!] HyperAttnRes weights not found: {path}")
        return None
    model = build_hyperattnres(in_bands=in_bands, n_blocks=n_blocks)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(emb, labels, gt_flat):
    """Compute Silhouette, DBI, ARI — handles subsampling for large datasets."""
    valid = labels >= 0
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {'silhouette': float('nan'), 'dbi': float('nan'), 'ari': float('nan')}

    emb_v, cl_v, gt_v = emb[valid], labels[valid], gt_flat[valid]

    # Silhouette — subsample if needed
    n = len(emb_v)
    if n > 10000:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n, 10000, replace=False)
        sil = silhouette_score(emb_v[idx], cl_v[idx])
    else:
        sil = silhouette_score(emb_v, cl_v)

    dbi = davies_bouldin_score(emb_v, cl_v)

    labeled = gt_v > 0
    ari = adjusted_rand_score(gt_v[labeled], cl_v[labeled]) if labeled.sum() > 0 else float('nan')

    return {'silhouette': sil, 'dbi': dbi, 'ari': ari}


def run_kmeans(emb, n_clusters):
    print(f"    KMeans (k={n_clusters})...", end=" ", flush=True)
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(emb)
    print(f"Done ({time.time()-t0:.1f}s)")
    return labels


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def _run_tsne(emb, max_pts=5000):
    N = len(emb)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(N, max_pts, replace=False) if N > max_pts else np.arange(N)
    print(f"    t-SNE on {len(idx)}/{N} pts...", end=" ", flush=True)
    t0 = time.time()
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED,
                perplexity=30, n_iter=1000, init='pca')
    coords = tsne.fit_transform(emb[idx])
    print(f"Done ({time.time()-t0:.1f}s)")
    return coords, idx


def plot_3model_tsne(emb_dict, km_dict, gt_flat, class_names, dataset_name, prefix):
    """
    3-panel t-SNE: colored by GT, then CNN-AE KMeans, Transformer-AE KMeans,
    HyperAttnRes KMeans.
    """
    # Use HyperAttnRes embeddings for t-SNE (or first available)
    base_key = "HyperAttnRes" if "HyperAttnRes" in emb_dict else list(emb_dict.keys())[0]
    coords, idx = _run_tsne(emb_dict[base_key])
    sub_gt = gt_flat[idx]

    # Determine layout: 1 GT panel + N model panels
    model_keys = list(km_dict.keys())
    n_panels   = 1 + len(model_keys)
    fig, axes  = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    fig.suptitle(f"{dataset_name} — t-SNE Embedding Comparison (64D → 2D)",
                 fontweight='bold', fontsize=14)

    # Panel 0: Ground truth
    ax = axes[0]
    for c in range(len(class_names)):
        mask = sub_gt == c
        if mask.sum() == 0: continue
        color = 'lightgray' if c == 0 else None
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=6, alpha=0.5 if c == 0 else 0.7,
                   label=class_names[c], rasterized=True)
    ax.set_title("Ground Truth")
    ax.legend(fontsize=5, loc='upper right', ncol=2, framealpha=0.7)
    ax.set_xticks([]); ax.set_yticks([])

    # Panels 1..N: each model's KMeans
    for i, key in enumerate(model_keys):
        sub_km = km_dict[key][idx]
        ax = axes[i + 1]
        for c in np.unique(sub_km):
            mask = sub_km == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=6, alpha=0.7, rasterized=True,
                       color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)])
        ax.set_title(key)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_3model_tsne.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(path)}")


def plot_3model_landcover(km_dict, gt_img, n_classes, class_names, dataset_name, prefix):
    """
    Land cover maps: Ground Truth + one column per model.
    """
    H, W = gt_img.shape
    model_keys = list(km_dict.keys())
    n_panels   = 1 + len(model_keys)
    fig, axes  = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    fig.suptitle(f"{dataset_name} — Land Cover Map Comparison", fontweight='bold', fontsize=14)

    gt_cmap = ListedColormap(CLUSTER_COLORS[:n_classes + 1])
    axes[0].imshow(gt_img, cmap=gt_cmap, interpolation='nearest')
    axes[0].set_title("Ground Truth")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    cmap = ListedColormap(CLUSTER_COLORS[:n_classes + 1])
    for i, key in enumerate(model_keys):
        ax = axes[i + 1]
        labels_2d = km_dict[key].reshape(H, W)
        ax.imshow(labels_2d, cmap=cmap, interpolation='nearest')
        ax.set_title(key)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_3model_landcover.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Per-Dataset Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_dataset(prefix):
    ds_cfg       = CFG["datasets"][prefix]
    dataset_name = ds_cfg["name"]
    n_classes    = ds_cfg["n_classes"]
    patches_file = os.path.join(BASE_DIR, ds_cfg["patches_file"])
    class_names  = INDIAN_PINES_CLASSES if prefix == "ip" else PAVIA_UNIVERSITY_CLASSES

    print(f"\n{'═'*65}")
    print(f"  {dataset_name} — 3-Model Comparison")
    print(f"{'═'*65}")

    # Ground truth
    _, gt_img = load_dataset(ds_cfg["code"])
    gt_flat   = gt_img.ravel()

    # in_bands from patch file
    sample   = np.load(patches_file, mmap_mode='r')
    in_bands = sample.shape[1]
    print(f"  in_bands={in_bands}, patches={sample.shape[0]:,}")

    # ── Load all 3 models ─────────────────────────────────────────────────────
    models_info = {
        "CNN-AE":         load_cnn_ae_model(prefix, in_bands),
        "Transformer-AE": load_transformer_ae_model(prefix, in_bands),
        "HyperAttnRes":   load_hyperattnres_model(prefix, in_bands, n_blocks=CFG["n_blocks"]),
    }

    available = {k: v for k, v in models_info.items() if v is not None}
    if not available:
        print("  [!] No trained models found. Run 08_hyperattnres_training.py first.")
        return {}

    # ── Extract / load cached embeddings ─────────────────────────────────────
    emb_dict = {}
    tag_map = {"CNN-AE": "cnn_ae", "Transformer-AE": "transformer_ae",
               "HyperAttnRes": f"hyperattnres_N{CFG['n_blocks']}"}

    print("\n  [1] Load / Extract embeddings:")
    for model_name, model in available.items():
        emb = extract_and_cache(patches_file, model, prefix, tag_map[model_name])
        emb_dict[model_name] = emb

    # ── KMeans clustering ─────────────────────────────────────────────────────
    print(f"\n  [2] KMeans (k={n_classes}):")
    km_dict     = {}
    metrics_dict = {}
    for model_name, emb in emb_dict.items():
        print(f"    {model_name}:")
        labels = run_kmeans(emb, n_clusters=n_classes)
        km_dict[model_name]      = labels
        metrics_dict[model_name] = compute_metrics(emb, labels, gt_flat)
        m = metrics_dict[model_name]
        print(f"      Silhouette: {m['silhouette']:.4f} | DBI: {m['dbi']:.4f} | ARI: {m['ari']:.4f}")

    # ── Visualizations ────────────────────────────────────────────────────────
    print(f"\n  [3] t-SNE comparison plot:")
    plot_3model_tsne(emb_dict, km_dict, gt_flat, class_names, dataset_name, prefix)

    print(f"\n  [4] Land cover map comparison:")
    plot_3model_landcover(km_dict, gt_img, n_classes, class_names, dataset_name, prefix)

    # ── Save per-dataset CSV table ────────────────────────────────────────────
    headers = ["Dataset", "Model", "Silhouette↑", "DBI↓", "ARI↑"]
    rows    = []
    for model_name, m in metrics_dict.items():
        rows.append([dataset_name, model_name,
                     f"{m['silhouette']:.4f}", f"{m['dbi']:.4f}", f"{m['ari']:.4f}"])

    csv_path = os.path.join(OUTPUT_DIR, f"{prefix}_3model_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\n  ✓ Saved CSV: {os.path.basename(csv_path)}")

    print(f"\n  Results for {dataset_name}:")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    return {dataset_name: metrics_dict}


# ══════════════════════════════════════════════════════════════════════════════
# CLI + Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ip", "pu"], default=None,
                   help="Process one dataset. Omit for both.")
    return p.parse_args()


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  09 — 3-Model Embedding Comparison                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    args = parse_args()
    prefixes = [args.dataset] if args.dataset else ["ip", "pu"]

    all_results = {}
    for prefix in prefixes:
        res = process_dataset(prefix)
        all_results.update(res)

    if not all_results:
        return

    # ── Combined table across all datasets ───────────────────────────────────
    print("\n\n" + "═" * 70)
    print("   FULL 3-MODEL COMPARISON (KMeans Clustering)")
    print("═" * 70)

    all_rows = []
    headers  = ["Dataset", "Model", "Silhouette↑", "DBI↓", "ARI↑"]
    for dname, model_results in all_results.items():
        for model_name, m in model_results.items():
            all_rows.append([dname, model_name,
                             f"{m['silhouette']:.4f}", f"{m['dbi']:.4f}", f"{m['ari']:.4f}"])

    print(tabulate(all_rows, headers=headers, tablefmt="double_outline"))

    combined_csv = os.path.join(OUTPUT_DIR, "3model_full_comparison.csv")
    with open(combined_csv, 'w', newline='') as f:
        csv.writer(f).writerows([headers] + all_rows)
    print(f"\n  ✓ Full table saved: {combined_csv}")


if __name__ == "__main__":
    main()
