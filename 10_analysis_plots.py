"""Generate analysis plots for the HyperAttnRes comparison."""

import os
import sys
import csv
import warnings
import argparse

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn as nn
import importlib.util

from config import PLOT_RCPARAMS, setup_stdout

setup_stdout()

# ── Load project modules ──────────────────────────────────────────────────────
def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADVANCED_DIR = os.path.join(BASE_DIR, "Advanced")

cnn_mod  = _load_module("cnn_autoencoder",     os.path.join(BASE_DIR, "04_cnn_autoencoder.py"))
har_mod  = _load_module("hyperattnres_model",   os.path.join(ADVANCED_DIR, "06_hyperattnres_model.py"))
std_mod  = _load_module("standard_transformer", os.path.join(ADVANCED_DIR, "07_standard_transformer_ae.py"))

HyperspectralPatchDataset     = cnn_mod.HyperspectralPatchDataset
build_hyperattnres            = har_mod.build_hyperattnres
build_standard_transformer_ae = std_mod.build_standard_transformer_ae
CFG = har_mod.CFG

MODELS_DIR    = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR    = os.path.join(BASE_DIR, "outputs", "week8")
WEEK7_DIR     = os.path.join(BASE_DIR, "outputs", "week7")
WEEK3_DIR     = os.path.join(BASE_DIR, "outputs", "week3")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = CFG["random_seed"]

plt.rcParams.update(PLOT_RCPARAMS)

# Palette for the 3 main models (consistent across all figures)
MODEL_COLORS = {
    "Standard Transformer AE": "#e6194b",
    "HyperAttnRes":            "#4363d8",
    "CNN-AE":                  "#3cb44b",
    "PCA Baseline":            "#f58231",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: load model + get a batch of real patches
# ──────────────────────────────────────────────────────────────────────────────

def _load_sample_batch(prefix, n=256):
    """Load a small batch of patches from the dataset for norm measurement."""
    ds_cfg       = CFG["datasets"][prefix]
    patches_file = os.path.join(BASE_DIR, ds_cfg["patches_file"])
    dataset      = HyperspectralPatchDataset(patches_file)
    rng          = np.random.default_rng(RANDOM_SEED)
    idx          = rng.choice(len(dataset), n, replace=False)
    batch        = torch.stack([dataset[int(i)] for i in idx])
    return batch.to(DEVICE)


def _load_weights(model, prefix, tag):
    """Load state dict from models/. Returns False if weights not found."""
    path = os.path.join(MODELS_DIR, f"{prefix}_{tag}.pth")
    if not os.path.exists(path):
        print(f"  [!] Weights not found: {path}  — skipping.")
        return False
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1: PreNorm Dilution (Output Magnitude Across Layers)
# ══════════════════════════════════════════════════════════════════════════════

def fig_dilution(prefix):
    """
    Plot ‖h_l‖ per transformer layer (12 layers) for:
      - Standard Transformer AE  (expected: monotonically growing)
      - HyperAttnRes N=4         (expected: bounded, periodic within each block)

    This is the key diagnostic figure proving AttnRes fixes PreNorm dilution in HSI.
    """
    print(f"\n  [Fig 1] PreNorm dilution — {prefix.upper()}")

    ds_cfg   = CFG["datasets"][prefix]
    in_bands = np.load(os.path.join(BASE_DIR, ds_cfg["patches_file"]),
                       mmap_mode='r').shape[1]
    batch    = _load_sample_batch(prefix)

    norm_results = {}

    # Standard Transformer AE
    tag   = "transformer_ae"
    model = build_standard_transformer_ae(in_bands=in_bands).to(DEVICE)
    if _load_weights(model, prefix, tag):
        with torch.no_grad():
            _, norms = model.encode(batch, record_norms=True)
        norm_results["Standard Transformer AE"] = norms
        print(f"    Standard Transformer AE: {len(norms)} layer norms")

    # HyperAttnRes N=4
    n_blocks = CFG["n_blocks"]
    tag      = f"hyperattnres_N{n_blocks}"
    model    = build_hyperattnres(in_bands=in_bands, n_blocks=n_blocks).to(DEVICE)
    if _load_weights(model, prefix, tag):
        with torch.no_grad():
            _, norms = model.encode(batch, record_norms=True)
        norm_results["HyperAttnRes"] = norms
        print(f"    HyperAttnRes N={n_blocks}: {len(norms)} layer norms")

    if not norm_results:
        print("  Skipping dilution figure — no weights found.")
        return

    n_layers = CFG["n_blocks"] * CFG["n_layers_per_block"]   # 12
    fig, ax  = plt.subplots(figsize=(10, 5))

    for name, norms in norm_results.items():
        color  = MODEL_COLORS.get(name, "gray")
        layers = list(range(1, len(norms) + 1))
        ax.plot(layers, norms, '-o', label=name, color=color,
                linewidth=2, markersize=5)

    # Draw block boundary lines for HyperAttnRes
    S = CFG["n_layers_per_block"]
    for b in range(1, CFG["n_blocks"]):
        ax.axvline(x=b * S + 0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax.text(0.5, 0.97, "│ Block 1 │ Block 2 │ Block 3 │ Block 4 │",
            transform=ax.transAxes, ha='center', va='top', fontsize=8, color='gray')

    ax.set_xlabel("Transformer Layer Index")
    ax.set_ylabel("Mean Output Norm  ‖h_l‖")
    ax.set_title(f"{ds_cfg['name']} — Output Magnitude vs Depth (PreNorm Dilution)",
                 fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xticks(range(1, n_layers + 1))

    path = os.path.join(OUTPUT_DIR, f"{prefix}_dilution_figure.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: Gradient Magnitude Per Block
# ══════════════════════════════════════════════════════════════════════════════

def fig_gradient_norms(prefix):
    """
    Compare gradient magnitudes per transformer block across the two models.
    Standard residuals → disproportionately large gradients in early blocks.
    AttnRes → more uniform gradient distribution.
    """
    print(f"\n  [Fig 2] Gradient magnitudes — {prefix.upper()}")

    ds_cfg   = CFG["datasets"][prefix]
    in_bands = np.load(os.path.join(BASE_DIR, ds_cfg["patches_file"]),
                       mmap_mode='r').shape[1]
    batch    = _load_sample_batch(prefix, n=64)

    grad_results = {}
    n_blocks = CFG["n_blocks"]
    S        = CFG["n_layers_per_block"]

    for model_name, tag, build_fn, kwargs in [
        ("Standard Transformer AE", "transformer_ae",
         build_standard_transformer_ae, {"in_bands": in_bands}),
        ("HyperAttnRes", f"hyperattnres_N{n_blocks}",
         build_hyperattnres, {"in_bands": in_bands, "n_blocks": n_blocks}),
    ]:
        model = build_fn(**kwargs).to(DEVICE)
        if not _load_weights(model, prefix, tag):
            continue

        model.train()   # need grad tracking
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad()

        loss, _ = model(batch)
        loss.backward()

        # Collect gradient norms per transformer block
        block_grads = []
        if model_name == "HyperAttnRes":
            for block in model.encoder.transformer_blocks:
                g = []
                for p in block.parameters():
                    if p.grad is not None:
                        g.append(p.grad.norm().item())
                block_grads.append(np.mean(g) if g else 0.0)
        else:
            # Standard Transformer: group layers into blocks of S for comparability
            layers = list(model.encoder.layers)
            for b in range(n_blocks):
                g = []
                for layer in layers[b*S : (b+1)*S]:
                    for p in layer.parameters():
                        if p.grad is not None:
                            g.append(p.grad.norm().item())
                block_grads.append(np.mean(g) if g else 0.0)

        grad_results[model_name] = block_grads
        print(f"    {model_name}: block grads = {[f'{v:.2e}' for v in block_grads]}")

    if not grad_results:
        print("  Skipping gradient figure — no weights found.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x       = np.arange(1, n_blocks + 1)
    width   = 0.35

    for i, (name, grads) in enumerate(grad_results.items()):
        offset = (i - 0.5) * width
        color  = MODEL_COLORS.get(name, "gray")
        ax.bar(x + offset, grads, width=width, label=name, color=color, alpha=0.8, edgecolor='white')

    ax.set_xlabel("Transformer Block Index")
    ax.set_ylabel("Mean Gradient Norm")
    ax.set_title(f"{ds_cfg['name']} — Gradient Magnitude per Block",
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Block {i}" for i in range(1, n_blocks + 1)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    path = os.path.join(OUTPUT_DIR, f"{prefix}_gradient_magnitude.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Ablation — N sweep bar chart
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_N(prefix):
    """
    Bar chart: ARI and Silhouette for HyperAttnRes N ∈ {1, 2, 4, 6}.
    Reads from cached embeddings in outputs/week7/.
    """
    print(f"\n  [Fig 3] Ablation N sweep — {prefix.upper()}")

    ds_cfg   = CFG["datasets"][prefix]
    in_bands = np.load(os.path.join(BASE_DIR, ds_cfg["patches_file"]),
                       mmap_mode='r').shape[1]

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    from preprocessing import load_dataset

    _, gt_img = load_dataset(ds_cfg["code"])
    gt_flat   = gt_img.ravel()
    n_classes = ds_cfg["n_classes"]

    ablation_N = CFG["ablation_N"]   # [1, 2, 4, 6]
    ari_vals, sil_vals, valid_N = [], [], []

    for N in ablation_N:
        tag   = f"hyperattnres_N{N}"
        # Check for cached embeddings first
        cache = os.path.join(WEEK7_DIR, f"{prefix}_{tag}_embeddings.npy")
        if os.path.exists(cache):
            emb = np.load(cache)
            print(f"    N={N}: loaded cached embeddings ({emb.shape})")
        else:
            # Extract on the fly
            model = build_hyperattnres(in_bands=in_bands, n_blocks=N).to(DEVICE)
            if not _load_weights(model, prefix, tag):
                print(f"    N={N}: skip (no weights)")
                continue
            from torch.utils.data import DataLoader
            from torch.amp import autocast
            patches_file = os.path.join(BASE_DIR, ds_cfg["patches_file"])
            dataset = HyperspectralPatchDataset(patches_file)
            loader  = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0)
            model.eval()
            batch_embs = []
            with torch.no_grad():
                for patches in loader:
                    patches = patches.to(DEVICE)
                    with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                        e, _ = model.encode(patches)
                    batch_embs.append(e.cpu().float().numpy())
            emb = np.concatenate(batch_embs, axis=0)
            np.save(cache, emb)
            print(f"    N={N}: extracted + cached ({emb.shape})")

        # KMeans
        labels = KMeans(n_clusters=n_classes, random_state=RANDOM_SEED, n_init=10).fit_predict(emb)
        valid  = labels >= 0
        if valid.sum() > 10000:
            rng = np.random.default_rng(RANDOM_SEED)
            idx = rng.choice(valid.sum(), 10000, replace=False)
            sil = silhouette_score(emb[valid][idx], labels[valid][idx])
        else:
            sil = silhouette_score(emb[valid], labels[valid])
        labeled = gt_flat > 0
        ari = adjusted_rand_score(gt_flat[labeled], labels[labeled]) if labeled.sum() > 0 else float('nan')

        sil_vals.append(sil)
        ari_vals.append(ari)
        valid_N.append(N)
        print(f"    N={N}: Silhouette={sil:.4f}  ARI={ari:.4f}")

    if not valid_N:
        print("  Skipping ablation — no weights found.")
        return

    # Bar chart
    x     = np.arange(len(valid_N))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f"{ds_cfg['name']} — Ablation Study: Number of AttnRes Blocks (N)",
                 fontweight='bold')

    axes[0].bar(x, sil_vals, width=0.6, color="#4363d8", edgecolor='white')
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"N={n}" for n in valid_N])
    axes[0].set_ylabel("Silhouette Score ↑")
    axes[0].set_title("Silhouette Score")
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    axes[1].bar(x, ari_vals, width=0.6, color="#e6194b", edgecolor='white')
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"N={n}" for n in valid_N])
    axes[1].set_ylabel("ARI ↑")
    axes[1].set_title("Adjusted Rand Index")
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{prefix}_ablation_N_sweep.png")
    plt.savefig(path)
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(path)}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4: Full 4-Model Comparison Table (CSV + LaTeX)
# ══════════════════════════════════════════════════════════════════════════════

def fig_full_comparison_table():
    """
    Assembles the definitive 4-model comparison table from all result CSVs.
    Also includes PCA Baseline (Week 3) and CNN-AE (Week 5).
    Exports as CSV and LaTeX.
    """
    print(f"\n  [Fig 4] Full comparison table")

    # Try to gather results from downstream scripts' CSVs
    all_rows = []
    headers  = ["Dataset", "Model", "Silhouette↑", "DBI↓", "ARI↑"]

    # Load Week 3 PCA Baseline results
    pca_csv = os.path.join(WEEK3_DIR, "pca_baseline_results.csv")
    if os.path.exists(pca_csv):
        with open(pca_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Clustering", "") == "KMeans":
                    all_rows.append([
                        row["Dataset"], "PCA Baseline (30D)",
                        row["Silhouette(hi)"], row["DBI(lo)"], row["ARI(hi)"]
                    ])
        print(f"    Loaded PCA results: {pca_csv}")

    # Load Week 5 CNN-AE results
    cnn_csv = os.path.join(BASE_DIR, "outputs", "week5", "cnn_clustering_results.csv")
    if os.path.exists(cnn_csv):
        with open(cnn_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Clustering", "") == "KMeans":
                    all_rows.append([
                        row["Dataset"], "CNN-AE (64D)",
                        row["Silhouette(hi)"], row["DBI(lo)"], row["ARI(hi)"]
                    ])
        print(f"    Loaded CNN-AE results: {cnn_csv}")

    # Load Week 7 3-model comparison results
    for prefix in ["ip", "pu"]:
        w7_csv = os.path.join(WEEK7_DIR, f"{prefix}_3model_comparison.csv")
        if os.path.exists(w7_csv):
            with open(w7_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    model = row.get("Model", "")
                    if model not in ("CNN-AE",):  # Already included from week5
                        all_rows.append([
                            row["Dataset"], model,
                            row["Silhouette↑"], row["DBI↓"], row["ARI↑"]
                        ])
            print(f"    Loaded 3-model results: {w7_csv}")

    if not all_rows:
        print("  No result CSVs found yet. Run scripts 03/05/09 first.")
        return

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "full_comparison_table.csv")
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerows([headers] + all_rows)
    print(f"  ✓ CSV saved: {os.path.basename(csv_path)}")

    # Save LaTeX table
    tex_path = os.path.join(OUTPUT_DIR, "full_comparison_table.tex")
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Clustering performance comparison on Indian Pines and Pavia University. "
                "Silhouette$\\uparrow$ and ARI$\\uparrow$ (higher is better); "
                "DBI$\\downarrow$ (lower is better). "
                "All models use KMeans with $k$ equal to the number of ground-truth classes.}\n")
        f.write("\\label{tab:comparison}\n")
        f.write("\\begin{tabular}{llccc}\n")
        f.write("\\toprule\n")
        f.write("Dataset & Model & Silhouette$\\uparrow$ & DBI$\\downarrow$ & ARI$\\uparrow$ \\\\\n")
        f.write("\\midrule\n")
        cur_dataset = ""
        for row in all_rows:
            if row[0] != cur_dataset:
                if cur_dataset:
                    f.write("\\midrule\n")
                cur_dataset = row[0]
            f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"  ✓ LaTeX table saved: {os.path.basename(tex_path)}")

    # Print to console
    from tabulate import tabulate
    print("\n  Full comparison:")
    print(tabulate(all_rows, headers=headers, tablefmt="double_outline"))


# ══════════════════════════════════════════════════════════════════════════════
# CLI + Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["ip", "pu"], default=None,
                   help="Process one dataset. Omit for both.")
    p.add_argument("--skip-ablation", action="store_true",
                   help="Skip the N ablation (runs only dilution + gradient + table)")
    return p.parse_args()


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  10 — Analysis Plots for Paper                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    args     = parse_args()
    prefixes = [args.dataset] if args.dataset else ["ip", "pu"]

    for prefix in prefixes:
        ds_name = CFG["datasets"][prefix]["name"]
        print(f"\n{'═'*65}")
        print(f"  {ds_name}")
        print(f"{'═'*65}")

        fig_dilution(prefix)
        fig_gradient_norms(prefix)

        if not args.skip_ablation:
            fig_ablation_N(prefix)
        else:
            print(f"  [Fig 3] Ablation N — skipped (--skip-ablation)")

    fig_full_comparison_table()

    print(f"\n{'═'*65}")
    print(f"  All analysis plots saved to: outputs/week8/")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
