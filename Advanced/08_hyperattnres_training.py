"""Training pipeline for HyperAttnRes and transformer autoencoder models."""

import os
import sys
import time
import argparse
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

# ── Import project modules ────────────────────────────────────────────────────
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from config import PLOT_RCPARAMS, setup_stdout

setup_stdout()

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

BASE_DIR     = PROJECT_ROOT
cnn_mod      = _load_module("cnn_autoencoder",     os.path.join(BASE_DIR, "04_cnn_autoencoder.py"))
har_mod      = _load_module("hyperattnres_model",   os.path.join(SCRIPT_DIR, "06_hyperattnres_model.py"))
std_mod      = _load_module("standard_transformer", os.path.join(SCRIPT_DIR, "07_standard_transformer_ae.py"))

HyperspectralPatchDataset    = cnn_mod.HyperspectralPatchDataset
build_hyperattnres           = har_mod.build_hyperattnres
build_standard_transformer_ae = std_mod.build_standard_transformer_ae
CFG = har_mod.CFG

# ── Dirs ─────────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week6")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plt.rcParams.update(PLOT_RCPARAMS)

# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model,
    dataset_name: str,
    prefix: str,
    patches_file: str,
    tag: str,
    epochs: int,
    resume: bool = False,
):
    """
    Train one model on one dataset.

    tag   : short identifier string for filenames, e.g. 'hyperattnres_N4' or 'transformer_ae'
    """
    print(f"\n{'═'*70}")
    print(f"  Training: {dataset_name}  |  Model: {tag}  |  Epochs: {epochs}")
    print(f"{'═'*70}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = HyperspectralPatchDataset(patches_file)
    loader  = DataLoader(
        dataset,
        batch_size=CFG["batch_size"],
        shuffle=True,
        num_workers=CFG["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    in_bands = dataset[0].shape[0]
    print(f"  Dataset  : {len(dataset):,} patches, {in_bands} bands")
    print(f"  Device   : {DEVICE.type.upper()}"
          f" ({torch.cuda.get_device_name() if DEVICE.type=='cuda' else ''})")
    print(f"  Batch    : {CFG['batch_size']} × {len(loader)} batches/epoch")

    # ── Rebuild model with correct in_bands (done in caller, passed in) ─────
    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # ── Optimizer + LR schedule ──────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG["learning_rate"],
        weight_decay=CFG["weight_decay"],
    )
    # Cosine annealing (no warmup for simplicity; consistent across models)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    scaler = GradScaler("cuda", enabled=DEVICE.type == "cuda")

    # ── Checkpoint paths ─────────────────────────────────────────────────────
    ckpt_path  = os.path.join(MODELS_DIR, f"{prefix}_{tag}_ckpt.pth")
    final_path = os.path.join(MODELS_DIR, f"{prefix}_{tag}.pth")

    start_epoch = 1
    epoch_losses = []

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch  = ckpt["epoch"] + 1
        epoch_losses = ckpt["epoch_losses"]
        print(f"  Resumed from epoch {start_epoch - 1}")
    elif resume:
        print(f"  [!] No checkpoint found at {ckpt_path}. Starting fresh.")

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\n  Starting from epoch {start_epoch} / {epochs}...\n")
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, patches in enumerate(loader):
            patches = patches.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                loss, _ = model(patches)

            scaler.scale(loss).backward()
            # Gradient clipping — helps with AttnRes early training
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * patches.size(0)

            if batch_idx % max(1, len(loader) // 5) == 0 and batch_idx > 0:
                print(f"    Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(loader)}] "
                      f"Loss: {loss.item():.4f}  LR: {scheduler.get_last_lr()[0]:.2e}")

        scheduler.step()
        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        print(f"  → Epoch {epoch:03d} | Avg MSE Loss: {epoch_loss:.6f}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == epochs:
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "scheduler":    scheduler.state_dict(),
                "scaler":       scaler.state_dict(),
                "epoch_losses": epoch_losses,
            }, ckpt_path)
            print(f"    ✓ Checkpoint saved (epoch {epoch})")

    total_time = time.time() - start_time
    print(f"\n  Training complete in {total_time/60:.1f} min | "
          f"Final loss: {epoch_losses[-1]:.6f}")

    # ── Save final weights ───────────────────────────────────────────────────
    torch.save(model.state_dict(), final_path)
    print(f"  Saved final weights: models/{os.path.basename(final_path)}")

    # ── Loss curve ───────────────────────────────────────────────────────────
    plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_{tag}_loss.png")
    _plot_loss(epoch_losses, dataset_name, tag, plot_path)

    return epoch_losses


def _plot_loss(losses, dataset_name, tag, path):
    """Save loss curve plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, label='Train MSE (masked bands)')
    ax.set_title(f"{dataset_name} — {tag} Training Loss", fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (masked bands only)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    if len(losses) > 2 and losses[0] / max(losses[-1], 1e-9) > 10:
        ax.set_yscale('log')
    plt.savefig(path)
    plt.close()
    print(f"  Saved loss curve: {os.path.basename(path)}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train HyperAttnRes or Standard Transformer AE")
    p.add_argument("--dataset", choices=["ip", "pu"],  required=True,
                   help="Dataset: ip=Indian Pines, pu=Pavia University")
    p.add_argument("--model",   choices=["hyperattnres", "transformer_ae"], required=True,
                   help="Model architecture to train")
    p.add_argument("--N",       type=int, default=None,
                   help="Number of AttnRes blocks (hyperattnres only). "
                        "Default taken from config (4). Ablation: 1,2,4,6")
    p.add_argument("--epochs",  type=int, default=None,
                   help="Override config epoch count")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from last checkpoint if it exists")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Resolve dataset config ────────────────────────────────────────────────
    ds_cfg      = CFG["datasets"][args.dataset]
    dataset_name  = ds_cfg["name"]
    prefix        = args.dataset
    patches_file  = os.path.join(BASE_DIR, ds_cfg["patches_file"])
    epochs        = args.epochs if args.epochs is not None else CFG["epochs"]

    if not os.path.exists(patches_file):
        print(f"[!] Patches file not found: {patches_file}")
        print(f"    Run the Week 2 preprocessing pipeline first.")
        sys.exit(1)

    # ── Determine in_bands from the patch file ───────────────────────────────
    print(f"  Loading patch shape from {os.path.basename(patches_file)}...")
    sample = np.load(patches_file, mmap_mode='r')
    in_bands = sample.shape[1]   # (N, bands, 7, 7)
    print(f"  in_bands = {in_bands}")

    # ── Build model ───────────────────────────────────────────────────────────
    if args.model == "hyperattnres":
        n_blocks = args.N if args.N is not None else CFG["n_blocks"]
        tag   = f"hyperattnres_N{n_blocks}"
        model = build_hyperattnres(in_bands=in_bands, n_blocks=n_blocks)
        print(f"\n  Model: HyperAttnRes  N={n_blocks}  "
              f"(S={CFG['n_layers_per_block']} layers/block, "
              f"total={n_blocks * CFG['n_layers_per_block']} layers)")
    else:
        tag   = "transformer_ae"
        model = build_standard_transformer_ae(in_bands=in_bands)
        n_layers = CFG["n_blocks"] * CFG["n_layers_per_block"]
        print(f"\n  Model: Standard Transformer AE  ({n_layers} layers, no AttnRes)")

    print(f"  Tag: {tag}")
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  08 — HyperAttnRes Training ({dataset_name})              ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    train(
        model=model,
        dataset_name=dataset_name,
        prefix=prefix,
        patches_file=patches_file,
        tag=tag,
        epochs=epochs,
        resume=args.resume,
    )

    print(f"\n{'═'*70}")
    print(f"  Done. Weights saved to models/{prefix}_{tag}.pth")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
