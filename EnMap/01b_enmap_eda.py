"""
=============================================================================
Week 1b — EnMAP Hyperspectral EDA & Visualization
=============================================================================

Scans the `datas/` directory for EnMAP L2A SPECTRAL_IMAGE.TIF files,
loads the first image in full for detailed analysis, and performs a
lightweight multi-image comparison across all available scenes.

No ground-truth labels — all analysis is purely data-driven.

Outputs saved to: outputs/week1b/

"""

import os
import sys
import glob
import re
import warnings
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config import PLOT_RCPARAMS

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(BASE_DIR, "datas")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "week1b")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# EnMAP L2A typically has 224 bands (VNIR 420–1000nm + SWIR 900–2450nm)
# Approximate wavelength centres (nm), linearly spaced as guide
# VNIR: bands 0–91  (420–1000 nm, step ~6.4 nm)
# SWIR: bands 91–223 (900–2450 nm, step ~11.7 nm)
ENMAP_N_BANDS = 224   # fallback; actual value resolved at runtime

# False-colour band indices (0-indexed) — adapted for ~224 bands
FC_VNIR = (62, 37, 22)   # ~832 nm, ~660 nm, ~564 nm → NIR / Red / Green
FC_SWIR = (150, 100, 70) # SWIR-2 / SWIR-1 / NIR

# Approximate wavelength array for x-axis labels (linear approximation)
def approx_wavelengths(n_bands):
    """Return a rough wavelength array [nm] for n EnMAP bands."""
    # VNIR 0–91 → 420–1000 nm; SWIR 92–n-1 → 1000–2450 nm
    n_vnir = min(92, n_bands)
    n_swir = max(0, n_bands - n_vnir)
    wl_vnir = np.linspace(420, 1000, n_vnir)
    wl_swir = np.linspace(1005, 2450, n_swir) if n_swir > 0 else np.array([])
    return np.concatenate([wl_vnir, wl_swir])

SAMPLE_PIXELS = 5000   # pixels to subsample for heavy computations
RANDOM_SEED   = 42
SCENE_DATE_RE = re.compile(r'_(\d{8})T\d{6}Z_')

plt.rcParams.update(PLOT_RCPARAMS)


# ══════════════════════════════════════════════════════════════════════════════
# Utility
# ══════════════════════════════════════════════════════════════════════════════

def save(fig, filename):
    fig.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


def extract_scene_date(scene_name):
    """Return YYYYMMDD extracted from an EnMAP scene folder name."""
    match = SCENE_DATE_RE.search(scene_name)
    return match.group(1) if match else None


def load_enmap_tif(filepath):
    """
    Load an EnMAP SPECTRAL_IMAGE.TIF and return (H, W, B) float32 array.
    Handles both (B,H,W) and (H,W,B) layouts.
    EnMAP nodata fill = -32768 (int16 origin). Replace with NaN.
    """
    # Loading files as float32
    data = tifffile.imread(filepath).astype(np.float32)
    # EnMAP ships as (Bands, H, W)
    if data.ndim == 3 and data.shape[0] > 100 and data.shape[1] > 500:
        data = data.transpose(1, 2, 0)   # → (H, W, B)
    # EnMAP nodata: -32768 (int16 min) or any value <= -10000
    data[data <= -10000] = np.nan
    return data


def pixel_valid_mask(flat_pixels):
    """Keep pixels that contain at least one finite spectral value."""
    return np.isfinite(flat_pixels).any(axis=1)


def fill_missing_with_band_means(flat_pixels):
    """
    Replace NaNs/inf in sampled pixels with band means.
    Used only for algorithms such as correlation that require finite input.
    """
    if len(flat_pixels) == 0:
        return flat_pixels

    filled = flat_pixels.copy()
    band_means = np.nanmean(filled, axis=0)
    band_means = np.where(np.isfinite(band_means), band_means, 0.0)

    missing = ~np.isfinite(filled)
    if missing.any():
        rows, cols = np.where(missing)
        filled[rows, cols] = band_means[cols]

    return filled


def percent_stretch(band_2d, p_lo=2, p_hi=98):
    """Stretch a 2-D band to [0,1] using percentile clipping."""
    lo, hi = np.nanpercentile(band_2d, [p_lo, p_hi])
    out = np.clip((band_2d - lo) / (hi - lo + 1e-8), 0, 1)
    return out


def sample_pixels(data_2d_or_3d, n, seed=RANDOM_SEED):
    """Randomly sample n pixel rows from a (N, B) or (H*W, B) array."""
    rng = np.random.default_rng(seed)
    if data_2d_or_3d.ndim == 3:
        H, W, B = data_2d_or_3d.shape
        flat = data_2d_or_3d.reshape(-1, B)
    else:
        flat = data_2d_or_3d
    idx = rng.choice(len(flat), min(n, len(flat)), replace=False)
    return flat[idx]


# ══════════════════════════════════════════════════════════════════════════════
# EDA Functions — Single Image
# ══════════════════════════════════════════════════════════════════════════════

def plot_false_color(data, bands, title, filename):
    """False-colour composite from three band indices."""
    H, W, B = data.shape
    rgb = np.stack([
        percent_stretch(np.where(np.isnan(data[:, :, bands[i]]), 0,
                                  data[:, :, bands[i]]))
        for i in range(3)
    ], axis=-1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, interpolation='bilinear')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.text(0.5, -0.06,
            f"R: Band {bands[0]}  |  G: Band {bands[1]}  |  B: Band {bands[2]}",
            transform=ax.transAxes, ha='center', fontsize=9,
            style='italic', color='gray')
    save(fig, filename)


def plot_spectral_signature(data, wavelengths, img_label, filename):
    """Mean ± 1-std spectral signature across all valid pixels."""
    H, W, B = data.shape
    flat = data.reshape(-1, B)
    # Mask nodata
    valid_mask = ~np.all(np.isnan(flat), axis=1)
    flat = flat[valid_mask]

    # Subsample to avoid huge computation
    pix = sample_pixels(flat, SAMPLE_PIXELS)
    mean_sig = np.nanmean(pix, axis=0)
    std_sig  = np.nanstd(pix,  axis=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wavelengths, mean_sig, color='#4363d8', linewidth=1.5, label='Mean reflectance')
    ax.fill_between(wavelengths, mean_sig - std_sig, mean_sig + std_sig,
                    color='#4363d8', alpha=0.20, label='±1 Std Dev')

    # Shade water-vapour absorption bands (~940, ~1135, ~1380, ~1900 nm)
    for lo, hi, lbl in [(930, 960, '940nm'), (1110, 1160, '1135nm'),
                         (1350, 1420, '1380nm'), (1800, 1950, '1900nm')]:
        in_range = (wavelengths >= lo) & (wavelengths <= hi)
        if in_range.any():
            ax.axvspan(wavelengths[in_range][0], wavelengths[in_range][-1],
                       alpha=0.12, color='red', label=f'H₂O abs.' if lo == 930 else '')

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Reflectance (L2A physical units)")
    ax.set_title(f"{img_label} — Mean Spectral Signature (±1 σ, {len(pix):,} pixels)",
                 fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    save(fig, filename)


def plot_band_stddev(data, wavelengths, img_label, filename):
    """Per-band standard deviation — highlights noisy / absorption bands."""
    H, W, B = data.shape
    flat = data.reshape(-1, B)
    std_per_band = np.nanstd(flat, axis=0)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(wavelengths, std_per_band, color='#e6194b', linewidth=1.3)
    ax.fill_between(wavelengths, 0, std_per_band, alpha=0.15, color='#e6194b')
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Per-Band Std Dev")
    ax.set_title(f"{img_label} — Per-Band Spectral Variance (scene variability)",
                 fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    save(fig, filename)


def plot_value_distribution(data_raw, data_norm, img_label, filename):
    """Histograms: raw reflectance vs z-score normalised values."""
    H, W, B = data_raw.shape
    # Sample random pixels, pick 10 evenly-spaced bands
    flat_raw  = data_raw.reshape(-1, B)
    flat_norm = data_norm.reshape(-1, B)

    valid = ~np.all(np.isnan(flat_raw), axis=1)
    flat_raw  = flat_raw[valid]
    flat_norm = flat_norm[valid]

    sample_raw  = sample_pixels(flat_raw,  3000).ravel()
    sample_norm = sample_pixels(flat_norm, 3000).ravel()

    # Drop NaNs before plotting histograms
    sample_raw = sample_raw[~np.isnan(sample_raw)]
    sample_norm = sample_norm[~np.isnan(sample_norm)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{img_label} — Pixel Value Distributions", fontweight='bold', fontsize=13)

    axes[0].hist(sample_raw, bins=100, color='#3cb44b', alpha=0.8, edgecolor='none')
    axes[0].set_xlabel("Reflectance (raw L2A)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Raw pixel values")
    axes[0].grid(True, alpha=0.3, linestyle='--')

    axes[1].hist(sample_norm, bins=100, color='#4363d8', alpha=0.8, edgecolor='none')
    axes[1].set_xlabel("Z-score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("After Z-score normalisation")
    axes[1].grid(True, alpha=0.3, linestyle='--')

    save(fig, filename)


def plot_band_correlation(data, img_label, filename):
    """Inter-band Pearson correlation heatmap (sampled pixels)."""
    H, W, B = data.shape
    flat = data.reshape(-1, B)
    flat = flat[pixel_valid_mask(flat)]
    pix = fill_missing_with_band_means(sample_pixels(flat, SAMPLE_PIXELS))
    corr = np.corrcoef(pix.T)   # (B, B)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto',
                   interpolation='nearest')
    ax.set_xlabel("Band Index")
    ax.set_ylabel("Band Index")
    ax.set_title(f"{img_label} — Inter-Band Correlation Matrix", fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation")
    save(fig, filename)


def plot_spatial_stats(data, wavelengths, img_label, filename):
    """
    Spatial heat map of mean reflectance in three broad spectral regions:
      VNIR (420–700 nm), NIR (700–1000 nm), SWIR (1000–2450 nm).
    Helps reveal cloud, shadow, or water bodies.
    """
    wl = wavelengths
    B = data.shape[2]
    # Band index masks
    vnir_mask  = wl < 700
    nir_mask   = (wl >= 700) & (wl < 1000)
    swir_mask  = wl >= 1000

    def region_mean(mask):
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return np.zeros(data.shape[:2])
        sub = np.nanmean(data[:, :, idxs], axis=2)
        return sub

    vnir_img = region_mean(vnir_mask)
    nir_img  = region_mean(nir_mask)
    swir_img = region_mean(swir_mask)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{img_label} — Spatial Mean per Spectral Region", fontweight='bold')
    for ax, img, label, cmap in zip(
        axes,
        [vnir_img, nir_img, swir_img],
        ["VNIR (420–700 nm)", "NIR (700–1000 nm)", "SWIR (1000–2450 nm)"],
        ['YlGn', 'PuRd', 'inferno']
    ):
        im = ax.imshow(img, cmap=cmap, interpolation='bilinear')
        ax.set_title(label)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean Refl.")
    save(fig, filename)


def plot_overview(data, wavelengths, img_label, fc_bands_vnir, filename):
    """2×3 combined overview figure."""
    H, W, B = data.shape
    flat = data.reshape(-1, B)
    flat_valid = flat[pixel_valid_mask(flat)]
    pix = sample_pixels(flat_valid, SAMPLE_PIXELS)
    mean_sig = np.nanmean(pix, axis=0)
    std_sig  = np.nanstd(pix,  axis=0)

    # Build RGB
    rgb = np.stack([
        percent_stretch(np.where(np.isnan(data[:, :, fc_bands_vnir[i]]),
                                  0, data[:, :, fc_bands_vnir[i]]))
        for i in range(3)
    ], axis=-1)

    # Band std
    std_bands = np.nanstd(flat_valid, axis=0)

    # Corr (small sample)
    pix_corr = fill_missing_with_band_means(
        sample_pixels(flat_valid, min(2000, len(flat_valid)))
    )
    corr = np.corrcoef(pix_corr.T)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"{img_label} — Dataset Overview", fontweight='bold', fontsize=15, y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # (0,0) False-colour VNIR
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(rgb, interpolation='bilinear')
    ax0.set_title(f"VNIR False-Colour\n(B{fc_bands_vnir[0]}, B{fc_bands_vnir[1]}, B{fc_bands_vnir[2]})")
    ax0.set_xlabel("Column"); ax0.set_ylabel("Row")

    # (0,1) Mean spectral signature
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(wavelengths, mean_sig, color='#4363d8', linewidth=1.4, label='Mean')
    ax1.fill_between(wavelengths, mean_sig - std_sig, mean_sig + std_sig,
                     color='#4363d8', alpha=0.18, label='±1σ')
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Reflectance")
    ax1.set_title("Mean Spectral Signature")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(wavelengths[0], wavelengths[-1])

    # (0,2) Per-band std dev
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(wavelengths, std_bands, color='#e6194b', linewidth=1.3)
    ax2.fill_between(wavelengths, 0, std_bands, alpha=0.15, color='#e6194b')
    ax2.set_xlabel("Wavelength (nm)")
    ax2.set_ylabel("Std Dev")
    ax2.set_title("Per-Band Scene Variability")
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(wavelengths[0], wavelengths[-1])

    # (1,0) Band correlation (downsampled)
    ax3 = fig.add_subplot(gs[1, 0])
    # Downsample bands for display
    step = max(1, B // 50)
    idx_b = np.arange(0, B, step)
    corr_sub = corr[np.ix_(idx_b, idx_b)]
    im = ax3.imshow(corr_sub, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax3.set_xlabel("Band (downsampled)")
    ax3.set_ylabel("Band (downsampled)")
    ax3.set_title("Inter-Band Correlation")
    fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # (1,1) Pixel value histogram
    ax4 = fig.add_subplot(gs[1, 1])
    vals = pix.ravel()
    vals = vals[~np.isnan(vals)]
    ax4.hist(vals, bins=80, color='#3cb44b', alpha=0.85, edgecolor='none')
    ax4.set_xlabel("Raw Reflectance")
    ax4.set_ylabel("Count")
    ax4.set_title("Pixel Value Distribution (raw)")
    ax4.grid(True, alpha=0.3, linestyle='--')

    # (1,2) Spatial mean (NIR region, ~700–1000nm)
    nir_mask = (wavelengths >= 700) & (wavelengths < 1000)
    idxs = np.where(nir_mask)[0]
    nir_img = np.nanmean(data[:, :, idxs], axis=2) if len(idxs) > 0 else np.zeros(data.shape[:2])
    ax5 = fig.add_subplot(gs[1, 2])
    im5 = ax5.imshow(nir_img, cmap='PuRd', interpolation='bilinear')
    ax5.set_title("Spatial Map — NIR (700–1000 nm)")
    ax5.set_xlabel("Column"); ax5.set_ylabel("Row")
    fig.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04, label="Mean Refl.")

    save(fig, filename)


# ══════════════════════════════════════════════════════════════════════════════
# EDA Functions — Multi-Image Comparison
# ══════════════════════════════════════════════════════════════════════════════

def plot_multi_image_comparison(tif_files, wavelengths, filename):
    """
    For each EnMAP scene: load, subsample pixels, compute mean spectrum.
    Plot all mean spectra on one figure + a bar chart of mean reflectance
    in VNIR and SWIR separately (shows inter-scene variability / seasonality).
    """
    print(f"\n  Multi-image comparison ({len(tif_files)} scenes)...")
    n_images = len(tif_files)
    palette = plt.cm.tab10(np.linspace(0, 1, n_images))
    labels = []

    mean_spectra = []
    vnir_means   = []
    swir_means   = []

    for i, fp in enumerate(tif_files):
        scene_id = os.path.basename(os.path.dirname(fp))
        # Extract date (YYYYMMDD) from folder name
        date_str = extract_scene_date(scene_id)
        if date_str:
            label = f"Scene {i+1}\n({date_str[:4]}-{date_str[4:6]}-{date_str[6:]})"
        else:
            label = f"Scene {i+1}"
        labels.append(label)

        print(f"    [{i+1}/{n_images}] Loading {scene_id[:40]}...")
        data = load_enmap_tif(fp)
        B = data.shape[2]
        wl = wavelengths[:B]

        flat = data.reshape(-1, B)
        valid = pixel_valid_mask(flat)
        flat = flat[valid]
        pix = sample_pixels(flat, SAMPLE_PIXELS)

        mean_sig = np.nanmean(pix, axis=0)
        mean_spectra.append(mean_sig)

        wl_arr = wl
        vnir_mask = wl_arr < 1000
        swir_mask = wl_arr >= 1000
        vnir_means.append(float(np.nanmean(mean_sig[vnir_mask])))
        swir_means.append(float(np.nanmean(mean_sig[swir_mask])))

        del data, flat, pix

    # ── Figure 1: Overlay spectral signatures ─────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, sig in enumerate(mean_spectra):
        wl_i = wavelengths[:len(sig)]
        ax.plot(wl_i, sig, color=palette[i], linewidth=1.3,
                label=labels[i].replace('\n', ' '), alpha=0.85)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Mean Reflectance")
    ax.set_title("EnMAP Multi-Scene Spectral Comparison — Mean Signatures", fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    save(fig, filename.replace(".png", "_spectra.png"))

    # ── Figure 2: VNIR vs SWIR mean bar chart ─────────────────────────────
    x = np.arange(n_images)
    bar_w = 0.35
    short_labels = [f"S{i+1}" for i in range(n_images)]

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    bars1 = ax2.bar(x - bar_w/2, vnir_means, bar_w, label='VNIR (<1000nm)',
                    color='#4363d8', edgecolor='#333', linewidth=0.5)
    bars2 = ax2.bar(x + bar_w/2, swir_means, bar_w, label='SWIR (≥1000nm)',
                    color='#f58231', edgecolor='#333', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Mean Reflectance")
    ax2.set_title("EnMAP — Per-Scene Mean Reflectance: VNIR vs SWIR", fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    # Annotate bars
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    save(fig2, filename.replace(".png", "_region_bars.png"))

    print("  ✓ Multi-image comparison complete.")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Week 1b — EnMAP Hyperspectral EDA & Visualization               ║")
    print("║  Unsupervised Hyperspectral Land Cover Mapping                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")

    # ── Discover TIF files ──────────────────────────────────────────────────
    search_pat = os.path.join(DATA_DIR, "**", "*-SPECTRAL_IMAGE.TIF")
    tif_files = sorted(glob.glob(search_pat, recursive=True))

    if not tif_files:
        print(f"[!] No *-SPECTRAL_IMAGE.TIF files found in {DATA_DIR}")
        return

    print(f"Found {len(tif_files)} EnMAP images:\n")
    for fp in tif_files:
        size_mb = os.path.getsize(fp) / 1024**2
        print(f"  • {os.path.basename(os.path.dirname(fp))[:60]}  ({size_mb:.0f} MB)")

    # ── Load FIRST image for detailed single-scene analysis ─────────────────
    first_fp   = tif_files[0]
    scene_name = os.path.basename(os.path.dirname(first_fp))
    # Extract date string
    date_str = extract_scene_date(scene_name)
    if date_str:
        img_label = f"EnMAP Scene 1 ({date_str[:4]}-{date_str[4:6]}-{date_str[6:]})"
    else:
        img_label = "EnMAP Scene 1"

    print(f"\n{'─'*70}")
    print(f"  DETAILED ANALYSIS: {img_label}")
    print(f"{'─'*70}")
    print("  Loading TIF...")
    data_raw = load_enmap_tif(first_fp)
    H, W, B = data_raw.shape
    print(f"  Shape: {H} × {W} × {B}  |  dtype: {data_raw.dtype}")
    print(f"  Memory: {data_raw.nbytes / 1024**3:.2f} GB")

    # Approximate wavelengths
    wavelengths = approx_wavelengths(B)

    # NaN stats
    n_nodata = (~np.isfinite(data_raw).any(axis=2)).sum()
    n_valid  = H * W - n_nodata
    print(f"  Valid pixels : {n_valid:,} / {H*W:,}  "
          f"({100*n_valid/(H*W):.1f}%)")

    # Summary stats table
    flat = data_raw.reshape(-1, B)
    valid_mask = pixel_valid_mask(flat)
    complete_mask = np.isfinite(flat).all(axis=1)
    flat_valid = flat[valid_mask]
    n_complete = int(complete_mask.sum())
    print(f"  Fully observed spectra : {n_complete:,} / {H*W:,}  "
          f"({100*n_complete/(H*W):.1f}%)")

    if len(flat_valid) == 0:
        print("  [!] WARNING: No valid (non-nodata) pixels found after masking.")
        print("      Check nodata filter in load_enmap_tif().")
        return

    sample = sample_pixels(flat_valid, SAMPLE_PIXELS)
    print(f"\n  Per-pixel statistics ({len(sample):,} sampled pixels):")
    print(f"    Min          : {np.nanmin(sample):.4f}")
    print(f"    Max          : {np.nanmax(sample):.4f}")
    print(f"    Mean         : {np.nanmean(sample):.4f}")
    print(f"    Std          : {np.nanstd(sample):.4f}")
    print(f"    Median       : {np.nanmedian(sample):.4f}")
    print(f"    Bands        : {B}")

    # Z-score normalised copy (for distribution plot only — in-place)
    print("\n  Computing z-score normalisation (for distribution plot)...")
    data_norm = np.zeros_like(data_raw)
    for b in range(B):
        band = data_raw[:, :, b]
        m, s = np.nanmean(band), np.nanstd(band)
        if s > 1e-10:
            data_norm[:, :, b] = (band - m) / s
        else:
            # Dead/empty bands should remain NaN rather than artificially injecting 0.0
            data_norm[:, :, b] = np.nan

    # ── Visualizations — Single Scene ───────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  GENERATING SINGLE-SCENE PLOTS")
    print(f"{'─'*70}")

    # Clamp band indices to actual B
    fc_vnir = tuple(min(b, B-1) for b in FC_VNIR)
    fc_swir = tuple(min(b, B-1) for b in FC_SWIR)

    # 1. VNIR false-colour
    plot_false_color(
        data_raw, fc_vnir,
        title=f"{img_label} — VNIR False-Colour Composite (NIR–Red–Green)",
        filename="enmap_false_color_vnir.png"
    )

    # 2. SWIR false-colour
    plot_false_color(
        data_raw, fc_swir,
        title=f"{img_label} — SWIR False-Colour Composite",
        filename="enmap_false_color_swir.png"
    )

    # 3. Spectral signature
    plot_spectral_signature(
        data_raw, wavelengths, img_label,
        filename="enmap_spectral_signature.png"
    )

    # 4. Per-band std dev (scene variability)
    plot_band_stddev(
        data_raw, wavelengths, img_label,
        filename="enmap_band_stddev.png"
    )

    # 5. Band correlation heatmap
    plot_band_correlation(
        data_raw, img_label,
        filename="enmap_band_correlation.png"
    )

    # 6. Value distributions (raw vs normalised)
    plot_value_distribution(
        data_raw, data_norm, img_label,
        filename="enmap_value_distribution.png"
    )

    # 7. Spatial statistics (mean per spectral region)
    plot_spatial_stats(
        data_raw, wavelengths, img_label,
        filename="enmap_spatial_stats.png"
    )

    # 8. Combined overview figure
    plot_overview(
        data_raw, wavelengths, img_label, fc_vnir,
        filename="enmap_overview.png"
    )

    # Free memory before loading all images
    del data_raw, data_norm

    # ── Multi-image comparison ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  MULTI-SCENE COMPARISON")
    print(f"{'─'*70}")
    wl_global = approx_wavelengths(ENMAP_N_BANDS)
    plot_multi_image_comparison(
        tif_files, wl_global,
        filename="enmap_multi_scene.png"
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    saved_files = sorted(os.listdir(OUTPUT_DIR))
    print(f"\n{'═'*70}")
    print("  Week 1b Complete. All figures saved to:")
    print(f"    {OUTPUT_DIR}")
    print(f"\n  Generated {len(saved_files)} files:")
    for f in saved_files:
        size_kb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"    • {f}  ({size_kb:.0f} KB)")
    print("═"*70)


if __name__ == "__main__":
    main()
