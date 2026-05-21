"""Reusable preprocessing utilities for hyperspectral imagery."""

import os
import numpy as np
import scipy.io as sio
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from sklearn.decomposition import PCA

from config import INDIAN_PINES_CLASSES, PAVIA_UNIVERSITY_CLASSES

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PreprocessingConfig:
    """Configuration for the preprocessing pipeline."""
    
    # Normalization
    normalization: str = "minmax"  # 'zscore' or 'minmax'
    
    # Noise band removal (0-indexed)
    # Indian Pines: water absorption bands ~104-108, 150-163, 220
    # These are standard bands known to be noisy in the AVIRIS sensor
    ip_noisy_bands: List[int] = field(default_factory=lambda: 
        list(range(103, 109)) + list(range(149, 164)) + [219]
    )
    # Pavia University: ROSIS sensor — all 103 bands are usable
    pu_noisy_bands: List[int] = field(default_factory=list)
    
    # Patch extraction
    patch_size: int = 11  # Spatial neighborhood size (11x11)
    
    # PCA (for baseline only)
    pca_components: int = 64
    
    # Random seed for reproducibility
    random_seed: int = 42


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str, data_dir: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a hyperspectral dataset from .mat files.
    
    Parameters
    ----------
    name : str
        Either 'indian_pines' or 'pavia_university'.
    data_dir : str, optional
        Path to directory containing .mat files. Defaults to datasets/mat_files/.
    
    Returns
    -------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral image cube.
    gt : np.ndarray, shape (H, W)
        Ground truth label map (0 = background).
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                "datasets", "mat_files")
    
    if name == "indian_pines":
        data_file = os.path.join(data_dir, "Indian_pines_corrected.mat")
        gt_file = os.path.join(data_dir, "Indian_pines_gt.mat")
    elif name == "pavia_university":
        data_file = os.path.join(data_dir, "PaviaU.mat")
        gt_file = os.path.join(data_dir, "PaviaU_gt.mat")
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'indian_pines' or 'pavia_university'.")
    
    data_mat = sio.loadmat(data_file)
    gt_mat = sio.loadmat(gt_file)
    
    data_key = [k for k in data_mat.keys() if not k.startswith("__")][0]
    gt_key = [k for k in gt_mat.keys() if not k.startswith("__")][0]
    
    data = data_mat[data_key].astype(np.float64)
    gt = gt_mat[gt_key].astype(np.int32)
    
    return data, gt


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Band Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_bands(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Per-band normalization of hyperspectral data.
    
    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Raw hyperspectral image cube.
    method : str
        'zscore' → zero mean, unit variance per band
        'minmax' → scale each band to [0, 1]
    
    Returns
    -------
    normalized : np.ndarray, shape (H, W, B)
        Normalized image cube. NaN values are ignored when computing band
        statistics and preserved in the output.
    """
    H, W, B = data.shape
    normalized = np.zeros_like(data, dtype=np.float64)
    
    for b in range(B):
        band = data[:, :, b].astype(np.float64, copy=False)
        finite_mask = np.isfinite(band)

        if not finite_mask.any():
            normalized[:, :, b] = np.nan
            continue

        valid = band[finite_mask]
        norm_band = np.full_like(band, np.nan, dtype=np.float64)

        if method == "zscore":
            mean = valid.mean()
            std = valid.std()
            if std > 1e-10:
                norm_band[finite_mask] = (valid - mean) / std
            else:
                norm_band[finite_mask] = valid - mean
        elif method == "minmax":
            bmin, bmax = valid.min(), valid.max()
            if bmax - bmin > 1e-10:
                norm_band[finite_mask] = (valid - bmin) / (bmax - bmin)
            else:
                norm_band[finite_mask] = 0.0
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'.")

        normalized[:, :, b] = norm_band
    
    return normalized


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Noise Band Removal
# ──────────────────────────────────────────────────────────────────────────────

def remove_noisy_bands(data: np.ndarray, noisy_bands: List[int]) -> Tuple[np.ndarray, List[int]]:
    """
    Remove known noisy/water-absorption bands.
    
    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral image cube.
    noisy_bands : list of int
        0-indexed band indices to remove.
    
    Returns
    -------
    clean_data : np.ndarray, shape (H, W, B')
        Cube with noisy bands removed.
    retained_bands : list of int
        Indices of bands that were kept.
    """
    B = data.shape[2]
    all_bands = set(range(B))
    noisy_set = set(noisy_bands)
    retained = sorted(all_bands - noisy_set)
    
    clean_data = data[:, :, retained]
    
    return clean_data, retained


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Patch Extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_patches(
    data: np.ndarray,
    gt: np.ndarray,
    patch_size: int = 7,
    include_unlabeled: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract spatial patches centered at each pixel with zero-padding at borders.
    
    For each pixel (i, j), extract a patch_size × patch_size spatial neighborhood
    from the hyperspectral cube. Boundary pixels are zero-padded.
    
    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Preprocessed hyperspectral cube.
    gt : np.ndarray, shape (H, W)
        Ground truth labels.
    patch_size : int
        Spatial patch size (must be odd).
    include_unlabeled : bool
        If True, extract patches for ALL pixels (for unsupervised training).
        If False, extract only for labeled pixels (gt > 0).
    
    Returns
    -------
    patches : np.ndarray, shape (N, B, patch_size, patch_size)
        Extracted patches in (channels, height, width) format for PyTorch.
    labels : np.ndarray, shape (N,)
        Corresponding ground truth labels.
    coordinates : np.ndarray, shape (N, 2)
        (row, col) coordinates of each patch center pixel.
    """
    H, W, B = data.shape
    margin = patch_size // 2
    
    # Zero-pad the data cube
    padded = np.pad(
        data,
        ((margin, margin), (margin, margin), (0, 0)),
        mode='constant',
        constant_values=0
    )
    
    # Determine which pixels to extract
    if include_unlabeled:
        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        rows = rows.ravel()
        cols = cols.ravel()
    else:
        rows, cols = np.where(gt > 0)
    
    N = len(rows)
    patches = np.zeros((N, B, patch_size, patch_size), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int32)
    coordinates = np.zeros((N, 2), dtype=np.int32)
    
    for idx in range(N):
        r, c = rows[idx], cols[idx]
        # Extract patch from padded array (shift by margin)
        patch = padded[r:r + patch_size, c:c + patch_size, :]  # (ps, ps, B)
        patches[idx] = patch.transpose(2, 0, 1)  # → (B, ps, ps) for PyTorch
        labels[idx] = gt[r, c]
        coordinates[idx] = [r, c]
    
    return patches, labels, coordinates


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: PCA Dimensionality Reduction (for baseline)
# ──────────────────────────────────────────────────────────────────────────────

def apply_pca(
    data: np.ndarray,
    n_components: int = 30,
    return_model: bool = False
) -> Tuple[np.ndarray, Optional[PCA]]:
    """
    Apply PCA to reduce the spectral dimension.
    
    Each pixel's full spectral vector is projected to n_components dimensions.
    PCA is fitted on ALL pixels (unsupervised).
    
    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
        Hyperspectral cube.
    n_components : int
        Number of principal components.
    return_model : bool
        If True, also return the fitted PCA model.
    
    Returns
    -------
    data_pca : np.ndarray, shape (H, W, n_components)
        PCA-transformed cube.
    pca_model : PCA (optional)
        Fitted PCA model (if return_model=True).
    """
    H, W, B = data.shape
    pixels = data.reshape(-1, B)
    
    pca = PCA(n_components=n_components, random_state=42)
    pixels_pca = pca.fit_transform(pixels)
    
    data_pca = pixels_pca.reshape(H, W, n_components)
    
    if return_model:
        return data_pca, pca
    return data_pca, None


# ──────────────────────────────────────────────────────────────────────────────
# I/O: Save & Load Processed Data
# ──────────────────────────────────────────────────────────────────────────────

def save_patches(
    patches: np.ndarray,
    labels: np.ndarray,
    coordinates: np.ndarray,
    output_dir: str,
    prefix: str
) -> None:
    """
    Save extracted patches, labels, and coordinates to .npy files.
    
    Files saved:
        {prefix}_patches.npy    — shape (N, B, ps, ps)
        {prefix}_labels.npy     — shape (N,)
        {prefix}_coords.npy     — shape (N, 2)
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{prefix}_patches.npy"), patches)
    np.save(os.path.join(output_dir, f"{prefix}_labels.npy"), labels)
    np.save(os.path.join(output_dir, f"{prefix}_coords.npy"), coordinates)


def load_patches(
    output_dir: str,
    prefix: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load saved patches, labels, and coordinates."""
    patches = np.load(os.path.join(output_dir, f"{prefix}_patches.npy"))
    labels = np.load(os.path.join(output_dir, f"{prefix}_labels.npy"))
    coords = np.load(os.path.join(output_dir, f"{prefix}_coords.npy"))
    return patches, labels, coords


# ──────────────────────────────────────────────────────────────────────────────
# Full Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_dataset(
    data: np.ndarray,
    gt: np.ndarray,
    dataset_name: str,
    config: PreprocessingConfig = None
) -> dict:
    """
    Run the full preprocessing pipeline on a hyperspectral dataset.
    
    Pipeline:
        1. Band normalization (z-score)
        2. Noise band removal (dataset-specific)
        3. Patch extraction (7×7 with zero-padding)
        4. Optional PCA (30 components)
    
    Parameters
    ----------
    data : np.ndarray, shape (H, W, B)
    gt : np.ndarray, shape (H, W)
    dataset_name : str
        'indian_pines' or 'pavia_university'
    config : PreprocessingConfig
    
    Returns
    -------
    result : dict
        Keys: 'normalized', 'clean', 'retained_bands', 'patches_all',
              'labels_all', 'coords_all', 'patches_labeled', 'labels_labeled',
              'coords_labeled', 'pca_data', 'pca_model', 'config'
    """
    if config is None:
        config = PreprocessingConfig()
    
    result = {'config': config}
    
    # Step 1: Normalize
    print("    [1/4] Normalizing bands...")
    normalized = normalize_bands(data, method=config.normalization)
    result['normalized'] = normalized
    
    # Step 2: Remove noisy bands
    noisy_bands = config.ip_noisy_bands if dataset_name == "indian_pines" else config.pu_noisy_bands
    print(f"    [2/4] Removing {len(noisy_bands)} noisy bands...")
    clean, retained = remove_noisy_bands(normalized, noisy_bands)
    result['clean'] = clean
    result['retained_bands'] = retained
    print(f"          {data.shape[2]} → {clean.shape[2]} bands")
    
    # Step 3: Extract patches (both labeled-only and all pixels)
    print(f"    [3/4] Extracting {config.patch_size}×{config.patch_size} patches...")
    
    # All pixels (for unsupervised training)
    patches_all, labels_all, coords_all = extract_patches(
        clean, gt, patch_size=config.patch_size, include_unlabeled=True
    )
    result['patches_all'] = patches_all
    result['labels_all'] = labels_all
    result['coords_all'] = coords_all
    print(f"          All pixels: {patches_all.shape[0]:,} patches")
    
    # Labeled only (for evaluation)
    patches_labeled, labels_labeled, coords_labeled = extract_patches(
        clean, gt, patch_size=config.patch_size, include_unlabeled=False
    )
    result['patches_labeled'] = patches_labeled
    result['labels_labeled'] = labels_labeled
    result['coords_labeled'] = coords_labeled
    print(f"          Labeled only: {patches_labeled.shape[0]:,} patches")
    
    # Step 4: PCA (on clean data before patch extraction — pixel-level)
    print(f"    [4/4] Applying PCA → {config.pca_components} components...")
    pca_data, pca_model = apply_pca(clean, n_components=config.pca_components, return_model=True)
    result['pca_data'] = pca_data
    result['pca_model'] = pca_model
    var_explained = pca_model.explained_variance_ratio_.sum()
    print(f"          Variance explained: {var_explained:.4f} ({var_explained*100:.1f}%)")
    
    return result
