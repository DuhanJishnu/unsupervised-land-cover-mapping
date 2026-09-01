"""Dataset loading, normalization, masks, and coordinate-safe patch access."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import DATASETS, DEFAULT_DATA_DIR, DatasetSpec


@dataclass(frozen=True)
class NormalizationStats:
    method: str
    location: np.ndarray
    scale: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None


@dataclass
class Scene:
    cube: np.ndarray
    ground_truth: np.ndarray
    valid_mask: np.ndarray
    spec: DatasetSpec
    normalization: NormalizationStats | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.cube.shape

    def coordinates(self, valid_only: bool = True) -> np.ndarray:
        if valid_only:
            return np.column_stack(np.where(self.valid_mask)).astype(np.int32)
        rows, cols = np.indices(self.ground_truth.shape)
        return np.column_stack((rows.ravel(), cols.ravel())).astype(np.int32)


def _read_mat_array(path: Path, preferred_key: str) -> np.ndarray:
    import scipy.io as sio

    content = sio.loadmat(path)
    if preferred_key in content:
        return content[preferred_key]
    public = [key for key in content if not key.startswith("__")]
    if len(public) != 1:
        raise KeyError(
            f"Expected key {preferred_key!r} in {path}; available keys: {public}"
        )
    return content[public[0]]


def load_benchmark(dataset: str, data_dir: str | Path = DEFAULT_DATA_DIR) -> Scene:
    """Load a labeled benchmark without altering its spectral bands."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    spec = DATASETS[dataset]
    root = Path(data_dir)
    cube = _read_mat_array(root / spec.data_file, spec.data_key).astype(np.float32)
    gt = _read_mat_array(root / spec.gt_file, spec.gt_key).astype(np.int32)
    if cube.ndim != 3 or gt.shape != cube.shape[:2]:
        raise ValueError(f"Invalid cube/GT shapes: cube={cube.shape}, gt={gt.shape}")

    finite = np.isfinite(cube).all(axis=2)
    informative = np.any(np.abs(np.nan_to_num(cube)) > 0, axis=2)
    valid = finite & informative
    return Scene(cube=cube, ground_truth=gt, valid_mask=valid, spec=spec)


def fit_normalization(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    method: str = "robust",
) -> NormalizationStats:
    """Fit per-band statistics using only declared valid pixels."""
    pixels = cube[valid_mask]
    if pixels.size == 0:
        raise ValueError("Cannot fit normalization: valid mask is empty")
    if not np.isfinite(pixels).all():
        raise ValueError("Valid pixels must contain finite values in every band")

    eps = np.float32(1e-6)
    if method == "robust":
        lower = np.percentile(pixels, 2.0, axis=0).astype(np.float32)
        upper = np.percentile(pixels, 98.0, axis=0).astype(np.float32)
        scale = np.maximum(upper - lower, eps)
        return NormalizationStats(method, lower, scale, lower, upper)
    if method == "zscore":
        mean = pixels.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = pixels.std(axis=0, dtype=np.float64).astype(np.float32)
        return NormalizationStats(method, mean, np.maximum(std, eps))
    if method == "minmax":
        lower = pixels.min(axis=0).astype(np.float32)
        upper = pixels.max(axis=0).astype(np.float32)
        return NormalizationStats(
            method, lower, np.maximum(upper - lower, eps), lower, upper
        )
    raise ValueError("method must be one of: robust, zscore, minmax")


def apply_normalization(cube: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    result = (cube.astype(np.float32, copy=False) - stats.location) / stats.scale
    if stats.method == "robust":
        result = np.clip(result, 0.0, 1.0)
    result[~np.isfinite(result)] = 0.0
    return result.astype(np.float32, copy=False)


def normalize_scene(scene: Scene, method: str = "robust") -> Scene:
    stats = fit_normalization(scene.cube, scene.valid_mask, method)
    return Scene(
        cube=apply_normalization(scene.cube, stats),
        ground_truth=scene.ground_truth,
        valid_mask=scene.valid_mask,
        spec=scene.spec,
        normalization=stats,
    )


def extract_patch(
    cube: np.ndarray,
    row: int,
    col: int,
    patch_size: int,
    padding: str = "reflect",
) -> np.ndarray:
    """Return one `(bands, patch_size, patch_size)` patch."""
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError("patch_size must be a positive odd integer")
    height, width, _ = cube.shape
    if not (0 <= row < height and 0 <= col < width):
        raise IndexError((row, col))
    margin = patch_size // 2
    padded = np.pad(cube, ((margin, margin), (margin, margin), (0, 0)), mode=padding)
    patch = padded[row : row + patch_size, col : col + patch_size]
    return np.moveaxis(patch, -1, 0).astype(np.float32, copy=False)


def iter_patches(
    scene: Scene,
    patch_size: int,
    coordinates: np.ndarray | None = None,
    padding: str = "reflect",
) -> Iterator[tuple[np.ndarray, tuple[int, int]]]:
    """Yield patches lazily, preserving their source coordinates."""
    coords = scene.coordinates(valid_only=True) if coordinates is None else coordinates
    for row, col in np.asarray(coords, dtype=np.int64):
        yield extract_patch(scene.cube, int(row), int(col), patch_size, padding), (
            int(row),
            int(col),
        )
