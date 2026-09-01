"""Controlled baseline experiments for the rebuilt pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

from .config import ExperimentConfig
from .data import Scene, normalize_scene
from .evaluation import evaluate_clustering


def run_pixel_baseline(
    scene: Scene,
    config: ExperimentConfig,
    representation: str = "pca",
    clusterer: str = "kmeans",
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int]]:
    """Run a whole-scene transductive pixel baseline.

    Returns the full 2-D cluster map, valid-pixel representation, and metrics.
    Invalid pixels receive label -1. Ground truth is accessed only after fitting.
    """
    config.validate()
    normalized = normalize_scene(scene, config.normalization)
    valid_pixels = normalized.cube[normalized.valid_mask]

    if representation == "raw":
        embeddings = valid_pixels
    elif representation == "pca":
        components = min(config.pca_components, valid_pixels.shape[1], len(valid_pixels) - 1)
        if components < 2:
            raise ValueError("Not enough valid samples/bands for PCA")
        embeddings = PCA(n_components=components, random_state=config.seed).fit_transform(valid_pixels)
    else:
        raise ValueError("representation must be 'raw' or 'pca'")

    k = config.resolved_clusters()
    if clusterer == "kmeans":
        labels = KMeans(n_clusters=k, n_init=20, random_state=config.seed).fit_predict(embeddings)
    elif clusterer == "gmm":
        labels = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            n_init=3,
            random_state=config.seed,
        ).fit_predict(embeddings)
    else:
        raise ValueError("clusterer must be 'kmeans' or 'gmm'")

    full_labels = np.full(normalized.ground_truth.shape, -1, dtype=np.int32)
    full_labels[normalized.valid_mask] = labels
    metrics = evaluate_clustering(
        normalized.ground_truth.ravel(),
        full_labels.ravel(),
        features=_full_feature_matrix(embeddings, normalized.valid_mask),
        labeled_mask=(normalized.ground_truth > 0).ravel() & normalized.valid_mask.ravel(),
        seed=config.seed,
    )
    return full_labels, embeddings, metrics


def _full_feature_matrix(valid_features: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    full = np.zeros((valid_mask.size, valid_features.shape[1]), dtype=np.float32)
    full[valid_mask.ravel()] = valid_features
    return full


def estimate_cluster_count(
    features: np.ndarray,
    k_min: int,
    k_max: int,
    *,
    seed: int = 42,
    repeats: int = 5,
    fit_fraction: float = 0.8,
    silhouette_sample: int = 5000,
) -> tuple[int, dict[str, object]]:
    """Estimate k from subsample stability and silhouette without labels."""
    x = np.asarray(features)
    if x.ndim != 2 or len(x) < 10:
        raise ValueError("features must be a 2-D array with at least 10 samples")
    if not 2 <= k_min <= k_max < len(x):
        raise ValueError("require 2 <= k_min <= k_max < n_samples")
    if repeats < 2 or not 0.1 <= fit_fraction <= 1.0:
        raise ValueError("repeats must be >=2 and fit_fraction in [0.1, 1]")

    rng = np.random.default_rng(seed)
    diagnostics = []
    for k in range(k_min, k_max + 1):
        assignments = []
        silhouettes = []
        for repeat in range(repeats):
            fit_size = max(k * 10, round(len(x) * fit_fraction))
            fit_size = min(fit_size, len(x))
            fit_indices = rng.choice(len(x), fit_size, replace=False)
            model = KMeans(n_clusters=k, n_init=10, random_state=seed + repeat)
            model.fit(x[fit_indices])
            labels = model.predict(x)
            assignments.append(labels)
            sample = (
                rng.choice(len(x), silhouette_sample, replace=False)
                if len(x) > silhouette_sample
                else np.arange(len(x))
            )
            silhouettes.append(float(silhouette_score(x[sample], labels[sample])))
        stability_values = [
            adjusted_rand_score(assignments[left], assignments[right])
            for left in range(repeats)
            for right in range(left + 1, repeats)
        ]
        stability = float(np.mean(stability_values))
        silhouette = float(np.mean(silhouettes))
        # Both terms are bounded; map silhouette [-1,1] to [0,1].
        score = 0.5 * stability + 0.5 * ((silhouette + 1.0) / 2.0)
        diagnostics.append(
            {
                "k": k,
                "stability": stability,
                "silhouette": silhouette,
                "score": score,
            }
        )
    best = max(diagnostics, key=lambda row: (row["score"], row["stability"], -row["k"]))
    return int(best["k"]), {
        "method": "subsample-stability-plus-silhouette",
        "k_min": k_min,
        "k_max": k_max,
        "repeats": repeats,
        "fit_fraction": fit_fraction,
        "candidates": diagnostics,
    }


def pca_reference_features(
    scene: Scene,
    normalization: str,
    components: int,
    seed: int,
) -> tuple[Scene, np.ndarray]:
    normalized = normalize_scene(scene, normalization)
    pixels = normalized.cube[normalized.valid_mask]
    count = min(components, pixels.shape[1], len(pixels) - 1)
    if count < 2:
        raise ValueError("Not enough samples or bands for a PCA reference")
    return normalized, PCA(n_components=count, random_state=seed).fit_transform(pixels)
