"""Permutation-safe and rejection-aware clustering evaluation."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    silhouette_score,
)


def hungarian_match(
    y_true: np.ndarray,
    y_cluster: np.ndarray,
    reject_label: int = -1,
) -> tuple[np.ndarray, dict[int, int]]:
    """Map cluster IDs to class IDs using maximum-overlap Hungarian matching."""
    truth = np.asarray(y_true).ravel()
    clusters = np.asarray(y_cluster).ravel()
    if truth.shape != clusters.shape:
        raise ValueError("y_true and y_cluster must have the same shape")

    usable = clusters != reject_label
    true_ids = np.unique(truth)
    cluster_ids = np.unique(clusters[usable])
    mapped = np.full(clusters.shape, reject_label, dtype=np.int64)
    if len(true_ids) == 0 or len(cluster_ids) == 0:
        return mapped, {}

    contingency = np.zeros((len(cluster_ids), len(true_ids)), dtype=np.int64)
    for row, cluster_id in enumerate(cluster_ids):
        values, counts = np.unique(truth[clusters == cluster_id], return_counts=True)
        positions = np.searchsorted(true_ids, values)
        contingency[row, positions] = counts

    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(cluster_ids[row]): int(true_ids[col]) for row, col in zip(rows, cols)}
    for cluster_id, class_id in mapping.items():
        mapped[clusters == cluster_id] = class_id
    return mapped, mapping


def _mean_iou(y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray) -> float:
    values = []
    for class_id in class_ids:
        truth = y_true == class_id
        pred = y_pred == class_id
        union = np.count_nonzero(truth | pred)
        if union:
            values.append(np.count_nonzero(truth & pred) / union)
    return float(np.mean(values)) if values else math.nan


def _normalized_cluster_entropy(labels: np.ndarray, reject_label: int) -> float:
    values = labels[labels != reject_label]
    _, counts = np.unique(values, return_counts=True)
    if len(counts) < 2:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum() / np.log(len(counts)))


def _internal_metrics(
    features: np.ndarray,
    labels: np.ndarray,
    reject_label: int,
    sample_size: int,
    seed: int,
) -> tuple[float, float]:
    valid = labels != reject_label
    x, y = np.asarray(features)[valid], labels[valid]
    if len(x) < 3 or len(np.unique(y)) < 2 or len(np.unique(y)) >= len(x):
        return math.nan, math.nan
    if len(x) > sample_size:
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(x), sample_size, replace=False)
        sil = silhouette_score(x[selected], y[selected])
    else:
        sil = silhouette_score(x, y)
    return float(sil), float(davies_bouldin_score(x, y))


def evaluate_clustering(
    y_true: np.ndarray,
    y_cluster: np.ndarray,
    *,
    features: np.ndarray | None = None,
    labeled_mask: np.ndarray | None = None,
    reject_label: int = -1,
    sample_size: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    """Evaluate clustering without hiding rejected or unmatched samples.

    Semantic full metrics include the reject label as an ordinary cluster.
    Covered metrics are additionally reported for comparison with density-based
    methods. ACC, macro-F1, and mIoU count rejects/unmatched clusters as errors.
    """
    truth = np.asarray(y_true).ravel()
    clusters = np.asarray(y_cluster).ravel()
    if truth.shape != clusters.shape:
        raise ValueError("y_true and y_cluster must have the same shape")
    mask = truth > 0 if labeled_mask is None else np.asarray(labeled_mask).ravel().astype(bool)
    if mask.shape != truth.shape or not mask.any():
        raise ValueError("labeled_mask must select at least one sample")

    yt, yc = truth[mask], clusters[mask]
    covered = yc != reject_label
    mapped, mapping = hungarian_match(yt, yc, reject_label)
    class_ids = np.unique(yt)

    result: dict[str, float | int] = {
        "n_evaluated": int(len(yt)),
        "n_clusters": int(len(np.unique(yc[covered]))),
        "coverage": float(covered.mean()),
        "ari": float(adjusted_rand_score(yt, yc)),
        "nmi": float(normalized_mutual_info_score(yt, yc)),
        "accuracy": float(np.mean(mapped == yt)),
        "macro_f1": float(f1_score(yt, mapped, labels=class_ids, average="macro", zero_division=0)),
        "miou": _mean_iou(yt, mapped, class_ids),
        "cluster_entropy": _normalized_cluster_entropy(yc, reject_label),
        "matched_clusters": int(len(mapping)),
    }
    if covered.any():
        result["ari_covered"] = float(adjusted_rand_score(yt[covered], yc[covered]))
        result["nmi_covered"] = float(normalized_mutual_info_score(yt[covered], yc[covered]))
    else:
        result["ari_covered"] = math.nan
        result["nmi_covered"] = math.nan

    if features is not None:
        feature_array = np.asarray(features)
        if len(feature_array) != len(truth):
            raise ValueError("features must have one row per label")
        silhouette, dbi = _internal_metrics(
            feature_array[mask], yc, reject_label, sample_size, seed
        )
        result["silhouette"] = silhouette
        result["davies_bouldin"] = dbi
    return result


def semantic_diagnostics(
    y_true: np.ndarray,
    y_cluster: np.ndarray,
    *,
    labeled_mask: np.ndarray | None = None,
    reject_label: int = -1,
) -> dict[str, object]:
    """Return Hungarian mapping and per-class errors for failure analysis."""
    truth = np.asarray(y_true).ravel()
    clusters = np.asarray(y_cluster).ravel()
    if truth.shape != clusters.shape:
        raise ValueError("y_true and y_cluster must have the same shape")
    mask = truth > 0 if labeled_mask is None else np.asarray(labeled_mask).ravel().astype(bool)
    if mask.shape != truth.shape or not mask.any():
        raise ValueError("labeled_mask must select at least one sample")
    yt, yc = truth[mask], clusters[mask]
    mapped, mapping = hungarian_match(yt, yc, reject_label)
    class_ids = np.unique(yt)
    precision, recall, f1, support = precision_recall_fscore_support(
        yt, mapped, labels=class_ids, zero_division=0
    )
    per_class = []
    for index, class_id in enumerate(class_ids):
        truth_class = yt == class_id
        pred_class = mapped == class_id
        union = np.count_nonzero(truth_class | pred_class)
        per_class.append(
            {
                "class_id": int(class_id),
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "iou": float(np.count_nonzero(truth_class & pred_class) / union) if union else 0.0,
            }
        )
    return {
        "mapping": {str(cluster): int(class_id) for cluster, class_id in mapping.items()},
        "per_class": per_class,
        "unmatched_predictions": int(np.count_nonzero(mapped == reject_label)),
    }
