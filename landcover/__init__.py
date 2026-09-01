"""Research-grade utilities for unsupervised hyperspectral land-cover mapping.

The numbered scripts in the repository are retained as legacy experiments. New
experiments should use this package so data contracts, metrics, and manifests
remain consistent.
"""

from .config import DATASETS, DatasetSpec, ExperimentConfig

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "ExperimentConfig",
]
