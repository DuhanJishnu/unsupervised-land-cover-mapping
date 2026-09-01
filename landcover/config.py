"""Typed configuration for the rebuilt research pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    display_name: str
    data_file: str
    data_key: str
    gt_file: str
    gt_key: str
    class_count: int
    sensor: str
    corrected: bool = True


DATASETS: dict[str, DatasetSpec] = {
    "ip": DatasetSpec(
        key="ip",
        display_name="Indian Pines",
        data_file="Indian_pines_corrected.mat",
        data_key="indian_pines_corrected",
        gt_file="Indian_pines_gt.mat",
        gt_key="indian_pines_gt",
        class_count=16,
        sensor="AVIRIS",
        corrected=True,
    ),
    "pu": DatasetSpec(
        key="pu",
        display_name="Pavia University",
        data_file="PaviaU.mat",
        data_key="paviaU",
        gt_file="PaviaU_gt.mat",
        gt_key="paviaU_gt",
        class_count=9,
        sensor="ROSIS",
        corrected=True,
    ),
}


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: Literal["ip", "pu"]
    seed: int = 42
    patch_size: int = 7
    embedding_dim: int = 64
    pca_components: int = 30
    normalization: Literal["robust", "zscore", "minmax"] = "robust"
    cluster_count_protocol: Literal["oracle", "estimated", "explicit"] = "oracle"
    n_clusters: int | None = None
    k_estimation: dict[str, Any] | None = None

    def resolved_clusters(self) -> int:
        if self.n_clusters is not None:
            if self.n_clusters < 2:
                raise ValueError("n_clusters must be at least 2")
            return self.n_clusters
        if self.cluster_count_protocol == "oracle":
            return DATASETS[self.dataset].class_count
        raise ValueError(f"{self.cluster_count_protocol}-k requires a resolved n_clusters value")

    def validate(self) -> None:
        if self.patch_size < 1 or self.patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if self.embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2")
        if self.pca_components < 2:
            raise ValueError("pca_components must be at least 2")
        self.resolved_clusters()

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "mat_files"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "research"
