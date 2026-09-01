"""Training and evaluation pipeline for clustering-aligned embeddings."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from .artifacts import RunManifest, file_sha256, git_commit, source_tree_sha256
from .augmentations import spectral_spatial_view
from .config import DATASETS, PROJECT_ROOT, ExperimentConfig
from .data import Scene, normalize_scene
from .evaluation import evaluate_clustering, semantic_diagnostics
from .models import (
    ClusteringAlignedModel,
    LossWeights,
    apply_band_mask,
    clustering_aligned_loss,
    random_band_mask,
    prototype_diagnostics,
)
from .spatial import local_spatial_graph
from .torch_data import ScenePatchDataset, SpatialTileBatchSampler


ABLATIONS: dict[str, LossWeights] = {
    "full": LossWeights(),
    "reconstruction": LossWeights(
        masked=1.0,
        spectral_angle=0.0,
        view=0.0,
        prototype=0.0,
        spatial=0.0,
    ),
    "no_spectral_angle": LossWeights(spectral_angle=0.0),
    "no_view": LossWeights(view=0.0),
    "no_prototype": LossWeights(prototype=0.0),
    "no_spatial": LossWeights(spatial=0.0),
}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    scheduler_epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-5
    mask_ratio: float = 0.30
    tile_size: int = 32
    spatial_group_size: int = 32
    connectivity: int = 4
    graph_temperature: float = 0.10
    minimum_edge_weight: float = 0.05
    prototype_multiplier: int = 1
    prototype_stop_entropy: float = 0.0
    early_stop_usage_entropy: float = 0.0
    evaluation_every: int = 0
    num_workers: int = 0
    checkpoint_every: int = 10
    ablation: str = "full"
    device: str = "auto"

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("epochs must be positive and batch_size at least 2")
        if self.scheduler_epochs < self.epochs:
            raise ValueError("scheduler_epochs must be at least epochs")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must be between zero and one")
        if self.ablation not in ABLATIONS:
            raise ValueError(f"unknown ablation {self.ablation!r}; choose from {sorted(ABLATIONS)}")
        if self.connectivity not in {4, 8}:
            raise ValueError("connectivity must be 4 or 8")
        if self.spatial_group_size < 2 or self.spatial_group_size > self.batch_size:
            raise ValueError("spatial_group_size must be between 2 and batch_size")
        if self.graph_temperature <= 0 or not 0 <= self.minimum_edge_weight <= 1:
            raise ValueError("invalid graph weighting settings")
        if self.prototype_multiplier < 1:
            raise ValueError("prototype_multiplier must be at least one")
        if not 0 <= self.prototype_stop_entropy <= 1:
            raise ValueError("prototype_stop_entropy must be between zero and one")
        if not 0 <= self.early_stop_usage_entropy <= 1:
            raise ValueError("early_stop_usage_entropy must be between zero and one")
        if self.evaluation_every < 0:
            raise ValueError("evaluation_every cannot be negative")
        if self.checkpoint_every < 1 or self.num_workers < 0:
            raise ValueError("invalid checkpoint/worker settings")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        result = asdict(self)
        result["loss_weights"] = asdict(ABLATIONS[self.ablation])
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    history: list[dict[str, float]],
    experiment: ExperimentConfig,
    training: TrainingConfig,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "experiment_config": experiment.to_dict(),
        "training_config": training.to_dict(),
    }


def extract_embeddings(
    model: ClusteringAlignedModel,
    dataset: ScenePatchDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            patches = batch["patch"].to(device, non_blocking=True)
            chunks.append(model.encoder(patches).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def semantic_epoch_probe(
    model: ClusteringAlignedModel,
    dataset: ScenePatchDataset,
    ground_truth: np.ndarray,
    n_clusters: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> dict[str, float]:
    """Evaluate one checkpoint for development-only convergence diagnostics."""
    embeddings = extract_embeddings(model, dataset, device, batch_size, num_workers)
    labels = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit_predict(embeddings)
    truth = ground_truth[dataset.coordinates[:, 0], dataset.coordinates[:, 1]]
    metrics = evaluate_clustering(truth, labels, labeled_mask=truth > 0)
    names = ("ari", "nmi", "accuracy", "macro_f1", "miou", "cluster_entropy")
    return {f"probe_{name}": float(metrics[name]) for name in names}


def train_and_evaluate(
    scene: Scene,
    experiment: ExperimentConfig,
    training: TrainingConfig,
    run_dir: str | Path,
    *,
    data_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train without labels, with optional development-only checkpoint probes."""
    experiment.validate()
    training.validate()
    seed_everything(experiment.seed)
    device = resolve_device(training.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)

    normalized = normalize_scene(scene, experiment.normalization)
    dataset = ScenePatchDataset(normalized, patch_size=experiment.patch_size)
    sampler = SpatialTileBatchSampler(
        dataset.coordinates,
        batch_size=training.batch_size,
        tile_size=training.tile_size,
        spatial_group_size=training.spatial_group_size,
        seed=experiment.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = ClusteringAlignedModel(
        in_bands=normalized.cube.shape[2],
        embedding_dim=experiment.embedding_dim,
        n_prototypes=experiment.resolved_clusters() * training.prototype_multiplier,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training.scheduler_epochs, eta_min=training.learning_rate * 0.01
    )
    amp_enabled = device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    loss_weights = ABLATIONS[training.ablation]
    start_epoch = 1
    history: list[dict[str, float]] = []
    prototype_stopped = False

    if resume_checkpoint is not None:
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        _validate_resume_configuration(checkpoint, experiment, training)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        if training.prototype_stop_entropy:
            prototype_stopped = any(
                record.get("prototype_usage_entropy", 0.0)
                >= training.prototype_stop_entropy
                for record in history
            )

    checkpoint_path = output / "checkpoint.pt"
    training_started = time.perf_counter()
    periodic_evaluation_seconds = 0.0
    for epoch in range(start_epoch, training.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss_weights = replace(
            loss_weights,
            prototype=0.0 if prototype_stopped else loss_weights.prototype,
        )
        totals = {
            name: 0.0
            for name in (
                "total",
                "masked",
                "spectral_angle",
                "view",
                "prototype",
                "spatial",
                "prototype_usage_entropy",
                "prototype_max_share",
                "prototype_active_fraction",
            )
        }
        samples_seen = 0

        for batch in loader:
            patches = batch["patch"].to(device, non_blocking=True)
            coordinates = batch["coordinate"]
            center = patches[:, :, patches.shape[2] // 2, patches.shape[3] // 2]
            mask = random_band_mask(len(patches), patches.shape[1], training.mask_ratio, device)
            first_view = apply_band_mask(spectral_spatial_view(patches), mask)
            second_view = spectral_spatial_view(patches)
            edges, edge_weights = local_spatial_graph(
                coordinates,
                center,
                connectivity=training.connectivity,
                spectral_temperature=training.graph_temperature,
                minimum_weight=training.minimum_edge_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                first = model(first_view)
                second = model(second_view)
                loss, terms = clustering_aligned_loss(
                    first,
                    second,
                    center,
                    mask,
                    edges,
                    edge_weights,
                    epoch_loss_weights,
                )
                diagnostics = prototype_diagnostics(
                    first.prototype_logits, second.prototype_logits
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            batch_size = len(patches)
            samples_seen += batch_size
            totals["total"] += float(loss.detach()) * batch_size
            for name, value in terms.items():
                totals[name] += float(value.detach()) * batch_size
            for name, value in diagnostics.items():
                totals[name] += float(value) * batch_size

        scheduler.step()
        epoch_record = {name: value / max(samples_seen, 1) for name, value in totals.items()}
        epoch_record.update(
            {
                "epoch": float(epoch),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "prototype_weight": float(epoch_loss_weights.prototype),
            }
        )
        if (
            training.prototype_stop_entropy
            and epoch_record["prototype_usage_entropy"] >= training.prototype_stop_entropy
        ):
            prototype_stopped = True
        if training.evaluation_every and (
            epoch % training.evaluation_every == 0 or epoch == training.epochs
        ):
            probe_started = time.perf_counter()
            epoch_record.update(
                semantic_epoch_probe(
                    model,
                    dataset,
                    normalized.ground_truth,
                    experiment.resolved_clusters(),
                    device,
                    training.batch_size,
                    training.num_workers,
                    experiment.seed,
                )
            )
            periodic_evaluation_seconds += time.perf_counter() - probe_started
        history.append(epoch_record)
        should_early_stop = bool(
            training.early_stop_usage_entropy
            and epoch_record["prototype_usage_entropy"]
            >= training.early_stop_usage_entropy
        )
        probe_text = ""
        if "probe_ari" in epoch_record:
            probe_text = (
                f" ari={epoch_record['probe_ari']:.3f} "
                f"macro_f1={epoch_record['probe_macro_f1']:.3f}"
            )
        print(
            f"epoch={epoch:03d} total={epoch_record['total']:.6f} "
            f"masked={epoch_record['masked']:.6f} proto={epoch_record['prototype']:.6f} "
            f"spatial={epoch_record['spatial']:.6f} "
            f"usage={epoch_record['prototype_usage_entropy']:.3f} "
            f"proto_w={epoch_record['prototype_weight']:.1f}{probe_text}"
        )
        if epoch % training.checkpoint_every == 0 or epoch == training.epochs or should_early_stop:
            payload = _checkpoint_payload(
                epoch, model, optimizer, scheduler, scaler, history, experiment, training
            )
            _atomic_torch_save(payload, checkpoint_path)
            if training.evaluation_every:
                _atomic_torch_save(payload, output / f"checkpoint_epoch_{epoch:03d}.pt")
        if should_early_stop:
            print(
                f"early_stop epoch={epoch:03d} reason=prototype_usage_entropy "
                f"threshold={training.early_stop_usage_entropy:.3f}"
            )
            break

    if not checkpoint_path.exists():
        completed_epoch = min(training.epochs, max(start_epoch - 1, 0))
        _atomic_torch_save(
            _checkpoint_payload(
                completed_epoch, model, optimizer, scheduler, scaler, history, experiment, training
            ),
            checkpoint_path,
        )

    training_seconds = time.perf_counter() - training_started - periodic_evaluation_seconds
    model_path = output / "model.pt"
    _atomic_torch_save(
        {
            "model": model.state_dict(),
            "experiment_config": experiment.to_dict(),
            "training_config": training.to_dict(),
            "in_bands": normalized.cube.shape[2],
        },
        model_path,
    )
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    stats = normalized.normalization
    normalization_path = output / "normalization.npz"
    np.savez(
        normalization_path,
        method=np.array(stats.method),
        location=stats.location,
        scale=stats.scale,
        lower=np.array([]) if stats.lower is None else stats.lower,
        upper=np.array([]) if stats.upper is None else stats.upper,
    )

    inference_started = time.perf_counter()
    embeddings = extract_embeddings(
        model, dataset, device, training.batch_size, training.num_workers
    )
    cluster_labels = KMeans(
        n_clusters=experiment.resolved_clusters(),
        n_init=20,
        random_state=experiment.seed,
    ).fit_predict(embeddings)
    inference_seconds = time.perf_counter() - inference_started
    cluster_map = np.full(normalized.ground_truth.shape, -1, dtype=np.int32)
    cluster_map[dataset.coordinates[:, 0], dataset.coordinates[:, 1]] = cluster_labels
    embeddings_path = output / "embeddings.npy"
    cluster_map_path = output / "cluster_map.npy"
    coordinates_path = output / "coordinates.npy"
    np.save(embeddings_path, embeddings)
    np.save(cluster_map_path, cluster_map)
    np.save(coordinates_path, dataset.coordinates)

    full_features = np.zeros((normalized.ground_truth.size, embeddings.shape[1]), dtype=np.float32)
    full_features[normalized.valid_mask.ravel()] = embeddings
    metrics = evaluate_clustering(
        normalized.ground_truth.ravel(),
        cluster_map.ravel(),
        features=full_features,
        labeled_mask=(normalized.ground_truth > 0).ravel() & normalized.valid_mask.ravel(),
        seed=experiment.seed,
    )
    diagnostics_path = output / "semantic_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(
            semantic_diagnostics(
                normalized.ground_truth.ravel(),
                cluster_map.ravel(),
                labeled_mask=(normalized.ground_truth > 0).ravel()
                & normalized.valid_mask.ravel(),
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    metrics.update(
        {
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
            "training_seconds": float(training_seconds),
            "periodic_evaluation_seconds": float(periodic_evaluation_seconds),
            "completed_epochs": int(history[-1]["epoch"] if history else 0),
            "inference_clustering_seconds": float(inference_seconds),
            "peak_device_memory_mb": (
                float(torch.cuda.max_memory_allocated(device) / 1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
    )

    spec = DATASETS[experiment.dataset]
    data_root = Path(data_dir)
    data_path, gt_path = data_root / spec.data_file, data_root / spec.gt_file
    manifest = RunManifest(
        run_id=output.name,
        experiment=f"clustering-aligned:{training.ablation}",
        config={"experiment": experiment.to_dict(), "training": training.to_dict()},
        seed=experiment.seed,
        dataset_files={str(data_path): file_sha256(data_path), str(gt_path): file_sha256(gt_path)},
        artifacts={
            "model": str(model_path),
            "checkpoint": str(checkpoint_path),
            "history": str(output / "history.json"),
            "normalization": str(normalization_path),
            "embeddings": str(embeddings_path),
            "coordinates": str(coordinates_path),
            "cluster_map": str(cluster_map_path),
            "semantic_diagnostics": str(diagnostics_path),
        },
        metrics=metrics,
        git_commit=git_commit(PROJECT_ROOT),
        source_tree_sha256=source_tree_sha256(PROJECT_ROOT),
    )
    manifest.write(output / "manifest.json")
    return {"run_id": output.name, "run_dir": str(output), "metrics": metrics}


def _validate_resume_configuration(
    checkpoint: dict[str, Any],
    experiment: ExperimentConfig,
    training: TrainingConfig,
) -> None:
    previous_experiment = checkpoint.get("experiment_config")
    if previous_experiment != experiment.to_dict():
        raise ValueError("resume checkpoint uses a different experiment configuration")
    previous_training = dict(checkpoint.get("training_config", {}))
    current_training = training.to_dict()
    # Extending total epochs and changing execution-only settings is safe.
    for key in ("epochs", "device", "num_workers", "checkpoint_every", "evaluation_every"):
        previous_training.pop(key, None)
        current_training.pop(key, None)
    if previous_training != current_training:
        raise ValueError("resume checkpoint uses incompatible training settings")


def aggregate_seed_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate numeric metrics across independent seeds."""
    if not results:
        raise ValueError("results cannot be empty")
    metric_names = sorted(set.intersection(*(set(item["metrics"]) for item in results)))
    summary: dict[str, dict[str, float | int]] = {}
    for name in metric_names:
        values = np.asarray([item["metrics"][name] for item in results], dtype=float)
        finite = values[np.isfinite(values)]
        if len(finite):
            summary[name] = {
                "mean": float(finite.mean()),
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
                "n": int(len(finite)),
            }
    return {"runs": results, "summary": summary}
