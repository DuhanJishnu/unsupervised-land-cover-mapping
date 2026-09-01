"""Entry point for reproducible research baselines.

Example:
    python research_pipeline.py baseline --dataset pu --representation pca
"""

from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path

import numpy as np

from landcover.artifacts import RunManifest, file_sha256, git_commit, source_tree_sha256
from landcover.baselines import estimate_cluster_count, pca_reference_features, run_pixel_baseline
from landcover.config import DATASETS, DEFAULT_DATA_DIR, DEFAULT_RUNS_DIR, PROJECT_ROOT, ExperimentConfig
from landcover.data import load_benchmark
from landcover.evaluation import semantic_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline", help="run raw/PCA + KMeans/GMM")
    baseline.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    baseline.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    baseline.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    baseline.add_argument("--representation", choices=["raw", "pca"], default="pca")
    baseline.add_argument("--clusterer", choices=["kmeans", "gmm"], default="kmeans")
    baseline.add_argument("--normalization", choices=["robust", "zscore", "minmax"], default="robust")
    baseline.add_argument("--pca-components", type=int, default=30)
    baseline.add_argument("--seed", type=int, default=42)
    baseline_k = baseline.add_mutually_exclusive_group()
    baseline_k.add_argument("--n-clusters", type=int, help="explicit non-oracle cluster count")
    baseline_k.add_argument(
        "--k-range", type=int, nargs=2, metavar=("MIN", "MAX"), help="estimate k without labels"
    )

    train = sub.add_parser("train", help="train clustering-aligned embeddings")
    train.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_RUNS_DIR)
    train.add_argument("--normalization", choices=["robust", "zscore", "minmax"], default="robust")
    train.add_argument("--patch-size", type=int, default=7)
    train.add_argument("--embedding-dim", type=int, default=64)
    train_k = train.add_mutually_exclusive_group()
    train_k.add_argument("--n-clusters", type=int, help="explicit non-oracle prototype count")
    train_k.add_argument(
        "--k-range", type=int, nargs=2, metavar=("MIN", "MAX"), help="estimate k without labels"
    )
    train.add_argument("--k-pca-components", type=int, default=30)
    train.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument(
        "--scheduler-epochs",
        type=int,
        default=50,
        help="fixed cosine-schedule horizon, independent of early stopping epoch",
    )
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--mask-ratio", type=float, default=0.30)
    train.add_argument("--tile-size", type=int, default=32)
    train.add_argument(
        "--spatial-group-size",
        type=int,
        default=32,
        help="neighbor-preserving samples contributed by each tile per batch",
    )
    train.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    train.add_argument("--graph-temperature", type=float, default=0.10)
    train.add_argument("--minimum-edge-weight", type=float, default=0.05)
    train.add_argument(
        "--prototype-multiplier",
        type=int,
        default=1,
        help="internal prototypes per final cluster; values >1 enable overclustering",
    )
    train.add_argument(
        "--prototype-stop-entropy",
        type=float,
        default=0.0,
        help="disable prototype balancing after reaching this label-free usage entropy; zero disables",
    )
    train.add_argument(
        "--early-stop-usage-entropy",
        type=float,
        default=0.0,
        help="label-free early stopping threshold for normalized hard-prototype usage; zero disables",
    )
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--checkpoint-every", type=int, default=10)
    train.add_argument(
        "--evaluation-every",
        type=int,
        default=0,
        help="development-only semantic checkpoint probe interval; zero disables it",
    )
    train.add_argument(
        "--ablation",
        choices=["full", "reconstruction", "no_spectral_angle", "no_view", "no_prototype", "no_spatial"],
        default="full",
    )
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, mps, or a concrete device")
    train.add_argument("--resume", type=Path, help="checkpoint to resume; valid with one seed only")

    compare = sub.add_parser("compare", help="generate JSON, CSV, and Markdown run comparisons")
    compare.add_argument("--runs", type=Path, nargs="+", required=True)
    compare.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def run_baseline_command(args: argparse.Namespace) -> None:
    scene = load_benchmark(args.dataset, args.data_dir)
    protocol, resolved_k, k_details = resolve_cluster_protocol(
        args, scene, args.seed, args.pca_components
    )
    config = ExperimentConfig(
        dataset=args.dataset,
        seed=args.seed,
        pca_components=args.pca_components,
        normalization=args.normalization,
        cluster_count_protocol=protocol,
        n_clusters=resolved_k,
        k_estimation=k_details,
    )
    labels, embeddings, metrics = run_pixel_baseline(
        scene, config, args.representation, args.clusterer
    )

    run_id = f"{args.dataset}-{args.representation}-{args.clusterer}-{uuid.uuid4().hex[:8]}"
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    labels_path = run_dir / "cluster_map.npy"
    embeddings_path = run_dir / "embeddings.npy"
    np.save(labels_path, labels)
    np.save(embeddings_path, embeddings)
    diagnostics_path = run_dir / "semantic_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(semantic_diagnostics(scene.ground_truth, labels), indent=2, sort_keys=True) + "\n"
    )

    spec = DATASETS[args.dataset]
    data_path = args.data_dir / spec.data_file
    gt_path = args.data_dir / spec.gt_file
    manifest = RunManifest(
        run_id=run_id,
        experiment=f"{args.representation}+{args.clusterer}",
        config=config.to_dict(),
        seed=args.seed,
        dataset_files={
            str(data_path): file_sha256(data_path),
            str(gt_path): file_sha256(gt_path),
        },
        artifacts={
            "cluster_map": str(labels_path),
            "embeddings": str(embeddings_path),
            "semantic_diagnostics": str(diagnostics_path),
        },
        metrics=metrics,
        git_commit=git_commit(PROJECT_ROOT),
        source_tree_sha256=source_tree_sha256(PROJECT_ROOT),
    )
    manifest.write(run_dir / "manifest.json")
    print(json.dumps({"run_id": run_id, "metrics": metrics}, indent=2))


def run_train_command(args: argparse.Namespace) -> None:
    from landcover.training import TrainingConfig, aggregate_seed_results, train_and_evaluate

    if args.resume is not None and len(args.seeds) != 1:
        raise ValueError("--resume can only be used with one seed")
    scene = load_benchmark(args.dataset, args.data_dir)
    protocol, resolved_k, k_details = resolve_cluster_protocol(
        args, scene, args.seeds[0], args.k_pca_components
    )
    group_id = f"{args.dataset}-aligned-{args.ablation}-{uuid.uuid4().hex[:8]}"
    group_dir = args.output_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=False)
    results = []
    for seed in args.seeds:
        experiment = ExperimentConfig(
            dataset=args.dataset,
            seed=seed,
            patch_size=args.patch_size,
            embedding_dim=args.embedding_dim,
            normalization=args.normalization,
            cluster_count_protocol=protocol,
            n_clusters=resolved_k,
            k_estimation=k_details,
        )
        training = TrainingConfig(
            epochs=args.epochs,
            scheduler_epochs=args.scheduler_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            mask_ratio=args.mask_ratio,
            tile_size=args.tile_size,
            spatial_group_size=args.spatial_group_size,
            connectivity=args.connectivity,
            graph_temperature=args.graph_temperature,
            minimum_edge_weight=args.minimum_edge_weight,
            prototype_multiplier=args.prototype_multiplier,
            prototype_stop_entropy=args.prototype_stop_entropy,
            early_stop_usage_entropy=args.early_stop_usage_entropy,
            num_workers=args.num_workers,
            checkpoint_every=args.checkpoint_every,
            evaluation_every=args.evaluation_every,
            ablation=args.ablation,
            device=args.device,
        )
        run_dir = group_dir / f"seed-{seed}"
        print(f"\n=== Training {group_id}, seed={seed} ===")
        results.append(
            train_and_evaluate(
                scene,
                experiment,
                training,
                run_dir,
                data_dir=args.data_dir,
                resume_checkpoint=args.resume,
            )
        )

    aggregate = aggregate_seed_results(results)
    (group_dir / "summary.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    with (group_dir / "metrics.csv").open("w", newline="") as stream:
        metric_names = sorted(set().union(*(result["metrics"] for result in results)))
        writer = csv.DictWriter(stream, fieldnames=["run_id", *metric_names])
        writer.writeheader()
        for result in results:
            writer.writerow({"run_id": result["run_id"], **result["metrics"]})
    print(json.dumps({"group_id": group_id, "summary": aggregate["summary"]}, indent=2))


def resolve_cluster_protocol(
    args: argparse.Namespace,
    scene,
    seed: int,
    pca_components: int,
) -> tuple[str, int | None, dict | None]:
    if args.k_range is not None:
        _, features = pca_reference_features(
            scene, args.normalization, pca_components, seed
        )
        selected, details = estimate_cluster_count(
            features, args.k_range[0], args.k_range[1], seed=seed
        )
        print(f"Estimated cluster count: k={selected}")
        return "estimated", selected, details
    if args.n_clusters is not None:
        return "explicit", args.n_clusters, None
    return "oracle", None, None


def main() -> None:
    args = parse_args()
    if args.command == "baseline":
        run_baseline_command(args)
    elif args.command == "train":
        run_train_command(args)
    elif args.command == "compare":
        from landcover.reporting import write_comparison

        rows = write_comparison(args.runs, args.output_dir)
        print(json.dumps({"output_dir": str(args.output_dir), "runs": len(rows)}, indent=2))
    else:
        raise RuntimeError(args.command)


if __name__ == "__main__":
    main()
