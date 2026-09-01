"""Machine-readable and paper-ready comparison tables for completed runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat

from .evaluation import semantic_diagnostics


PRIMARY_METRICS = (
    "ari",
    "nmi",
    "accuracy",
    "macro_f1",
    "miou",
    "cluster_entropy",
    "silhouette",
    "davies_bouldin",
    "training_seconds",
)
DIAGNOSTICS = (
    "prototype_usage_entropy",
    "prototype_max_share",
    "prototype_active_fraction",
)


def _load_or_compute_semantic_diagnostics(
    manifest: dict[str, Any], manifest_path: Path
) -> dict[str, Any] | None:
    recorded = manifest.get("artifacts", {}).get("semantic_diagnostics")
    if recorded and Path(recorded).exists():
        return json.loads(Path(recorded).read_text())
    cluster_value = manifest.get("artifacts", {}).get("cluster_map")
    gt_values = [path for path in manifest.get("dataset_files", {}) if "gt" in Path(path).stem.lower()]
    if not cluster_value or len(gt_values) != 1:
        return None
    cluster_path = Path(cluster_value)
    if not cluster_path.is_absolute():
        cluster_path = manifest_path.parent / cluster_path
    gt_path = Path(gt_values[0])
    if not cluster_path.exists() or not gt_path.exists():
        return None
    arrays = [value for key, value in loadmat(gt_path).items() if not key.startswith("__")]
    if len(arrays) != 1:
        return None
    return semantic_diagnostics(np.asarray(arrays[0]), np.load(cluster_path))


def _mean_metrics(summary: dict[str, Any]) -> dict[str, float | int | None]:
    return {
        name: values.get("mean")
        for name, values in summary.get("summary", {}).items()
        if isinstance(values, dict) and "mean" in values
    }


def _training_row(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text())
    manifests = sorted(run_dir.glob("seed-*/manifest.json"))
    if not manifests:
        raise ValueError(f"no seed manifests found in {run_dir}")
    manifest = json.loads(manifests[0].read_text())
    experiment = manifest["config"]["experiment"]
    training = manifest["config"]["training"]
    prototype_multiplier = int(training.get("prototype_multiplier", 1))
    method = training["ablation"]
    if prototype_multiplier != 1:
        method = f"{method}:prototypes-{prototype_multiplier}x"
    if float(training.get("prototype_stop_entropy", 0.0)):
        method += f":prototype-off@{float(training['prototype_stop_entropy']):g}"
    if float(training.get("early_stop_usage_entropy", 0.0)):
        method += f":early-stop@{float(training['early_stop_usage_entropy']):g}"
    row: dict[str, Any] = {
        "run_id": run_dir.name,
        "dataset": experiment["dataset"],
        "method": method,
        "epochs": training["epochs"],
        "prototype_multiplier": prototype_multiplier,
        "seeds": len(manifests),
        "cluster_protocol": experiment["cluster_count_protocol"],
        **_mean_metrics(summary),
    }
    if len(manifests) == 1:
        row["semantic_diagnostics"] = _load_or_compute_semantic_diagnostics(
            manifest, manifests[0]
        )
    final_diagnostics: dict[str, list[float]] = {name: [] for name in DIAGNOSTICS}
    for manifest_path in manifests:
        history_path = manifest_path.parent / "history.json"
        history = json.loads(history_path.read_text())
        if not history:
            continue
        for name in DIAGNOSTICS:
            if name in history[-1]:
                final_diagnostics[name].append(float(history[-1][name]))
    for name, values in final_diagnostics.items():
        row[name] = sum(values) / len(values) if values else None
    return row


def _baseline_row(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {
        "run_id": run_dir.name,
        "dataset": manifest["config"]["dataset"],
        "method": manifest["experiment"],
        "epochs": None,
        "prototype_multiplier": None,
        "seeds": 1,
        "cluster_protocol": manifest["config"]["cluster_count_protocol"],
        **manifest["metrics"],
        **{name: None for name in DIAGNOSTICS},
        "semantic_diagnostics": _load_or_compute_semantic_diagnostics(manifest, manifest_path),
    }


def load_comparison_rows(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for value in run_dirs:
        run_dir = Path(value).resolve()
        if (run_dir / "summary.json").exists():
            rows.append(_training_row(run_dir))
        elif (run_dir / "manifest.json").exists():
            rows.append(_baseline_row(run_dir))
        else:
            raise ValueError(f"{run_dir} is not a baseline or training run directory")
    return rows


def _markdown(rows: list[dict[str, Any]]) -> str:
    headings = ("Method", "ARI", "NMI", "ACC", "Macro-F1", "mIoU", "Silhouette")
    keys = ("method", "ari", "nmi", "accuracy", "macro_f1", "miou", "silhouette")
    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] + ["---:"] * 6) + " |"]
    for row in rows:
        values = [str(row["method"])]
        values.extend(
            "—" if row.get(key) is None else f"{float(row[key]):.4f}" for key in keys[1:]
        )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _learning_curve_rows(run_dirs: Iterable[str | Path]) -> list[dict[str, Any]]:
    curves = []
    for value in run_dirs:
        run_dir = Path(value).resolve()
        for manifest_path in sorted(run_dir.glob("seed-*/manifest.json")):
            manifest = json.loads(manifest_path.read_text())
            training = manifest["config"]["training"]
            multiplier = int(training.get("prototype_multiplier", 1))
            method = training["ablation"]
            if multiplier != 1:
                method = f"{method}:prototypes-{multiplier}x"
            if float(training.get("prototype_stop_entropy", 0.0)):
                method += f":prototype-off@{float(training['prototype_stop_entropy']):g}"
            if float(training.get("early_stop_usage_entropy", 0.0)):
                method += f":early-stop@{float(training['early_stop_usage_entropy']):g}"
            history = json.loads((manifest_path.parent / "history.json").read_text())
            for record in history:
                curves.append(
                    {
                        "run_id": run_dir.name,
                        "method": method,
                        "seed": manifest.get("seed", manifest_path.parent.name.removeprefix("seed-")),
                        **record,
                    }
                )
    return curves


def write_comparison(run_dirs: Iterable[str | Path], output_dir: str | Path) -> list[dict[str, Any]]:
    run_dirs = list(run_dirs)
    rows = load_comparison_rows(run_dirs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    fieldnames = [
        "run_id",
        "dataset",
        "method",
        "epochs",
        "prototype_multiplier",
        "seeds",
        "cluster_protocol",
        *PRIMARY_METRICS,
        *DIAGNOSTICS,
    ]
    with (output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_class.csv").open("w", newline="") as stream:
        fieldnames = ["run_id", "method", "class_id", "support", "precision", "recall", "f1", "iou"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            diagnostics = row.get("semantic_diagnostics") or {}
            for per_class in diagnostics.get("per_class", []):
                writer.writerow(
                    {"run_id": row["run_id"], "method": row["method"], **per_class}
                )
    (output / "comparison.md").write_text(_markdown(rows))
    curves = _learning_curve_rows(run_dirs)
    curve_fields = sorted(set().union(*(row.keys() for row in curves))) if curves else []
    with (output / "learning_curves.csv").open("w", newline="") as stream:
        if curve_fields:
            writer = csv.DictWriter(stream, fieldnames=curve_fields)
            writer.writeheader()
            writer.writerows(curves)
    return rows
