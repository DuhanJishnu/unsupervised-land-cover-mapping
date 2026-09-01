"""Run manifests and JSON-safe artifact metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml"}
SOURCE_FILENAMES = {"requirements.txt", "requirements-lock.txt"}
IGNORED_SOURCE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "datasets",
    "datas",
    "models",
    "outputs",
    "processed_data",
}


def package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "scipy",
        "scikit-learn",
        "torch",
        "umap-learn",
        "hdbscan",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(project_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_sha256(project_root: str | Path) -> str:
    """Fingerprint executable source, including files not yet tracked by Git."""
    root = Path(project_root).resolve()
    digest = hashlib.sha256()
    paths = (
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in IGNORED_SOURCE_DIRS for part in path.relative_to(root).parts)
        and (path.suffix.lower() in SOURCE_SUFFIXES or path.name in SOURCE_FILENAMES)
    )
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


@dataclass
class RunManifest:
    run_id: str
    experiment: str
    config: dict[str, Any]
    seed: int
    dataset_files: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float | int | None] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = None
    source_tree_sha256: str | None = None
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    packages: dict[str, str] = field(default_factory=package_versions)

    def write(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
