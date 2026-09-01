"""Lazy PyTorch datasets that retain source coordinates and avoid patch dumps."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset

from .data import Scene


class ScenePatchDataset(Dataset):
    def __init__(
        self,
        scene: Scene,
        patch_size: int = 7,
        coordinates: np.ndarray | None = None,
        padding: str = "reflect",
    ):
        self.scene = scene
        self.patch_size = patch_size
        self.coordinates = (
            scene.coordinates(valid_only=True)
            if coordinates is None
            else np.asarray(coordinates, dtype=np.int32)
        )
        self.padding = padding
        margin = patch_size // 2
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        self._padded_cube = np.pad(
            scene.cube,
            ((margin, margin), (margin, margin), (0, 0)),
            mode=padding,
        )

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, col = self.coordinates[index]
        patch = self._padded_cube[
            int(row) : int(row) + self.patch_size,
            int(col) : int(col) + self.patch_size,
        ]
        patch = np.moveaxis(patch, -1, 0).astype(np.float32, copy=False)
        return {
            "patch": torch.from_numpy(np.ascontiguousarray(patch)),
            "coordinate": torch.tensor((int(row), int(col)), dtype=torch.int64),
        }


class SpatialTileBatchSampler(BatchSampler):
    """Shuffle spatial tiles while retaining local neighbors within batches.

    Every dataset index is emitted once per epoch. Tiles and samples within
    each tile are independently shuffled using `seed + epoch`.
    """

    def __init__(
        self,
        coordinates: np.ndarray,
        batch_size: int,
        tile_size: int = 32,
        spatial_group_size: int = 32,
        seed: int = 42,
        drop_last: bool = False,
    ):
        if batch_size < 2 or tile_size < 2 or spatial_group_size < 2:
            raise ValueError("batch_size, tile_size, and spatial_group_size must be at least 2")
        if spatial_group_size > batch_size:
            raise ValueError("spatial_group_size cannot exceed batch_size")
        self.coordinates = np.asarray(coordinates, dtype=np.int64)
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n_samples, 2)")
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.spatial_group_size = spatial_group_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        tile_ids = self.coordinates // tile_size
        self._tiles: list[np.ndarray] = []
        _, inverse = np.unique(tile_ids, axis=0, return_inverse=True)
        for tile_id in range(int(inverse.max()) + 1 if len(inverse) else 0):
            self._tiles.append(np.flatnonzero(inverse == tile_id))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        groups: list[list[int]] = []
        for tile_indices in self._tiles:
            tile_coords = self.coordinates[tile_indices]
            if self.epoch % 2:
                order = np.lexsort((tile_coords[:, 1], tile_coords[:, 0]))
            else:
                order = np.lexsort((tile_coords[:, 0], tile_coords[:, 1]))
            ordered = tile_indices[order]
            for start in range(0, len(ordered), self.spatial_group_size):
                groups.append(
                    [int(index) for index in ordered[start : start + self.spatial_group_size]]
                )
        rng.shuffle(groups)
        pending: list[int] = []
        for group in groups:
            offset = 0
            while offset < len(group):
                remaining = self.batch_size - len(pending)
                take = min(remaining, len(group) - offset)
                pending.extend(group[offset : offset + take])
                offset += take
                if len(pending) == self.batch_size:
                    yield pending
                    pending = []
        if pending and not self.drop_last:
            yield pending

    def __len__(self) -> int:
        count = len(self.coordinates)
        if self.drop_last:
            return count // self.batch_size
        return (count + self.batch_size - 1) // self.batch_size
