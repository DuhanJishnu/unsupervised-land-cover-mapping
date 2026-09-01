"""Boundary-aware graph construction for spatial minibatches."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def local_spatial_graph(
    coordinates: torch.Tensor,
    spectra: torch.Tensor,
    *,
    connectivity: int = 4,
    spectral_temperature: float = 0.1,
    minimum_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build local edges weighted by center-spectrum similarity.

    Only exact 4- or 8-connected neighbors present in the current batch are
    connected. This prevents arbitrary within-tile smoothing across distance.
    """
    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be 4 or 8")
    if spectral_temperature <= 0:
        raise ValueError("spectral_temperature must be positive")
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (batch, 2)")
    if spectra.ndim != 2 or len(spectra) != len(coordinates):
        raise ValueError("spectra must have shape (batch, bands)")

    offsets = [(0, 1), (1, 0)]
    if connectivity == 8:
        offsets.extend([(1, 1), (1, -1)])
    lookup = {
        (int(row), int(col)): index
        for index, (row, col) in enumerate(coordinates.detach().cpu().tolist())
    }
    pairs = []
    for index, (row, col) in enumerate(coordinates.detach().cpu().tolist()):
        for delta_row, delta_col in offsets:
            neighbor = lookup.get((int(row) + delta_row, int(col) + delta_col))
            if neighbor is not None:
                pairs.append((index, neighbor))
    if not pairs:
        return (
            torch.empty((0, 2), dtype=torch.long, device=spectra.device),
            torch.empty((0,), dtype=spectra.dtype, device=spectra.device),
        )

    edges = torch.tensor(pairs, dtype=torch.long, device=spectra.device)
    cosine = F.cosine_similarity(spectra[edges[:, 0]], spectra[edges[:, 1]], dim=1)
    weights = torch.exp(-(1.0 - cosine.clamp(-1.0, 1.0)) / spectral_temperature)
    keep = weights >= minimum_weight
    return edges[keep], weights[keep]
