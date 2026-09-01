"""Initial clustering-aligned spectral-spatial representation model.

This is the first implementation milestone, not a frozen paper architecture.
Every objective term is exposed separately to support controlled ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralSpatialEncoder(nn.Module):
    """Fuse a center-spectrum branch with a lightweight spatial branch."""

    def __init__(self, in_bands: int, embedding_dim: int = 64):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(in_bands, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),
            nn.Conv2d(64, 96, kernel_size=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(64 + 96, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError("patches must have shape (batch, bands, height, width)")
        center = patches[:, :, patches.shape[2] // 2, patches.shape[3] // 2]
        spectral = self.spectral(center.unsqueeze(1)).flatten(1)
        spatial = self.spatial(patches).flatten(1)
        return F.normalize(self.fusion(torch.cat((spectral, spatial), dim=1)), dim=1)


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    reconstruction: torch.Tensor
    prototype_logits: torch.Tensor


class ClusteringAlignedModel(nn.Module):
    """Encoder with masked-spectrum decoder and balanced prototype head."""

    def __init__(self, in_bands: int, embedding_dim: int = 64, n_prototypes: int = 16):
        super().__init__()
        self.encoder = SpectralSpatialEncoder(in_bands, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, in_bands)
        self.prototypes = nn.Linear(embedding_dim, n_prototypes, bias=False)

    def forward(self, patches: torch.Tensor) -> ModelOutput:
        embedding = self.encoder(patches)
        prototype_weight = F.normalize(self.prototypes.weight, dim=1)
        return ModelOutput(
            embedding=embedding,
            reconstruction=self.decoder(embedding),
            prototype_logits=F.linear(embedding, prototype_weight),
        )


def random_band_mask(
    batch_size: int,
    bands: int,
    ratio: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate an exact-size random band mask for each sample."""
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between zero and one")
    count = max(1, round(bands * ratio))
    scores = torch.rand(batch_size, bands, device=device)
    indices = scores.topk(count, dim=1, largest=False).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(1, indices, True)


def apply_band_mask(patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != patches.shape[:2]:
        raise ValueError("mask must have shape (batch, bands)")
    return patches.masked_fill(mask[:, :, None, None], 0.0)


def masked_reconstruction_loss(
    prediction: torch.Tensor,
    target_spectrum: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(prediction[mask], target_spectrum[mask])


def spectral_angle_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(prediction, target, dim=1, eps=1e-8).clamp(-1 + 1e-6, 1 - 1e-6)
    return torch.acos(cosine).mean()


def view_consistency_loss(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (1.0 - F.cosine_similarity(first, second, dim=1)).mean()


@torch.no_grad()
def sinkhorn_balanced_assignments(
    logits: torch.Tensor,
    temperature: float = 0.05,
    iterations: int = 10,
) -> torch.Tensor:
    """Return sharpened assignments with balanced prototype marginals.

    The alternating row/column normalization follows the equipartition step
    used by online clustering methods. Subtracting the global maximum before
    exponentiation makes the operation stable for low temperatures.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("logits must have shape (nonempty batch, at least 2 prototypes)")
    if temperature <= 0 or iterations < 1:
        raise ValueError("temperature and iterations must be positive")

    batch_size, n_prototypes = logits.shape
    assignments = torch.exp((logits - logits.max()) / temperature).t()
    assignments /= assignments.sum().clamp_min(1e-12)
    for _ in range(iterations):
        assignments /= assignments.sum(dim=1, keepdim=True).clamp_min(1e-12)
        assignments /= n_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True).clamp_min(1e-12)
        assignments /= batch_size
    assignments *= batch_size
    return assignments.t()


def balanced_prototype_loss(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    temperature: float = 0.2,
    target_temperature: float = 0.05,
    sinkhorn_iterations: int = 10,
) -> torch.Tensor:
    """Swapped prediction of sharpened, batch-balanced prototype targets."""
    first_target = sinkhorn_balanced_assignments(
        first_logits.detach(), target_temperature, sinkhorn_iterations
    )
    second_target = sinkhorn_balanced_assignments(
        second_logits.detach(), target_temperature, sinkhorn_iterations
    )
    return -0.5 * (
        (first_target * F.log_softmax(second_logits / temperature, dim=1)).sum(1).mean()
        + (second_target * F.log_softmax(first_logits / temperature, dim=1)).sum(1).mean()
    )


@torch.no_grad()
def prototype_diagnostics(
    first_logits: torch.Tensor, second_logits: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Measure hard prototype usage without affecting optimization."""
    n_prototypes = first_logits.shape[1]
    assignments = torch.cat((first_logits.argmax(1), second_logits.argmax(1)))
    proportions = torch.bincount(assignments, minlength=n_prototypes).float()
    proportions /= proportions.sum().clamp_min(1.0)
    entropy = -(proportions * proportions.clamp_min(1e-12).log()).sum()
    entropy /= torch.log(proportions.new_tensor(float(n_prototypes)))
    return {
        "prototype_usage_entropy": entropy,
        "prototype_max_share": proportions.max(),
        "prototype_active_fraction": (proportions > 0).float().mean(),
    }


def spatial_graph_loss(
    embeddings: torch.Tensor,
    edges: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize disagreement along precomputed boundary-aware graph edges."""
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (n_edges, 2)")
    similarity = F.cosine_similarity(embeddings[edges[:, 0]], embeddings[edges[:, 1]], dim=1)
    penalties = 1.0 - similarity
    if weights is not None:
        penalties = penalties * weights
        return penalties.sum() / weights.sum().clamp_min(1e-8)
    return penalties.mean()


@dataclass(frozen=True)
class LossWeights:
    masked: float = 1.0
    spectral_angle: float = 0.1
    view: float = 1.0
    prototype: float = 1.0
    spatial: float = 0.2


def clustering_aligned_loss(
    first: ModelOutput,
    second: ModelOutput,
    target_spectrum: torch.Tensor,
    band_mask: torch.Tensor,
    edges: torch.Tensor | None = None,
    edge_weights: torch.Tensor | None = None,
    weights: LossWeights = LossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = {
        "masked": masked_reconstruction_loss(first.reconstruction, target_spectrum, band_mask),
        "spectral_angle": spectral_angle_loss(first.reconstruction, target_spectrum),
        "view": view_consistency_loss(first.embedding, second.embedding),
        "prototype": balanced_prototype_loss(first.prototype_logits, second.prototype_logits),
    }
    if edges is not None and len(edges):
        terms["spatial"] = spatial_graph_loss(first.embedding, edges, edge_weights)
    else:
        terms["spatial"] = first.embedding.new_zeros(())
    total = (
        weights.masked * terms["masked"]
        + weights.spectral_angle * terms["spectral_angle"]
        + weights.view * terms["view"]
        + weights.prototype * terms["prototype"]
        + weights.spatial * terms["spatial"]
    )
    return total, terms
