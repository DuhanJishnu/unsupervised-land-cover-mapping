"""HSI-safe stochastic views for self-supervised representation learning."""

from __future__ import annotations

import torch


def spectral_spatial_view(
    patches: torch.Tensor,
    *,
    noise_std: float = 0.01,
    gain_std: float = 0.03,
    band_dropout: float = 0.05,
) -> torch.Tensor:
    """Apply mild spectral perturbations and geometry-preserving flips."""
    if patches.ndim != 4:
        raise ValueError("patches must have shape (batch, bands, height, width)")
    view = patches.clone()
    batch, bands = view.shape[:2]
    gain = 1.0 + gain_std * torch.randn(batch, 1, 1, 1, device=view.device)
    view = view * gain + noise_std * torch.randn_like(view)
    keep = torch.rand(batch, bands, 1, 1, device=view.device) >= band_dropout
    view = view * keep

    horizontal = torch.rand(batch, device=view.device) < 0.5
    vertical = torch.rand(batch, device=view.device) < 0.5
    view[horizontal] = view[horizontal].flip(-1)
    view[vertical] = view[vertical].flip(-2)
    return view

