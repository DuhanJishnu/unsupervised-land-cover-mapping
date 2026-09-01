"""
=============================================================================
07 — Standard Transformer Autoencoder (Ablation Baseline B)
=============================================================================

Implements a vanilla Transformer Autoencoder with the SAME architecture as
HyperAttnRes (same 3D CNN stem, same tokenization, same masked spectral
reconstruction objective) but WITHOUT Block Attention Residuals.

Standard residual connections are used instead:  h_l = h_{l-1} + f_l(h_{l-1})

This is the critical ablation that isolates the contribution of AttnRes
specifically — without it, reviewers will ask "is the improvement from
AttnRes or just from using a Transformer at all?"

Also includes record_norms=True so we can plot the PreNorm dilution figure
and show that output magnitudes grow monotonically here (vs bounded in HyperAttnRes).

"""

import os
import yaml
import torch
import torch.nn as nn

# Import stem + tokenization from the model file (shared architecture)
import importlib.util
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "hyperattnres_model", os.path.join(SCRIPT_DIR, "06_hyperattnres_model.py")
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)

SpectralSpatial3DStem = _mod.SpectralSpatial3DStem
TokenizationHead      = _mod.TokenizationHead
CFG                   = _mod.CFG

# ──────────────────────────────────────────────────────────────────────────────

class StandardTransformerLayer(nn.Module):
    """
    One standard transformer layer (PreNorm + standard residual).
    Pre-LayerNorm + MHA + residual, then Pre-LayerNorm + FFN + residual.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int = 4):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.0, batch_first=True
        )
        ffn_hidden = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard PreNorm residual: x = x + Attn(Norm(x))
        normed = self.attn_norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Standard PreNorm residual: x = x + FFN(Norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StandardTransformerEncoder(nn.Module):
    """
    Encoder with standard PreNorm residuals (no AttnRes).

    Identical to HyperAttnResEncoder in terms of:
      - 3D CNN stem (same architecture)
      - AdaptiveAvgPool3d to 16 spectral tokens (same)
      - 12 transformer layers total (4 blocks × 3 layers, same layer count)
      - Mean-pool + LayerNorm + Linear(128→64) aggregation (same)

    Only difference: STANDARD residual connections instead of Block AttnRes.

    record_norms=True records ‖h_l‖ at each layer for the PreNorm dilution plot.
    """

    def __init__(
        self,
        in_bands: int,
        d_model: int = 128,
        n_layers: int = 12,         # Total layers (=N_blocks * S; keep same as HyperAttnRes)
        n_heads: int = 4,
        ffn_ratio: int = 4,
        spectral_tokens: int = 16,
        embedding_dim: int = 64,
    ):
        super().__init__()

        # Stage 1+2: same as HyperAttnRes (shared implementation)
        self.stem     = SpectralSpatial3DStem()
        self.tokenize = TokenizationHead(d_model=d_model, spectral_tokens=spectral_tokens)

        # Stage 3: N vanilla transformer layers (no AttnRes)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])

        # Stage 4: Aggregation
        self.aggregation = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, x: torch.Tensor, record_norms: bool = False):
        """
        x: (B, bands, 7, 7)
        Returns: (embedding, norm_list)
        """
        x = x.unsqueeze(1)             # (B, 1, bands, 7, 7)
        x = self.stem(x)               # (B, 64, spectral_raw, 5, 5)
        h = self.tokenize(x)           # (B, T, d_model)

        norm_list = [] if record_norms else None

        for layer in self.layers:
            h = layer(h)
            if record_norms:
                norm_list.append(h.norm(dim=-1).mean().item())

        pooled = h.mean(dim=1)         # (B, d_model)
        embedding = self.aggregation(pooled)  # (B, embedding_dim)

        return embedding, norm_list


class StandardTransformerAE(nn.Module):
    """
    Full Standard Transformer AE with Masked Spectral Reconstruction.
    Identical training objective to HyperAttnResAE (center-pixel MSE on masked bands).
    """

    def __init__(
        self,
        in_bands: int,
        d_model: int = 128,
        n_layers: int = 12,
        n_heads: int = 4,
        ffn_ratio: int = 4,
        spectral_tokens: int = 16,
        embedding_dim: int = 64,
        mask_ratio: float = 0.30,
    ):
        super().__init__()
        self.in_bands   = in_bands
        self.mask_ratio = mask_ratio

        self.encoder = StandardTransformerEncoder(
            in_bands=in_bands,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ffn_ratio=ffn_ratio,
            spectral_tokens=spectral_tokens,
            embedding_dim=embedding_dim,
        )

        self.decoder = nn.Linear(embedding_dim, in_bands)

    def encode(self, x: torch.Tensor, record_norms: bool = False):
        """Clean inference encoding (no masking)."""
        return self.encoder(x, record_norms=record_norms)

    def forward(self, x: torch.Tensor, mask_ratio: float = None):
        """Training forward with masked spectral reconstruction."""
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, bands, H, W = x.shape
        center_h, center_w = H // 2, W // 2
        target = x[:, :, center_h, center_w]   # (B, bands)

        # Random band mask
        n_mask = max(1, int(bands * mask_ratio))
        noise = torch.rand(B, bands, device=x.device)
        mask_ids = noise.argsort(dim=1)[:, :n_mask]

        x_masked = x.clone()
        for b in range(B):
            x_masked[b, mask_ids[b], :, :] = 0.0

        embedding, _ = self.encoder(x_masked, record_norms=False)
        pred = self.decoder(embedding)

        # MSE on masked bands only
        mask_matrix = torch.zeros(B, bands, device=x.device)
        for b in range(B):
            mask_matrix[b, mask_ids[b]] = 1.0

        diff = (pred - target) ** 2
        loss = (diff * mask_matrix).sum() / mask_matrix.sum().clamp(min=1)

        return loss, embedding


def build_standard_transformer_ae(in_bands: int) -> StandardTransformerAE:
    """Build from global config — ensures same layer count as HyperAttnRes."""
    cfg = CFG
    n_layers = cfg["n_blocks"] * cfg["n_layers_per_block"]  # 4×3=12 total layers
    return StandardTransformerAE(
        in_bands=in_bands,
        d_model=cfg["d_model"],
        n_layers=n_layers,
        n_heads=cfg["n_heads"],
        ffn_ratio=cfg["ffn_ratio"],
        spectral_tokens=cfg["spectral_tokens"],
        embedding_dim=cfg["embedding_dim"],
        mask_ratio=cfg["mask_ratio"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Standard Transformer AE — Forward-Pass Shape Check")
    print("=" * 60)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")

    for dataset, bands in [("Indian Pines", 200), ("Pavia University", 103)]:
        print(f"\n  [{dataset}]  bands={bands}")
        model = build_standard_transformer_ae(in_bands=bands).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {n_params:,}")

        x = torch.randn(8, bands, 7, 7, device=DEVICE)
        loss, emb = model(x)
        print(f"    Training  — loss: {loss.item():.4f}, emb: {emb.shape}")
        assert emb.shape == (8, 64)

        emb2, norms = model.encode(x, record_norms=True)
        print(f"    Inference — emb: {emb2.shape}, norms per layer: {len(norms)}")
        assert emb2.shape == (8, 64)
        assert len(norms) == CFG["n_blocks"] * CFG["n_layers_per_block"]
        print(f"    ✓ All assertions passed")

    print("\n  ✓ Sanity check complete!")
