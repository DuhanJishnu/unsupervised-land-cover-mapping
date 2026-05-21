"""
=============================================================================
06 — HyperAttnRes Model Architecture
=============================================================================

Defines the HyperAttnRes architecture:

  Stage 1 : SpectralSpatial3DStem     — 3D CNN, input (B, 1, bands, 7, 7)
  Stage 2 : AdaptiveAvgPool + Tokenization — fixed 16 spectral tokens
  Stage 3 : HyperAttnResEncoder       — 4 blocks × 3 transformer layers with
                                        Block AttnRes (Kimi AttnRes paper, Fig 2)
  Stage 4 : Aggregation               — mean-pool → LayerNorm → Linear(128→64)
  Stage 5 : MaskedSpectralDecoder     — Linear(64 → bands), MSE on masked bands

Block AttnRes Op (eq. from §3.2 + Fig.2 of arXiv:2603.15031):
  - Stack completed block reps + current partial sum as keys/values V
  - Normalize K = RMSNorm(V)
  - Compute logits via learned pseudo-query w_l (shape [d]):
      logits = einsum('d, n b t d -> n b t', w, K)
  - Weighted sum:  h = einsum('n b t, n b t d -> b t d', softmax(logits, dim=0), V)
  - CRITICAL: pseudo-query w_l is initialized to ZERO (§5 of paper) so that
    initial attention weights are uniform — prevents training instability.

"""

import os
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Load config
# ──────────────────────────────────────────────────────────────────────────────

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hyperattnres_config.yaml")

def load_config():
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

CFG = load_config()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: 3D Spectral-Spatial CNN Stem
# ══════════════════════════════════════════════════════════════════════════════

class SpectralSpatial3DStem(nn.Module):
    """
    3D CNN stem that jointly processes spectral and spatial dimensions.

    Input  : (B, 1, bands, 7, 7)  — 1 "channel", bands as spectral depth
    Output : (B, 64, *, 5, 5)    — spatial reduced to 5×5, spectral compressed

    Three Conv3D layers with (spectral, H, W) kernels so that spectral and
    spatial correlations are captured jointly from the start.
    """

    def __init__(self):
        super().__init__()

        # Build from config
        in_ch   = [1] + CFG["stem_channels"][:-1]
        out_ch  = CFG["stem_channels"]
        kernels = CFG["stem_kernels"]
        strides = CFG["stem_strides"]

        layers = []
        for ic, oc, k, s in zip(in_ch, out_ch, kernels, strides):
            # Padding: keep spatial dim; for spectral don't pad so it reduces
            pad = (0, k[1] // 2, k[2] // 2)   # (spectral_pad=0, h_pad, w_pad)
            layers += [
                nn.Conv3d(ic, oc, kernel_size=k, stride=s, padding=pad),
                nn.BatchNorm3d(oc),
                nn.GELU(),
            ]
        self.stem = nn.Sequential(*layers)

        # After 3 convolutions the spatial 7×7 becomes 5×5 (valid convs with 3×3 kernels twice,
        # 3rd conv preserves spatial via padding).
        # Spectral dimension varies by dataset, so we fix it with adaptive pooling downstream.

    def forward(self, x):
        """
        x: (B, 1, bands, 7, 7)
        returns: (B, 64, reduced_spectral, 5, 5)
        """
        return self.stem(x)   # (B, 64, *, 5, 5)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Tokenization
# ══════════════════════════════════════════════════════════════════════════════

class TokenizationHead(nn.Module):
    """
    Converts 3D CNN output to a token sequence for the transformer.

    Steps:
      1. AdaptiveAvgPool3d → always (64, spectral_tokens, 5, 5) regardless of dataset
      2. Flatten spatial dims: (B, 64, T, 5, 5) → (B, T, 64*5*5=1600)
      3. Linear projection: 1600 → d_model
      4. Add learnable positional embedding: (1, T, d_model)
    """

    def __init__(self, d_model: int = 128, spectral_tokens: int = 16):
        super().__init__()
        self.spectral_tokens = spectral_tokens
        stem_out_channels = CFG["stem_channels"][-1]   # 64
        spatial_size = 5
        self.flat_dim = stem_out_channels * spatial_size * spatial_size  # 1600

        self.pool = nn.AdaptiveAvgPool3d(output_size=(spectral_tokens, spatial_size, spatial_size))
        self.proj = nn.Linear(self.flat_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, spectral_tokens, d_model))
        # Initialize positional embeddings with small variance
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x: (B, 64, spectral_raw, 5, 5)
        returns: (B, T, d_model)  where T = spectral_tokens
        """
        x = self.pool(x)                         # (B, 64, T, 5, 5)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)             # (B, T, 64, 5, 5)
        x = x.contiguous().view(B, T, -1)        # (B, T, 1600)
        x = self.proj(x)                         # (B, T, d_model)
        x = x + self.pos_embed                   # (B, T, d_model) + positional
        return x


# ══════════════════════════════════════════════════════════════════════════════
# Block AttnRes Operation (§3.2, Figure 2 of arXiv:2603.15031)
# ══════════════════════════════════════════════════════════════════════════════

class BlockAttnResOp(nn.Module):
    """
    A single AttnRes attention operation over the block history.

    Given: completed block representations b_0 ... b_{n-1}  (each: B×T×d)
           current intra-block partial sum (b_n^i)          (B×T×d)

    Computes:
        V      = stack(blocks + [partial_block])   # [N+1, B, T, d]
        K      = RMSNorm(V)
        logits = einsum('d, n b t d -> n b t', w, K)   # w = pseudo-query [d]
        h      = einsum('n b t, n b t d -> b t d', softmax(logits, dim=0), V)

    The pseudo-query w_l is a learnable parameter of shape [d], initialized to
    ZERO so that the initial attention weights are uniform (§5 requirement).

    NOTE: The "partial_block" for the very first layer in block 0 is the token
    embedding output from Stage 2. The embedding is always b_0 and is prepended
    so it is always available as a source.
    """

    def __init__(self, d_model: int):
        super().__init__()
        # Pseudo-query: one d-dimensional vector, initialized to zero
        self.w = nn.Parameter(torch.zeros(d_model))
        # RMSNorm for keys (prevents large-magnitude blocks from dominating)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, blocks: list, partial_block: torch.Tensor) -> torch.Tensor:
        """
        blocks       : list of N tensors, each (B, T, d)  — completed block reps
        partial_block: (B, T, d)  — current intra-block partial sum b_n^i

        Returns h: (B, T, d)  — the AttnRes-gated hidden state for this layer
        """
        # Stack value matrix: [N+1, B, T, d]
        V = torch.stack(blocks + [partial_block], dim=0)

        # Normalize keys (same as V here — keys and values are the block reps)
        K = self.norm(V)                                  # [N+1, B, T, d]

        # Compute attention logits with learned pseudo-query w (shape [d])
        # ‹w, K_n› over n dimension — one scalar per (block, batch, token)
        logits = torch.einsum('d, n b t d -> n b t', self.w, K)  # [N+1, B, T]

        # Softmax over the N+1 sources (depth dimension, dim=0)
        weights = logits.softmax(dim=0)                           # [N+1, B, T]

        # Weighted sum of values
        h = torch.einsum('n b t, n b t d -> b t d', weights, V)  # [B, T, d]

        return h


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3: A Single Block AttnRes Transformer Block
# ══════════════════════════════════════════════════════════════════════════════

class HyperAttnResTransformerLayer(nn.Module):
    """
    One transformer layer with two AttnRes operations (before Attn and before FFN).

    Per the forward pseudocode in Figure 2 of the paper:
      h = block_attn_res(blocks, partial_block, attn_proj, attn_norm)
      attn_out    = Attention(LayerNorm(h))
      partial_sum = partial_sum + attn_out          ← accumulates within block

      h = block_attn_res(blocks, partial_block, ffn_proj, ffn_norm)
      ffn_out     = FFN(LayerNorm(h))
      partial_sum = partial_sum + ffn_out

    CRITICAL (as flagged in the feedback):
      The layer's input 'h' is the AttnRes output, NOT the partial_sum directly.
      partial_sum is only used as the value V in the next AttnRes call.
      This is the key distinction from standard residuals.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_ratio: int = 4):
        super().__init__()

        # AttnRes operations — one before Attn, one before FFN
        self.attn_res_op = BlockAttnResOp(d_model)
        self.ffn_res_op  = BlockAttnResOp(d_model)

        # Layer norms (applied to AttnRes output before Attn/FFN)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm  = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=0.0, batch_first=True
        )

        # FFN: Linear → GELU → Linear
        ffn_hidden = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(
        self,
        blocks: list,
        partial_sum: torch.Tensor,
    ) -> torch.Tensor:
        """
        blocks      : list of completed block tensors, each (B, T, d)
        partial_sum : (B, T, d) — intra-block accumulated residual so far

        Returns updated partial_sum: (B, T, d)
        """
        # ── Attention sub-layer ──────────────────────────────────────────────
        # h comes from AttnRes over block history + current partial_sum
        h = self.attn_res_op(blocks, partial_sum)           # (B, T, d)
        h_normed = self.attn_norm(h)                        # (B, T, d)
        attn_out, _ = self.attn(h_normed, h_normed, h_normed)
        partial_sum = partial_sum + attn_out                # accumulate within block

        # ── FFN sub-layer ───────────────────────────────────────────────────
        # Apply AttnRes again using updated partial_sum
        h = self.ffn_res_op(blocks, partial_sum)            # (B, T, d)
        h_normed = self.ffn_norm(h)                         # (B, T, d)
        ffn_out = self.ffn(h_normed)
        partial_sum = partial_sum + ffn_out                 # accumulate

        return partial_sum


class HyperAttnResBlock(nn.Module):
    """
    A 'block' of S transformer layers sharing an AttnRes block boundary.

    At the START of a block: attend over all previously completed blocks.
    Within the block: partial_sum accumulates the layer outputs.
    At the END of a block: the partial_sum is saved as the new block representation.

    b_0 is the token embedding output from Stage 2 (passed as blocks[0] for
    consistency — it's always the first entry in the blocks list).
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, ffn_ratio: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            HyperAttnResTransformerLayer(d_model, n_heads, ffn_ratio)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        blocks: list,
        partial_sum: torch.Tensor,
        record_norms: bool = False,
    ):
        """
        blocks      : completed block representations b_0 ... b_{n-1}
        partial_sum : starts at zeros for the first layer of this block
                      (intra-block partial sum, b_n^0 = 0 per §3.2)

        Returns:
            blocks          : updated list (this block's rep appended)
            partial_sum     : zero'd out (start of next block)
            norm_list       : list of (B,T,d) partial_sum norms per layer (if record_norms)
        """
        norm_list = [] if record_norms else None

        for layer in self.layers:
            partial_sum = layer(blocks, partial_sum)
            if record_norms:
                # Record L2 norm of the partial_sum across d, averaged over B and T
                norm_list.append(partial_sum.norm(dim=-1).mean().item())

        # Completed block: save this block's representation
        blocks = blocks + [partial_sum]   # don't mutate the input list

        # Next block's partial_sum starts fresh from zero — intra-block reset
        new_partial_sum = torch.zeros_like(partial_sum)

        return blocks, new_partial_sum, norm_list


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3+4: Full HyperAttnRes Encoder
# ══════════════════════════════════════════════════════════════════════════════

class HyperAttnResEncoder(nn.Module):
    """
    Full encoder: 3D Stem → Tokenize → N × BlockAttnRes → Aggregation → 64D

    record_norms=True returns (embedding, norm_list) for the PreNorm dilution
    analysis figure. norm_list is a flat list of floats, one per transformer layer.
    """

    def __init__(
        self,
        in_bands: int,
        d_model: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        n_layers_per_block: int = 3,
        ffn_ratio: int = 4,
        spectral_tokens: int = 16,
        embedding_dim: int = 64,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_blocks = n_blocks

        # Stage 1: 3D CNN Stem
        self.stem = SpectralSpatial3DStem()

        # Stage 2: Tokenize
        self.tokenize = TokenizationHead(d_model=d_model, spectral_tokens=spectral_tokens)

        # Stage 3: N transformer blocks
        self.transformer_blocks = nn.ModuleList([
            HyperAttnResBlock(
                d_model=d_model, n_heads=n_heads,
                n_layers=n_layers_per_block, ffn_ratio=ffn_ratio
            )
            for _ in range(n_blocks)
        ])

        # Stage 4: Aggregation (mean-pool → LayerNorm → Linear)
        self.aggregation = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, embedding_dim),
        )

    def forward(self, x: torch.Tensor, record_norms: bool = False):
        """
        x: (B, bands, 7, 7)  — standard patch format from dataset

        Returns:
            embedding : (B, embedding_dim)
            norm_list : list[float] of per-layer output norms  (if record_norms)
                        None otherwise
        """
        B = x.shape[0]

        # Stage 1: Unsqueeze to add channel dim for 3D conv
        x = x.unsqueeze(1)              # (B, 1, bands, 7, 7)
        x = self.stem(x)                # (B, 64, spectral_raw, 5, 5)

        # Stage 2: Tokenize with fixed number of spectral tokens
        tokens = self.tokenize(x)       # (B, T, d_model)

        # b_0 is defined as the token embedding (always the first source)
        b0 = tokens                     # (B, T, d_model)
        blocks = [b0]

        # Stage 3: Apply N transformer blocks
        # partial_sum starts at zeros FOR THE FIRST LAYER of the first block
        # (b_n^0 := 0 per §3.2 of the paper)
        partial_sum = torch.zeros_like(tokens)   # (B, T, d_model)

        all_norms = [] if record_norms else None

        for block in self.transformer_blocks:
            blocks, partial_sum, block_norms = block(
                blocks, partial_sum, record_norms=record_norms
            )
            if record_norms and block_norms:
                all_norms.extend(block_norms)

        # After all blocks, partial_sum is the residual of the LAST block
        # The final output draws from all N completed block reps + last partial
        # We use the last block representation (blocks[-1]) as the output since
        # the last block's partial_sum was just reset to zero.
        # Correct approach: use the second-to-last entry in blocks (the last
        # *completed* block), which was appended by the last HyperAttnResBlock.
        final_block_rep = blocks[-1]    # (B, T, d_model) — last completed block

        # Stage 4: Aggregate tokens (mean pool over T)
        pooled = final_block_rep.mean(dim=1)   # (B, d_model)
        embedding = self.aggregation(pooled)   # (B, embedding_dim)

        return embedding, all_norms


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5: Masked Spectral Reconstruction Decoder
# ══════════════════════════════════════════════════════════════════════════════

class MaskedSpectralDecoder(nn.Module):
    """
    Simple Linear decoder for Masked Spectral Reconstruction.

    Predicts the CENTER PIXEL's spectral signature (bands,) from the 64D embedding.
    This forces the encoder to compress spectral identity rather than spatial context.

    Loss is computed ONLY on the masked bands (those set to zero in the input).
    """

    def __init__(self, embedding_dim: int = 64, out_bands: int = 176):
        super().__init__()
        self.decoder = nn.Linear(embedding_dim, out_bands)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, embedding_dim) → (B, bands)"""
        return self.decoder(z)


# ══════════════════════════════════════════════════════════════════════════════
# Full HyperAttnRes Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

class HyperAttnResAE(nn.Module):
    """
    Full HyperAttnRes Autoencoder with Masked Spectral Reconstruction.

    Forward pass (training):
      1. Sample mask_ratio fraction of bands; set them to 0 in input patches
      2. Extract center pixel from original patches (the reconstruction target)
      3. Encode masked patches → 64D embedding
      4. Decode → predicted center-pixel band values
      5. MSE loss on masked bands only

    Forward pass (inference / embedding extraction):
      Call encode() directly — no masking.
    """

    def __init__(
        self,
        in_bands: int,
        d_model: int = 128,
        n_blocks: int = 4,
        n_heads: int = 4,
        n_layers_per_block: int = 3,
        ffn_ratio: int = 4,
        spectral_tokens: int = 16,
        embedding_dim: int = 64,
        mask_ratio: float = 0.30,
    ):
        super().__init__()

        self.in_bands   = in_bands
        self.mask_ratio = mask_ratio

        self.encoder = HyperAttnResEncoder(
            in_bands=in_bands,
            d_model=d_model,
            n_blocks=n_blocks,
            n_heads=n_heads,
            n_layers_per_block=n_layers_per_block,
            ffn_ratio=ffn_ratio,
            spectral_tokens=spectral_tokens,
            embedding_dim=embedding_dim,
        )

        self.decoder = MaskedSpectralDecoder(
            embedding_dim=embedding_dim,
            out_bands=in_bands,
        )

    def encode(self, x: torch.Tensor, record_norms: bool = False):
        """
        Clean encoding (no masking) — used for embedding extraction.
        x: (B, bands, 7, 7)
        Returns: (embedding, norm_list)
        """
        return self.encoder(x, record_norms=record_norms)

    def forward(self, x: torch.Tensor, mask_ratio: float = None):
        """
        Training forward pass with masked spectral reconstruction.

        x: (B, bands, 7, 7)
        Returns: (loss, embedding) where loss is MSE on masked bands only.
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, bands, H, W = x.shape
        center_h, center_w = H // 2, W // 2

        # ── Extract center-pixel spectral signature (reconstruction target) ──
        target = x[:, :, center_h, center_w]    # (B, bands)

        # ── Sample random band indices to mask ──────────────────────────────
        n_mask = max(1, int(bands * mask_ratio))
        # Different mask per sample in the batch
        noise = torch.rand(B, bands, device=x.device)
        mask_ids = noise.argsort(dim=1)[:, :n_mask]   # (B, n_mask) band indices to mask

        # ── Apply mask: set masked bands to 0 in input ──────────────────────
        x_masked = x.clone()
        for b in range(B):
            x_masked[b, mask_ids[b], :, :] = 0.0

        # ── Encode masked input ──────────────────────────────────────────────
        embedding, _ = self.encoder(x_masked, record_norms=False)   # (B, 64)

        # ── Decode → predicted center pixel ─────────────────────────────────
        pred = self.decoder(embedding)       # (B, bands)

        # ── MSE loss on MASKED BANDS only ────────────────────────────────────
        # Build a mask tensor: 1 where band is masked, 0 otherwise
        mask_matrix = torch.zeros(B, bands, device=x.device, dtype=torch.float32)
        for b in range(B):
            mask_matrix[b, mask_ids[b]] = 1.0

        # Only compute loss where mask == 1
        diff = (pred - target) ** 2           # (B, bands)
        loss = (diff * mask_matrix).sum() / mask_matrix.sum().clamp(min=1)

        return loss, embedding


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: build model from config
# ══════════════════════════════════════════════════════════════════════════════

def build_hyperattnres(in_bands: int, n_blocks: int = None) -> HyperAttnResAE:
    """Build a HyperAttnResAE from the global config."""
    cfg = CFG
    return HyperAttnResAE(
        in_bands=in_bands,
        d_model=cfg["d_model"],
        n_blocks=n_blocks if n_blocks is not None else cfg["n_blocks"],
        n_heads=cfg["n_heads"],
        n_layers_per_block=cfg["n_layers_per_block"],
        ffn_ratio=cfg["ffn_ratio"],
        spectral_tokens=cfg["spectral_tokens"],
        embedding_dim=cfg["embedding_dim"],
        mask_ratio=cfg["mask_ratio"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Quick sanity check — run as: python 06_hyperattnres_model.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  HyperAttnRes — Forward-Pass Shape Sanity Check")
    print("=" * 60)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")

    for dataset, bands in [("Indian Pines", 176), ("Pavia University", 103)]:
        print(f"\n  [{dataset}] bands={bands}")
        model = build_hyperattnres(in_bands=bands).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Parameters: {n_params:,}")

        # Training forward pass (with masking)
        x = torch.randn(8, bands, 7, 7, device=DEVICE)
        loss, emb = model(x)
        print(f"    Training — loss: {loss.item():.4f}, emb shape: {emb.shape}")
        assert emb.shape == (8, 64), f"Expected (8,64) got {emb.shape}"

        # Inference forward pass (encode only)
        emb2, norms = model.encode(x, record_norms=True)
        print(f"    Inference — emb shape: {emb2.shape}, norm_list len: {len(norms)}")
        assert emb2.shape == (8, 64), f"Expected (8,64) got {emb2.shape}"
        assert len(norms) == CFG["n_blocks"] * CFG["n_layers_per_block"], \
            f"Expected {CFG['n_blocks'] * CFG['n_layers_per_block']} norms, got {len(norms)}"

        print(f"    ✓ All shape assertions passed")

    print("\n  ✓ Sanity check complete!")
