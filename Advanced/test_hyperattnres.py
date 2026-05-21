"""Quick forward-pass shape test for HyperAttnRes and Standard Transformer AE."""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import yaml
with open(os.path.join(SCRIPT_DIR, "hyperattnres_config.yaml")) as f:
    cfg = yaml.safe_load(f)
print(f"Config: n_blocks={cfg['n_blocks']}, n_layers_per_block={cfg['n_layers_per_block']}, spectral_tokens={cfg['spectral_tokens']}")

import importlib.util, torch

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

har_mod = load_mod('hyperattnres_model', os.path.join(SCRIPT_DIR, '06_hyperattnres_model.py'))
std_mod = load_mod('standard_transformer', os.path.join(SCRIPT_DIR, '07_standard_transformer_ae.py'))

print("\n--- HyperAttnRes ---")
for label, bands in [("Indian Pines", 176), ("Pavia U", 103)]:
    model = har_mod.build_hyperattnres(in_bands=bands).cpu()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(4, bands, 7, 7)
    loss, emb = model(x)
    emb2, norms = model.encode(x, record_norms=True)
    assert emb.shape == (4, 64), f"emb shape {emb.shape}"
    assert emb2.shape == (4, 64)
    assert len(norms) == 12, f"expected 12 norms, got {len(norms)}"
    print(f"  {label}: params={n_p:,}  loss={loss.item():.4f}  emb={emb.shape}  norms={len(norms)} layers  OK")

print("\n--- Standard Transformer AE ---")
for label, bands in [("Indian Pines", 176), ("Pavia U", 103)]:
    model = std_mod.build_standard_transformer_ae(in_bands=bands).cpu()
    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(4, bands, 7, 7)
    loss, emb = model(x)
    emb2, norms = model.encode(x, record_norms=True)
    assert emb.shape == (4, 64)
    assert emb2.shape == (4, 64)
    assert len(norms) == 12
    print(f"  {label}: params={n_p:,}  loss={loss.item():.4f}  emb={emb.shape}  norms={len(norms)} layers  OK")

# Verify zero-init of pseudo-queries
model = har_mod.build_hyperattnres(in_bands=176).cpu()
for block in model.encoder.transformer_blocks:
    for layer in block.layers:
        assert layer.attn_res_op.w.sum().item() == 0.0, "attn pseudo-query not zero-init!"
        assert layer.ffn_res_op.w.sum().item() == 0.0, "ffn pseudo-query not zero-init!"
print("\n  Zero-init check on pseudo-queries: PASSED")

print("\nAll assertions PASSED!")
