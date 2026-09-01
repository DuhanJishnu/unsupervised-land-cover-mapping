# Research Plan

For the consolidated current evidence, implementation inventory, decision log,
risks, and all prioritized future possibilities, see
`docs/RESEARCH_LEDGER.md`.

## Working title

**Clustering-Aligned Spectral-Spatial Self-Supervision for Unsupervised
Hyperspectral Land-Cover Mapping**

## Central hypothesis

A representation objective that jointly enforces spectral reconstruction,
cluster-balanced view consistency, and boundary-aware spatial consistency will
recover semantic land-cover structure more reliably than PCA or
reconstruction-only autoencoders.

## Proposed method

The initial model is a compact spectral-spatial encoder trained with:

\[
L = L_{masked} + \lambda_{sam}L_{spectral-angle}
  + \lambda_{view}L_{view-consistency}
  + \lambda_{proto}L_{balanced-prototype}
  + \lambda_{spatial}L_{graph}.
\]

- `L_masked`: reconstruct masked center-pixel band groups.
- `L_spectral-angle`: preserve spectral shape independently of magnitude.
- `L_view-consistency`: align HSI-safe augmented views.
- `L_balanced-prototype`: form non-collapsed, cluster-oriented assignments.
- `L_graph`: align spatial neighbors only when spectral/edge evidence supports
  the connection.

HyperAttnRes is not part of the primary claim. It becomes an architecture
ablation after the objective and evaluation pipeline are stable.

## Research questions

- RQ1: Does clustering-aligned training outperform reconstruction-only learning?
- RQ2: Which spectral and spatial self-supervised terms contribute independently?
- RQ3: Does boundary-aware graph consistency improve region coherence without
  destroying small classes?
- RQ4: How much performance depends on oracle knowledge of the class count?
- RQ5: Do embeddings transfer across scenes and sensors?
- RQ6: Does HyperAttnRes add value under matched objective, depth, parameters,
  and compute?

## Datasets and protocols

Quantitative benchmarks:

- Indian Pines
- Pavia University
- Salinas or Houston as a third labeled scene

Protocol-selection roles are now explicit: Indian Pines is the development
scene; Pavia University is the first held scene. Epoch/stopping rules selected
from Indian Pines must be frozen before inspecting Pavia semantic metrics.

EnMAP is initially an unlabeled external case study. Quantitative semantic
claims require an independent reference map that is never used for training.

Two cluster-count protocols are mandatory:

- `oracle-k`: k equals the benchmark class count, for comparison with literature.
- `estimated-k`: k selected without labels using stability/eigengap/information
  criteria.

The primary setting is transductive whole-scene clustering and must be named as
such. Cross-scene experiments hold out complete scenes, not random patches.

## Baselines

1. Raw normalized spectrum + KMeans
2. PCA-30 + KMeans
3. PCA-30 + GMM
4. Reconstruction CNN-AE + KMeans
5. Masked spectral autoencoder + KMeans
6. Contrastive-only spectral-spatial encoder
7. Spatial/superpixel graph baseline
8. Proposed full objective
9. Standard transformer with the proposed objective
10. HyperAttnRes with the same objective and compute budget

## Metrics

Primary semantic metrics:

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Hungarian-matched clustering accuracy (ACC)
- Macro-F1 and mean IoU after matching

Diagnostics:

- silhouette and Davies-Bouldin in learned space
- silhouette and Davies-Bouldin in a fixed PCA reference space
- rejection coverage for density methods
- cluster-size entropy and stability across seeds
- boundary F-score / local consistency
- runtime, peak memory, parameters, and training compute

All main results use at least five seeds and report mean plus standard deviation.

## Ablation matrix

- Full-patch MSE vs masked center-spectrum reconstruction
- Random bands vs contiguous wavelength groups
- MSE vs MSE + spectral-angle loss
- No augmentation vs spectral-only vs spectral-spatial views
- No prototype loss vs prototype loss
- One prototype per final cluster vs 2x and 4x internal overclustering
- Fixed prototype pressure vs label-free stopping at a target usage entropy
- Label-free early stopping when prototype usage reaches the prespecified target
- No spatial term vs majority filter vs graph consistency
- Patch size 1, 5, 7, 11
- Embedding dimension 16, 32, 64, 128
- Oracle-k vs estimated-k
- CNN vs standard transformer vs HyperAttnRes

## Implementation milestones

### M0 — Research foundation

- Canonical configuration and artifact names
- Corrected dataset handling and masks
- Coordinate-preserving lazy patch extraction
- Run manifests and deterministic seeds
- Rigorous metric implementation and tests

Status: implemented and exercised with synthetic data. The repository now has
an exact tested-environment lock file in addition to the portable requirements.

### M1 — Trusted baselines

- Raw, PCA, KMeans, GMM
- Reconstruction CNN-AE
- Masked spectral AE
- Five-seed result aggregation

Implementation status: the executable training path now supports spatially
mixed minibatches, boundary-aware graph edges, checkpoints/resume, embedding
extraction, KMeans evaluation, five-seed summaries, and named loss ablations.
These components have now completed real-data baseline and one-epoch smoke
runs. Full multi-seed, multi-epoch experiments remain required before any
scientific result is claimed.

Validation status: all thirteen automated tests pass in the locked environment,
including a one-epoch end-to-end synthetic training run. Four real KMeans
baselines and a one-epoch full-model Indian Pines smoke run also complete. See
`docs/EXPERIMENT_LOG.md`; the benchmark files remain intentionally unversioned.

### M2 — Clustering-aligned objective

- HSI-safe view generation
- Balanced prototype head
- View-consistency loss
- Collapse diagnostics

Status: implemented. A three-epoch Indian Pines pilot shows the prototype term
is active and non-collapsed, but the full method currently trades improved ARI
and ACC for worse Macro-F1/mIoU than raw spectra. Full and no-spatial advance to
a longer convergence pilot; 2x prototype overclustering does not advance.

Convergence update: matched eight-epoch curves show continued uniform
prototype pressure degrades semantic structure after epoch 2. No-spatial plus
label-free early stopping at prototype-use entropy 0.85 is the selected
development protocol. The scheduler horizon is now independent of stopping
time, and every probed checkpoint is retained.

### M3 — Spatial reasoning

- Spectral-edge-aware local graph or superpixels
- Confidence-aware propagation
- Boundary and small-region evaluation

### M4 — Generalization

- Held-out scenes
- EnMAP coordinate-preserving inference
- Cross-sensor normalization/wavelength study

### M5 — Paper

- Frozen protocol and configurations
- Full ablation and statistical analysis
- Automatically generated tables and figures
- Limitations, failure cases, and reproducibility appendix

## Paper structure

1. Introduction and hypothesis
2. Related work
3. Clustering-aligned spectral-spatial method
4. Experimental protocol
5. Benchmark results
6. Objective and spatial ablations
7. Cross-scene EnMAP case study
8. Limitations and failure cases
9. Conclusion
