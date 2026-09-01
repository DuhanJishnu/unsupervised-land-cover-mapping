# Project Audit

For the consolidated current state, all validated experiments, decisions, and
future possibilities, start with `docs/RESEARCH_LEDGER.md`.

This audit covers every tracked Python module, notebook, configuration file,
runner, report generator, and the pipeline diagram as of 2026-09-01. Generated
models, datasets, and output tables are not versioned, so reported numerical
results could not be independently reproduced from the repository alone.

## Executive finding

The repository contains useful exploratory work, but its current outputs are
not yet suitable as research evidence. The main limitation is not model size:
the reconstruction objectives are not aligned with semantic clustering, while
file contracts, preprocessing choices, and evaluation rules vary between
stages. The recommended research contribution is therefore a
clustering-aligned spectral-spatial self-supervised learner with boundary-aware
spatial consistency. HyperAttnRes should remain an optional architecture
ablation until it demonstrates a repeatable advantage over a matched standard
transformer.

## Critical reproducibility findings

| Area | Finding | Consequence | Required correction |
| --- | --- | --- | --- |
| Artifact names | Preprocessing writes `{prefix}_all_patches.npy`; the old CNN trainer expected `PaviaU_patches.npy`. | The standard pipeline stops before training. | Use one artifact manifest and canonical prefixes (`ip`, `pu`, `enmap`). |
| Dataset coverage | The old preprocessing entry point ran Pavia University only, while later stages require Indian Pines. | A clean full run cannot produce all required inputs. | Iterate over the configured datasets. |
| Patch geometry | Preprocessing used 11x11 patches while CNN/advanced configurations declared 7x7. | Model descriptions and comparisons are inconsistent. | Treat patch size as one experiment parameter; default to 7x7. |
| PCA dimension | Configuration used 64 components while reports and figures called the result PCA-30D. | Tables do not describe the experiment that ran. | Use the configured value in all labels and metadata; baseline default is 30. |
| Indian Pines bands | `Indian_pines_corrected.mat` is already the corrected 200-band product, but old preprocessing removed water-absorption indices again. | Valid bands are deleted and band identity becomes incorrect. | Do not remove bands from the corrected product; only apply a raw-product mask to raw data. |
| EnMAP filenames | EnMAP preprocessing writes `enmap_train_patches.npy`; training/analysis expected an `_3` variant. | The EnMAP runner cannot complete. | Use `enmap_train_patches.npy` everywhere. |
| EnMAP loss range | EnMAP inputs are z-score normalized while the old CNN decoder ends in sigmoid. | Negative targets are impossible to reconstruct. | Use a linear decoder for z-score data or min-max inputs for sigmoid output. |
| EnMAP coordinates | Sample coordinates and scene identifiers were discarded. | A true map cannot be reconstructed; the synthetic rectangular mosaic has no geographic meaning. | Persist scene ID, row, column, valid mask, transform, CRS, and source path. |
| Advanced imports | The standard transformer imports another module relative to the process working directory. | Advanced scripts fail when launched from the repository root. | Resolve imports relative to `__file__`. |
| Dependency policy | Requirements use broad lower bounds and there is no lock file or environment record. | Results may change with library versions. | Add a lock/environment export before final experiments. |

## Methodological findings

### Data and preprocessing

- Global min-max scaling is sensitive to outliers. Robust percentile scaling or
  z-score statistics should be fitted and recorded explicitly.
- Scene-wise EnMAP z-scoring removes absolute inter-scene radiometric
  information. This may be acceptable for scene-specific transductive
  clustering but is unsuitable for cross-scene transfer unless tested as an
  ablation.
- EnMAP sampling accepts pixels with any finite band, then fills missing bands
  with zero. A zero can therefore mean normalized mean, missing observation, or
  spatial padding. A validity channel/mask is required.
- Zero padding creates artificial spectra around image boundaries. Reflect or
  validity-aware padding should be used, and invalid centers should not be
  sampled.
- Pre-extracting every overlapping patch duplicates the cube many times and can
  require tens of gigabytes. Patch extraction should be lazy or chunked.
- Benchmark ground truth may be read to evaluate results but must not affect
  normalization, representation learning, model selection, or early stopping.

### Models and objectives

- The CNN autoencoder minimizes patch MSE. Reconstruction rewards preservation
  of high-variance radiometry and local texture; it does not necessarily form
  semantic clusters.
- Adaptive global pooling can dilute the identity of the center pixel whose
  label is evaluated, particularly near class boundaries.
- A 64-dimensional bottleneck is hard-coded rather than justified by an
  ablation.
- The original decoder upsamples a small transposed-convolution output through
  bilinear interpolation. Low reconstruction loss does not establish useful
  spatial detail or semantic embeddings.
- The masked advanced objective is better aligned with spectral identity than
  full-patch MSE, but it masks a band at every spatial position. Neighboring
  spectra may make the task too easy; band-group masks and spatial masking
  should be ablated.
- The masking implementation uses Python loops per sample and can be vectorized.
- HyperAttnRes was motivated by depth-wise residual dilution in very large
  language models. Its relevance to a small HSI encoder must be demonstrated,
  not assumed.
- The first real-data smoke run exposed a uniform stationary point in the
  initial prototype objective: its loss remained at `log(k)` while dominating
  the scalar objective. Soft predictions were being used as their own stopped
  targets, so uniform assignments supplied no clustering signal. The rebuilt
  objective now uses sharpened Sinkhorn-balanced targets; its contribution must
  still be validated against `no_prototype` over multiple seeds.

### Clustering

- Setting `k` equal to the number of ground-truth classes uses oracle knowledge.
  It is acceptable only as a clearly named `oracle-k` protocol alongside an
  estimated-k experiment.
- KMeans is a necessary controlled baseline. Trying many clusterers and
  selecting one by ground-truth ARI is label-based model selection.
- Spectral and hierarchical clustering on a subset followed by k-NN or nearest
  centroid assignment is a different algorithm from full clustering and must
  be reported as such.
- Automatic DBSCAN epsilon selection from the largest second derivative is
  unstable and is not validated across scenes.
- Density-based results exclude noise before computing ARI, silhouette, and
  DBI. Coverage and rejected labeled pixels must be reported, and semantic
  metrics must follow a declared rejection policy.
- Mean-shift mode merging is order dependent in the custom implementation.

### Evaluation

- ARI is label-permutation invariant and useful, but it is insufficient alone.
  Add NMI, Hungarian-matched clustering accuracy, macro-F1, and mIoU.
- Report mean and standard deviation across at least five independent seeds.
- Silhouette and DBI measured in each model's own embedding space are not fair
  cross-model measures. Report them as within-space diagnostics, and separately
  measure every label assignment in one fixed reference feature space.
- Report density-clustering coverage, cluster-size entropy, runtime, peak
  memory, parameter count, and cluster stability.
- The existing majority filter can erase small classes and cross real
  boundaries. Compare it only as a baseline against a spectral-spatial graph,
  superpixel propagation, or edge-aware Potts/CRF refinement.
- Cluster colors are arbitrary. Qualitative ground-truth comparisons require
  Hungarian label matching; unlabeled EnMAP clusters must remain unnamed.
- The EnMAP PCA/CNN visualization uses CNN clusters to color both embeddings,
  making the comparison circular.
- The three-model t-SNE plot uses a projection from one embedding space to show
  labels produced in other spaces. Each representation requires its own
  projection, or all labels must be evaluated in a declared common space.

### Reporting and figures

- The README's representative results have no committed run manifest, exact
  environment, seed distribution, or raw experiment records.
- Figure titles contain conclusions such as "tighter, more distinct clusters"
  before a statistical comparison is made.
- The generated final report states that CNN and smoothing are successful
  regardless of the actual values read from CSV.
- The pipeline diagram differs from the implementation in patch geometry,
  architecture details, algorithms, and metrics.
- Notebooks duplicate older script logic, mix Indian Pines/Pavia and EnMAP state,
  rely on execution order, and contain undefined or stale variables. They
  should be exploratory only, never the source of final results.

## Scientific validity rules for the rebuilt pipeline

1. Every run receives a unique ID and records configuration, seed, code commit,
   dataset fingerprint, environment, and artifact paths.
2. Labels are loaded only in evaluation code.
3. Hyperparameters are fixed without looking at test labels. Cross-dataset
   transfer or a declared development scene is preferred.
4. Both oracle-k and estimated-k settings are reported.
5. Every comparison uses matched data, seeds, clusterer, and compute budget.
6. Every table is generated from machine-readable per-run records.
7. EnMAP outputs are maps only when original coordinates and georeferencing are
   retained.
8. Claims are based on repeated results and uncertainty, not a best seed.
