# Master Research Ledger

**Project:** Unsupervised hyperspectral land-cover mapping with learned embeddings  
**Ledger date:** 2026-09-01  
**Status:** active development; no final paper claim yet

This is the master record of what is known, what was changed, what the real
experiments show, what remains uncertain, and which research directions are
still credible. More focused evidence remains in `PROJECT_AUDIT.md`,
`EXPERIMENT_LOG.md`, `RESEARCH_PLAN.md`, and `DATA.md`.

## Contents

1. [Executive assessment](#1-executive-assessment)
2. [Claim status](#2-claim-status)
3. [Intended paper contribution](#3-intended-paper-contribution)
4. [Dataset facts and protocol roles](#4-dataset-facts-and-protocol-roles)
5. [Original repository findings](#5-original-repository-findings)
6. [Rebuilt research implementation](#6-rebuilt-research-implementation)
7. [Evaluation protocol](#7-evaluation-protocol)
8. [Completed real experiments](#8-completed-real-experiments)
9. [Findings classified by confidence](#9-findings-classified-by-confidence)
10. [Rejected or deprioritized choices](#10-rejected-superseded-or-deprioritized-choices)
11. [Threats and limitations](#11-threats-to-validity-and-current-limitations)
12. [Next possibilities](#12-next-possibilities)
13. [Prioritized roadmap](#13-prioritized-execution-roadmap)
14. [Paper experiment matrix](#14-paper-experiment-matrix)
15. [Recommended paper structure](#15-recommended-paper-structure)
16. [Immediate implementation task](#16-immediate-implementation-task)
17. [Decision log](#17-decision-log)
18. [Definition of paper readiness](#18-definition-of-paper-readiness)

## 1. Executive assessment

The original repository is a useful exploratory prototype, but its historical
results are not dependable research evidence. File contracts, dataset coverage,
patch geometry, preprocessing, objectives, clustering protocols, evaluation,
and reporting were inconsistent. The rebuilt research path is executable,
tested, provenance-aware, and produces reproducible artifacts.

The current scientific result is mixed:

- Learned embeddings can improve Indian Pines ARI and Hungarian-matched
  accuracy over raw-spectrum KMeans.
- They have not yet improved Indian Pines NMI, Macro-F1, or mIoU.
- Reconstruction-only embeddings can have excellent silhouette while being
  semantically poor.
- Exact uniform prototype balancing conflicts with the strongly imbalanced
  semantic structure of Indian Pines when applied for too long.
- A label-free prototype-usage stopping rule preserves the best early
  checkpoint on the development scene, but its first frozen Pavia transfer
  fails to beat raw/PCA baselines. Epoch-level stopping overshoots on the much
  larger scene.
- The spatial graph has not demonstrated a reliable benefit under the matched
  long-run schedule. It is an ablation, not part of the selected held-scene
  protocol.

The most defensible immediate direction is therefore not a larger architecture
or post-hoc Pavia tuning. It is step/sample-level prototype monitoring developed
without Pavia label selection, a better non-uniform prior, and validation on a
new untouched scene.

## 2. Claim status

### Supported by completed evidence

1. The original pipeline contained reproducibility-breaking artifact and
   preprocessing inconsistencies.
2. Raw and PCA KMeans baselines now run reproducibly on Indian Pines and Pavia
   University.
3. PCA-30 does not materially improve KMeans semantic metrics on either tested
   scene under the current robust-normalization protocol.
4. Reconstruction loss and within-space silhouette are not reliable proxies
   for semantic land-cover clustering.
5. The original stopped-soft-target prototype objective had a uniform
   stationary point at approximately `log(k)`.
6. Sinkhorn targets create active, increasingly balanced prototype use.
7. On Indian Pines, continued exact balancing past approximately 0.85
   normalized hard-usage entropy degrades ARI and accuracy.
8. Turning off only the prototype loss does not preserve the best partition;
   continued reconstruction/view learning still changes the embedding.
9. Label-free early stopping at usage entropy 0.85 preserves the best observed
   no-spatial development checkpoint.
10. The frozen 0.85 rule stops Pavia after one epoch at usage entropy 0.9433 and
    underperforms raw/PCA KMeans on every primary semantic metric.
11. Epoch-level stopping is not scale independent because an epoch contains
    very different numbers of optimizer steps across scenes.

### Not yet supported

1. The learned method is better than raw spectra overall.
2. The spatial graph improves results across seeds or scenes.
3. A fixed epoch-level entropy threshold transfers successfully across scenes.
4. The method is robust without oracle knowledge of class count.
5. The method generalizes across scenes rather than fitting each scene
   transductively.
6. HyperAttnRes improves the representation under matched compute.
7. The current EnMAP clusters correspond to named semantic land-cover classes.
8. Any single-seed development difference is statistically reliable.

## 3. Intended paper contribution

### Current working title

**Clustering-Aligned Spectral-Spatial Self-Supervision for Unsupervised
Hyperspectral Land-Cover Mapping**

### Initial hypothesis

A representation objective combining masked spectral reconstruction,
spectral-angle preservation, view consistency, balanced prototype assignments,
and boundary-aware spatial consistency should recover semantic land-cover
structure more reliably than PCA or reconstruction-only autoencoders.

### How evidence has changed the hypothesis

The prototype component is useful for early cluster formation but exact
equipartition is not a safe long-run prior for imbalanced land-cover scenes.
The revised hypothesis is:

> Clustering-aligned self-supervision is useful when anti-collapse pressure is
> bounded by a label-free stopping or prior-relaxation rule; permanent uniform
> assignment pressure can destroy semantically imbalanced structure.

This revised claim is promising but requires held-scene and multi-seed evidence.

### Possible final paper angles

1. **Method paper:** a clustering-aligned HSI encoder with a transferable,
   label-free usage-entropy stopping rule.
2. **Failure-aware paper:** an empirical study showing why reconstruction,
   silhouette, and permanent equipartition can mislead unsupervised HSI mapping,
   with a corrected evaluation protocol.
3. **Hybrid paper:** a compact method plus a rigorous evaluation and stopping
   framework. This is currently the strongest possible direction.

## 4. Dataset facts and protocol roles

| Dataset | Sensor | Shape | Bands | Classes | Labeled pixels | Role |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Indian Pines corrected | AVIRIS | 145 x 145 | 200 | 16 | 10,249 | Development scene |
| Pavia University | ROSIS | 610 x 340 | 103 | 9 | 42,776 | First held scene |
| EnMAP L2A | EnMAP | Scene-dependent | Product-dependent | Unlabeled | N/A | External case study |
| Salinas or Houston | TBD | TBD | TBD | TBD | TBD | Required third quantitative scene |

Important data rules:

- `Indian_pines_corrected.mat` already has water-absorption bands removed. Do
  not remove them again.
- Ground truth is background value zero and is used only for evaluation.
- Indian Pines labels may be inspected for development diagnostics, but Pavia
  label-dependent model selection is forbidden under the current protocol.
- EnMAP outputs are not maps unless scene ID, coordinates, valid mask,
  transform, CRS, and source association are preserved.
- Dataset bytes are local and ignored by Git. Exact hashes are recorded in
  `DATA.md` and every new manifest.

Validated SHA-256 values:

| File | SHA-256 |
| --- | --- |
| `Indian_pines_corrected.mat` | `ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939` |
| `Indian_pines_gt.mat` | `65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c` |
| `PaviaU.mat` | `28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb` |
| `PaviaU_gt.mat` | `23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829` |

The official UPV/EHU catalog supplied canonical metadata. Its download endpoint
returned HTTP 403, so validation bytes came from the public HybridSN mirror and
were checked by shape, keys, ranges, class counts, and hashes.

## 5. Original repository findings

### Critical reproducibility defects

| Defect | Consequence | Current response |
| --- | --- | --- |
| Preprocessor and trainer used different patch filenames | Clean pipeline stopped before training | Canonical dataset specifications and research run directories |
| Only Pavia was preprocessed while later stages expected both scenes | Indian Pines artifacts were missing | Benchmark iterator covers both datasets |
| Preprocessing used 11x11 while models/reports declared 7x7 | Experiment description did not match execution | Patch size is configured; research default is 7 |
| PCA configuration and figure labels disagreed | Tables described a different representation | PCA dimension is stored and rendered dynamically |
| Corrected Indian Pines bands were removed again | Valid spectral bands and identities were corrupted | Corrected product uses all 200 bands |
| EnMAP stages disagreed on filenames | Runner could not complete | Canonical EnMAP filenames |
| Z-score EnMAP target used sigmoid decoder | Negative targets could not be reconstructed | Linear decoder for z-score inputs |
| EnMAP coordinates and scenes were discarded | Fabricated rectangular maps had no geographic meaning | Coordinates, scene IDs, and metadata are persisted |
| Advanced imports depended on working directory | Root execution failed | Imports resolved relative to source location |
| Dependencies were broad and environment was absent | Reproduction could change across installations | Portable requirements plus exact tested lock file |

### Preprocessing risks that remain relevant

- Global min-max scaling is outlier sensitive; robust scaling is the research
  default.
- Scene-wise normalization may remove useful absolute radiometry and must be
  ablated for cross-scene transfer.
- Missing EnMAP bands need an explicit validity mask; zero cannot simultaneously
  represent a mean value, missing measurement, and spatial padding.
- Reflect padding is safer than zero padding, but boundary behavior still needs
  an ablation.
- Eager overlapping-patch materialization is wasteful; the research path uses
  lazy indexed patches and pre-pads each scene once.

### Legacy model risks

- Patch MSE rewards high-variance radiometry and texture, not semantic grouping.
- Adaptive global pooling can dilute the evaluated center pixel.
- A 64-dimensional bottleneck was historically assumed rather than justified.
- Transposed-convolution plus interpolation can report low reconstruction error
  without preserving meaningful spatial details.
- HyperAttnRes has a motivation imported from much deeper language models; it
  requires a matched HSI ablation before it can support a claim.

### Legacy clustering and reporting risks

- Oracle `k` was often treated as ordinary unsupervised clustering.
- Searching clusterers and selecting by ground-truth ARI is supervised model
  selection.
- Subset spectral/hierarchical clustering followed by assignment is not the
  same algorithm as full-scene clustering.
- Automatic DBSCAN epsilon selection was unstable.
- Noise-rejected clustering metrics hid coverage.
- Majority filtering can erase small classes and cross real boundaries.
- Arbitrary colors, shared t-SNE projections, and circular label coloring made
  some qualitative comparisons invalid.
- Historical README results lacked manifests, environment, and uncertainty.
- Notebooks contain duplicated and stale state; they remain exploratory only.

## 6. Rebuilt research implementation

### Package responsibilities

| Module | Responsibility |
| --- | --- |
| `landcover/config.py` | Canonical datasets, experiment configuration, oracle/explicit/estimated k |
| `landcover/data.py` | MAT loading, masks, normalization, coordinates, reflect-padded patch iteration |
| `landcover/torch_data.py` | Pre-padded patch dataset and spatially mixed tile batching |
| `landcover/augmentations.py` | Mild spectral noise, gain, band dropout, and spatial flips |
| `landcover/models.py` | Encoder, decoder, prototypes, losses, Sinkhorn, collapse diagnostics |
| `landcover/spatial.py` | Exact in-batch 4/8-neighbor graph with spectral weights |
| `landcover/baselines.py` | Raw/PCA KMeans/GMM and non-label k estimation |
| `landcover/evaluation.py` | Hungarian mapping, semantic/internal metrics, per-class diagnostics |
| `landcover/artifacts.py` | Dataset hashes, source-tree hash, environment, manifests |
| `landcover/training.py` | Deterministic training, checkpointing, probes, stopping, inference, aggregation |
| `landcover/reporting.py` | JSON/CSV/Markdown comparisons and learning curves |
| `research_pipeline.py` | `baseline`, `train`, and `compare` CLI commands |

### Encoder

The compact encoder has two branches:

- A 1D center-spectrum branch with convolutions, GroupNorm, GELU, and global
  spectral pooling.
- A lightweight spatial branch with 1x1 spectral mixing, depthwise 3x3 spatial
  convolution, pointwise projection, GroupNorm, GELU, and spatial pooling.

The branches are fused through an MLP into a normalized embedding. A linear
decoder reconstructs the center spectrum. A normalized linear prototype head
produces assignment logits.

### Current objective

\[
L = L_{masked} + 0.1L_{SAM} + L_{view} + L_{prototype} + 0.2L_{graph}.
\]

- `L_masked`: MSE on masked center-spectrum bands.
- `L_SAM`: spectral angle between reconstructed and target spectra.
- `L_view`: cosine agreement between HSI-safe augmented views.
- `L_prototype`: swapped prediction of sharpened, Sinkhorn-balanced targets.
- `L_graph`: cosine consistency for exact local neighbors weighted by spectral
  similarity.

Named ablations are `reconstruction`, `no_spectral_angle`, `no_view`,
`no_prototype`, `no_spatial`, and `full`.

### Important training behavior

- The default embedding dimension is 64 and patch size is 7; neither is yet
  justified by a completed ablation.
- AdamW uses initial learning rate `5e-4`, weight decay `1e-5`, and gradient
  clipping at 1.0.
- The cosine scheduler horizon is independent of the maximum or early stopping
  epoch. This fixes an earlier protocol error where changing `epochs` changed
  the entire early learning trajectory.
- Prototype use is logged as normalized hard-assignment entropy, maximum share,
  and active fraction.
- Optional semantic checkpoint probes are development-only because they read
  labels after each epoch.
- Label-free early stopping can use prototype-use entropy without reading
  ground truth.
- Every probed checkpoint is preserved; `checkpoint.pt` is the latest state.

### Reproducibility and artifacts

Every new run records:

- unique run ID;
- full experiment and training configuration;
- seed;
- dataset SHA-256 values;
- Git commit;
- executable source-tree SHA-256, including uncommitted Python/config files;
- Python, platform, and package versions;
- metrics and artifact paths;
- model, checkpoint, normalization, embeddings, coordinates, cluster map,
  history, and semantic diagnostics where applicable.

The source-tree fingerprint is necessary because most rebuilt files are still
uncommitted and a Git commit alone would misidentify the executed code.

### Test status

Thirteen automated tests pass in the locked local environment. They cover:

- invalid masks and robust normalization;
- patch shape and center preservation;
- Hungarian permutation invariance and rejection-aware metrics;
- per-class semantic diagnostics;
- model forward/loss finiteness;
- Sinkhorn normalization and approximate balance;
- spatial graph locality;
- spatial batch coverage;
- comparison artifact generation;
- one-epoch end-to-end artifacts;
- checkpoint semantic probes;
- persistent epoch checkpoints;
- label-free early stopping.

Compilation and `git diff --check` also pass. A harmless joblib warning about
detecting physical macOS cores remains; it falls back to logical core count and
does not change the algorithm.

## 7. Evaluation protocol

### Primary semantic metrics

- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)
- Hungarian-matched accuracy (ACC)
- Hungarian-matched Macro-F1
- Hungarian-matched mean IoU

### Diagnostics

- silhouette and Davies-Bouldin in the representation being evaluated;
- cluster-size entropy;
- rejection coverage and covered-only ARI/NMI;
- per-class precision, recall, F1, IoU, and support;
- prototype-use entropy, maximum share, and active fraction;
- runtime, inference/clustering time, parameters, and peak device memory;
- learning curves and completed epochs.

### Rules

1. Labels are not used for representation learning.
2. Indian Pines is the development scene. Pavia is held for frozen evaluation.
3. `--evaluation-every` is forbidden on held scenes.
4. Both oracle-k and estimated-k protocols must appear in the final paper.
5. Rejection must be counted as error in full semantic metrics and accompanied
   by coverage.
6. Silhouette and DBI are diagnostics, not semantic selection criteria.
7. Main claims require at least five seeds with mean and standard deviation.
8. Comparisons must match data, seeds, clusterer, architecture, and compute.
9. Final tables must come from machine-readable artifacts.
10. EnMAP clusters remain unnamed without independent reference labels.

## 8. Completed real experiments

### Raw and PCA oracle-k baselines

| Dataset | Representation | ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette | DBI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Indian Pines | Raw | 0.2238 | 0.4423 | 0.3751 | 0.3527 | 0.2431 | 0.2271 | 1.2963 |
| Indian Pines | PCA-30 | 0.2208 | 0.4390 | 0.3684 | 0.3524 | 0.2400 | 0.2526 | 1.2040 |
| Pavia University | Raw | 0.3125 | 0.5466 | 0.5363 | 0.5389 | 0.4418 | 0.4075 | 0.8022 |
| Pavia University | PCA-30 | 0.3121 | 0.5463 | 0.5360 | 0.5389 | 0.4418 | 0.4086 | 0.8003 |

Conclusion: PCA-30 is not an adequate improvement baseline; the learned method
must be compared to raw normalized spectra.

### Prototype-loss repair

The initial soft stopped-target loss stayed at 2.7730, approximately `log(16)`.
Uniform predictions were stationary. Replacing it with sharpened
Sinkhorn-balanced targets reduced the one-epoch prototype loss to 2.3242 and
created a real assignment gradient.

This fixed inactivity but revealed a second problem: persistent equipartition
can over-regularize imbalanced scenes.

### Three-epoch objective screening

| Method | ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw + KMeans | 0.2238 | 0.4423 | 0.3751 | 0.3527 | 0.2431 | 0.2271 |
| Reconstruction | 0.2264 | 0.4196 | 0.3366 | 0.1837 | 0.1179 | **0.5277** |
| No prototype | 0.2054 | 0.4084 | 0.3686 | 0.2749 | 0.1798 | 0.3070 |
| No spatial | 0.2634 | 0.4193 | 0.4057 | 0.2683 | 0.1867 | 0.4654 |
| Full | **0.2635** | 0.4189 | **0.4116** | **0.2810** | **0.1951** | 0.4750 |
| Full, 2x prototypes | 0.2500 | 0.4153 | 0.3840 | 0.2778 | 0.1838 | 0.4368 |

Key conclusions:

- Reconstruction has the best silhouette but poor semantic clustering.
- The prototype term is necessary for the observed ARI gain.
- Two-times overclustering does not solve imbalance and is not the default.
- These short runs used a scheduler horizon tied to three epochs; they are
  screening evidence, not directly comparable to later eight-epoch curves.

### Matched eight-epoch convergence

| Epoch | Full ARI | Full Macro-F1 | No-spatial ARI | No-spatial Macro-F1 | No-spatial ACC |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2519 | 0.2171 | 0.2613 | 0.2472 | 0.3974 |
| 2 | 0.2488 | 0.2485 | **0.2713** | **0.2690** | **0.4287** |
| 3 | 0.2400 | 0.2449 | 0.2327 | 0.2511 | 0.3582 |
| 4 | 0.2092 | 0.2580 | 0.2042 | 0.2632 | 0.3550 |
| 5 | 0.2124 | 0.2599 | 0.2110 | 0.2661 | 0.3479 |
| 6 | 0.2192 | 0.2481 | 0.2162 | 0.2560 | 0.3401 |
| 7 | 0.2073 | 0.2694 | 0.2040 | 0.2638 | 0.3514 |
| 8 | 0.2046 | 0.2555 | 0.2042 | 0.2562 | 0.3435 |

No-spatial is better at the useful early checkpoints. Both methods degrade as
prototype usage approaches uniformity. The graph is not the cause of the
long-run failure.

### Stopping experiments

| Method | Stop | ARI | NMI | ACC | Macro-F1 | mIoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| No-spatial, fixed pressure | Epoch 8 | 0.2042 | 0.3957 | 0.3435 | 0.2562 | 0.1619 |
| No-spatial, prototype off at entropy 0.85 | Epoch 8 | 0.2254 | 0.4173 | 0.3479 | 0.2562 | 0.1641 |
| No-spatial, early stop at entropy 0.85 | Epoch 2 | **0.2713** | **0.4322** | **0.4287** | **0.2690** | **0.1897** |

The selected development protocol is:

- no-spatial objective;
- maximum 8 epochs;
- fixed cosine horizon 8;
- label-free early stop at normalized hard-prototype usage entropy 0.85;
- one prototype per final cluster;
- robust normalization, 7x7 patches, 64-dimensional embedding;
- KMeans with 20 initializations for final clustering.

Raw KMeans still wins Indian Pines NMI, Macro-F1, and mIoU. No claim of overall
superiority is justified.

### Frozen held Pavia result

The selected no-spatial, eight-epoch-horizon, usage-entropy-0.85 protocol was
run once on Pavia without semantic checkpoint probes. It stopped after one
epoch at usage entropy 0.9433.

| Method | ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw + KMeans | **0.3125** | **0.5466** | **0.5363** | **0.5389** | **0.4418** | 0.4075 |
| PCA-30 + KMeans | 0.3121 | 0.5463 | 0.5360 | 0.5389 | 0.4418 | 0.4086 |
| Frozen learned protocol | 0.2475 | 0.4863 | 0.4446 | 0.4865 | 0.3606 | **0.4595** |

The held transfer fails. The learned method improves class 7 F1 from zero to
0.4311 but weakens several major classes and misses class 3. Better silhouette
again contradicts worse semantic performance. This result must remain in the
paper or project record and must not be erased by post-hoc Pavia tuning.

### Per-class failure pattern

The three-epoch full model had zero F1 for Indian Pines classes 1, 3, and 16.
It weakened several agricultural categories relative to raw spectra while
recovering a small amount of rare-class structure for classes 7 and 9.

This shows that global ARI/ACC gains can hide severe class failures. The final
paper must include per-class or size-stratified analysis and cannot rely only on
overall metrics.

## 9. Findings classified by confidence

### High confidence

- The old pipeline cannot support paper claims without the rebuild.
- Corrected Indian Pines must remain 200 bands.
- Raw KMeans is a strong mandatory baseline.
- PCA-30 adds little in the tested setup.
- Reconstruction loss and silhouette are misaligned with semantic quality.
- The original prototype loss was defective at uniform assignments.
- Long uniform balancing damages the Indian Pines development partition.
- Scheduler horizon must be fixed independently of stopping time.

### Medium confidence

- Prototype alignment helps early Indian Pines ARI/ACC.
- No-spatial is preferable to the current graph implementation.
- Usage entropy is a useful collapse/over-balancing diagnostic, but epoch-level
  threshold stopping is not scene-scale invariant.
- Class imbalance is the main reason exact equipartition is unsafe.

### Low confidence or unknown

- Whether step-level or sample-normalized entropy stopping transfers.
- Whether a revised method improves a new untouched held scene.
- Whether five-seed mean performance exceeds raw baselines.
- Whether estimated-k results remain competitive.
- Whether better spatial edges help after prototype dynamics are corrected.
- Whether cross-scene or cross-sensor transfer works.
- Whether HyperAttnRes adds any value.

## 10. Rejected, superseded, or deprioritized choices

| Choice | Status | Reason |
| --- | --- | --- |
| Historical README headline results | Rejected as evidence | No complete manifests/environment/uncertainty |
| Double-removing Indian Pines bands | Rejected | Corrected cube already has 200 valid bands |
| Sigmoid decoder on z-score data | Rejected | Cannot reconstruct negative targets |
| Fabricated EnMAP rectangular map | Rejected | Coordinates and scene identity were lost |
| Reconstruction-only as main method | Rejected | Poor Macro-F1/mIoU despite high silhouette |
| Soft self-target prototype loss | Superseded | Uniform stationary point at `log(k)` |
| Permanent exact equipartition | Deprioritized | Degrades semantic partition after early epochs |
| Two-times prototype overclustering | Deprioritized | Worse primary metrics in the pilot |
| Prototype-off continuation | Rejected as stopping response | Does not preserve peak embedding |
| Current spatial graph in primary method | Deprioritized | No reliable matched-schedule gain |
| HyperAttnRes as primary contribution | Deferred | Objective/protocol must stabilize first |
| Silhouette-based model selection | Rejected | Contradicted semantic metrics empirically |

## 11. Threats to validity and current limitations

### Experimental

- Learned results are currently seed 42 only.
- Indian Pines has been used for decisions and cannot be presented as an
  unbiased held benchmark.
- The first frozen Pavia learned run failed against raw/PCA baselines.
- The third labeled benchmark is not selected or downloaded.
- GMM and several planned baselines have implementation support but no recorded
  real comparison in the new experiment log.
- Estimated-k is implemented but not yet evaluated.
- Current final clustering is KMeans; conclusions may partly reflect its
  spherical-cluster assumption.

### Methodological

- Oracle-k uses ground-truth class count.
- Transductive whole-scene learning does not demonstrate inductive transfer.
- Sinkhorn assumes balanced prototype targets even when semantic classes are
  imbalanced.
- The entropy threshold may depend on batch size, prototype count, class count,
  scene complexity, or sensor.
- Prototype hard-use entropy can appear healthy while prototype semantics are
  poor.
- The current graph uses only in-batch local neighbors and spectral-similarity
  weights; it is not a full-scene graph or CRF.
- No boundary F-score, region consistency, or small-object retention metric is
  implemented yet.
- Fixed 7x7 patches and 64-dimensional embeddings remain unablated.
- Random independent band masking may be too easy because adjacent wavelengths
  and neighboring pixels reveal the answer.

### Reproducibility

- The rebuilt worktree is not committed yet. Source hashes protect run identity,
  but a paper release needs a clean tagged commit.
- Generated outputs and datasets are ignored by Git; a release needs a results
  bundle, DOI, or scripted download/regeneration path.
- The exact lock file describes one macOS arm64 Python 3.9 environment while the
  README advertises Python 3.10+. Cross-platform validation remains necessary.
- MPS peak memory is currently reported as zero because the CUDA memory API does
  not cover Apple MPS.

## 12. Next possibilities

The following options are ordered by scientific value and dependency, not by
novelty.

### P0 — Scale-independent stopping and new held validation

The frozen Pavia run is complete and failed. Its label-free trace reveals that
epoch-level monitoring overshot the target because Pavia has roughly ten times
the valid pixels and optimizer steps per epoch.

Implement next on Indian Pines or another development scene:

1. Measure prototype usage every fixed number of optimizer steps, not only at
   epoch boundaries.
2. Use an exponential moving average or consecutive-window requirement so a
   noisy batch cannot stop training.
3. Record total samples and optimizer updates at threshold crossing.
4. Preserve a checkpoint at the first valid crossing.
5. Do not validate the revision by repeatedly selecting on Pavia labels. Use a
   newly acquired third scene as the next untouched held test.

### P1 — Five-seed confirmation

Run the frozen protocol with seeds 42–46 on Indian Pines for development
uncertainty and Pavia for held evaluation. Report mean, standard deviation,
paired seed differences, and per-class variation. A single best seed must never
be the headline.

Success criterion: the mean learned result improves at least ARI and ACC without
catastrophic Macro-F1/mIoU loss, and the direction is stable across seeds.

### P1 — Complete trusted baselines

Record real runs for:

- PCA-30 + GMM;
- reconstruction CNN-AE with the frozen scheduler protocol;
- masked spectral autoencoder;
- view-consistency-only encoder;
- a spatial/superpixel method independent of learned prototypes.

All baselines need the same normalization, valid pixels, cluster-count protocol,
seeds, and final evaluation code.

### P1 — Estimated-k protocol

The code estimates k using repeated 80% subsample KMeans stability plus
silhouette without labels. Run a prespecified range on both scenes and freeze
range/scoring rules before semantic evaluation.

Questions:

- Does estimated k agree with oracle k?
- How much performance is lost without class-count knowledge?
- Is the learned embedding more stable for k estimation than PCA?

### P2 — Replace exact uniform prototype prior

This is the most important method-development possibility if held transfer
fails.

Candidates:

1. **EMA marginal prior:** estimate prototype frequencies over time and balance
   against a slowly changing non-uniform prior rather than uniform occupancy.
2. **Relaxed optimal transport:** constrain only minimum usage or use an
   unbalanced/Sinkhorn-Knopp variant with slack.
3. **Teacher-student centering and sharpening:** use an EMA teacher with
   centering to avoid collapse without forcing exact equal mass.
4. **Entropy band:** penalize only collapse below a lower usage threshold and
   stop adding balance pressure above it.
5. **Hierarchical prototypes:** allow multiple subprototypes per semantic mode,
   then merge using spectral/spatial evidence. Simple 2x prototypes already
   failed, so merging and prior design—not count alone—would be essential.

Required success criterion: improve Macro-F1/mIoU or preserve them while
retaining ARI/ACC gains, across seeds and on held Pavia.

### P2 — Improve spectral pretext tasks

Candidates:

- contiguous wavelength-group masking instead of independent random bands;
- mixed random/group masks;
- mask neighboring spatial context as well as bands to prevent leakage;
- reconstruct spectral derivatives or continuum-normalized shape;
- predict held spectral regions from disjoint spectral context;
- compare MSE, Huber, SAM, and derivative losses.

These should be ablated on Indian Pines only, then frozen before held testing.

### P2 — Revisit spatial reasoning

The present in-batch graph is not selected, but spatial structure remains
important for mapping.

Candidates:

- superpixels formed without labels, followed by prototype consistency;
- full-scene sparse graph with edge weights from spectral distance and image
  gradients;
- confidence-aware propagation after clustering;
- edge-aware Potts/CRF refinement;
- boundary-preserving neighborhood contrast rather than pure smoothing.

Required metrics: boundary F-score, small-region recall, local consistency,
Macro-F1, and mIoU. Majority filtering is only a baseline.

### P2 — Patch and embedding ablations

Test patch sizes 1, 5, 7, and 11 and embedding sizes 16, 32, 64, and 128 under a
small prespecified development budget. Patch size 1 separates spectral from
spatial gains. Match parameter counts where architecture comparisons require it.

### P3 — Cross-scene and cross-sensor generalization

Possibilities:

- train or pretrain on one scene and cluster another without fine-tuning;
- use a common wavelength grid or sensor-aware band embeddings;
- compare scene-wise, dataset-wise, and global normalization;
- evaluate frozen features with KMeans on held scenes;
- separate transductive results clearly from inductive transfer.

This direction is necessary for a strong generalization claim but depends on a
stable objective first.

### P3 — EnMAP case study

Use coordinate-preserving inference and restore georeferencing. Report
unnamed clusters, stability, spatial coherence, spectral profiles, and failure
cases. Quantitative semantic claims require an independent reference map that
is never used for training or selection.

### P4 — Architecture comparison

Only after the objective and protocol are frozen:

- compare compact CNN, standard transformer, and HyperAttnRes;
- match embedding dimension, objective, optimizer, scheduler, seeds, parameter
  count, and compute;
- report whether architecture adds value beyond the training objective.

If HyperAttnRes does not show a repeated matched advantage, omit it from the
main paper and retain it as an appendix or negative result.

## 13. Prioritized execution roadmap

| Priority | Experiment | Dependency | Approximate cost | Decision enabled |
| --- | --- | --- | --- | --- |
| 0 | Step/sample-level stopping monitor | Develop without Pavia label selection | Medium | Is stopping scene-scale invariant? |
| 0 | Acquire third labeled scene | Dataset/provenance decision | Medium | Provides a new untouched held test |
| 1 | Five seeds on a revised selected protocol | New held one-seed result | Very high | Are gains statistically stable? |
| 1 | PCA-GMM and masked-AE real baselines | Current code/minor additions | Medium | Is comparison set credible? |
| 1 | Estimated-k on raw/PCA/learned features | Current estimator | Medium | How dependent is method on oracle k? |
| 2 | Relaxed/non-uniform prototype prior | Held result/failure diagnosis | Medium-high | Can Macro-F1/mIoU improve? |
| 2 | Contiguous spectral masking ablation | Stable training | Medium | Is pretext task too easy? |
| 2 | Boundary-aware spatial alternatives | Boundary metrics | High | Is spatial reasoning actually useful? |
| 2 | Patch/embedding dimensions | Stable objective | High | Are defaults justified? |
| 3 | Third labeled scene | Dataset selection/provenance | High | Does evidence generalize? |
| 3 | EnMAP georeferenced case study | Stable encoder | High | External real-world demonstration |
| 4 | Transformer/HyperAttnRes | Frozen objective | Very high | Is architecture contribution real? |

## 14. Paper experiment matrix

### Main table

Rows:

- raw + KMeans;
- PCA-30 + KMeans;
- PCA-30 + GMM;
- reconstruction AE + KMeans;
- masked spectral AE + KMeans;
- view-consistency encoder + KMeans;
- selected clustering-aligned method;
- selected method with estimated k.

Columns:

- ARI, NMI, ACC, Macro-F1, mIoU;
- mean ± standard deviation over at least five seeds;
- runtime and parameters in a separate efficiency table.

### Ablation table

- reconstruction only;
- +SAM;
- +view consistency;
- +prototype objective;
- fixed versus relaxed/stopped balance;
- no-spatial versus selected spatial method;
- oracle versus estimated k.

### Figures

1. Method diagram showing spectral branch, spatial branch, objectives, and
   label-free stopping signal.
2. Indian Pines development learning curves: semantic metrics, prototype usage,
   prototype loss, and learning rate.
3. Held Pavia cluster map and Hungarian-matched reference comparison.
4. Per-class F1 versus class support.
5. Boundary/small-region analysis if a spatial method survives.
6. Failure cases where learned clusters split or merge semantic classes.
7. EnMAP georeferenced unnamed-cluster case study.

### Statistical analysis

- five or more matched seeds;
- mean and standard deviation;
- paired bootstrap or paired seed test where justified;
- effect sizes, not only p-values;
- no best-seed-only reporting;
- prespecified model selection and stopping rules.

## 15. Recommended paper structure

1. Introduction and problem statement
2. Related work: HSI self-supervision, deep clustering, spatial regularization
3. Audit-derived design principles
4. Clustering-aligned representation method
5. Label-free collapse/over-balancing control
6. Experimental protocol and leakage controls
7. Main benchmark results
8. Objective, stopping, and spatial ablations
9. Cross-scene and EnMAP case studies
10. Failure analysis and limitations
11. Reproducibility statement
12. Conclusion

## 16. Immediate implementation task

The frozen Pavia command has been completed and failed. The immediate valid
implementation task is step-level, sample-normalized prototype-usage monitoring
developed without selecting on Pavia labels. The next semantic validation must
use a new untouched scene. Repeated Pavia variants would convert the held scene
into a development set and invalidate the original protocol claim.

## 17. Decision log

| Decision | Evidence | Current status |
| --- | --- | --- |
| Treat historical outputs as non-paper evidence | Missing manifests and inconsistent pipeline | Final |
| Use corrected 200-band Indian Pines without further band deletion | Dataset validation | Final |
| Use robust normalization by default | Outlier concerns and matched baseline protocol | Active; still ablate |
| Keep raw KMeans as mandatory baseline | It beats PCA and many learned semantic metrics | Final |
| Replace soft self-target prototypes with Sinkhorn targets | Uniform stationary point | Final implementation |
| Do not use silhouette for semantic selection | Reconstruction counterexample | Final |
| Deprioritize 2x prototypes | Worse three-epoch metrics | Active unless better merging prior exists |
| Deprioritize current graph | No matched convergence advantage | Active |
| Use no-spatial for held evaluation | Best matched early development checkpoint | Frozen for Pavia |
| Stop at prototype-use entropy 0.85 | Preserves development peak but fails held Pavia | Completed negative transfer |
| Separate scheduler horizon from stopping time | Earlier trajectories were incomparable | Final implementation |
| Declare Indian Pines development, Pavia held | Prevent label-dependent Pavia tuning | Final protocol |
| Defer HyperAttnRes | Objective/protocol uncertainty dominates | Active |
| Record frozen Pavia transfer as failure | Learned method loses all primary semantic metrics | Final evidence |
| Do not tune revised stopping on Pavia | Held labels have now been inspected | Final protocol |

## 18. Definition of paper readiness

The project is not paper-ready until all of the following are true:

- Pavia held failure is reported transparently without post-hoc replacement.
- At least five seeds are complete for main learned methods.
- GMM, masked-AE, and other declared main baselines are recorded.
- Oracle-k and estimated-k results are reported.
- A third labeled scene is included or the scope explicitly justifies two.
- Patch/embedding defaults have evidence or are declared limitations.
- Boundary metrics exist if spatial claims remain.
- All tables and figures are generated from manifests.
- The working tree is committed and tagged.
- Reproduction instructions work in a clean environment.
- Claims match the mixed metric evidence and include failure cases.

Until then, all learned numbers are development findings, not final headline
results.
