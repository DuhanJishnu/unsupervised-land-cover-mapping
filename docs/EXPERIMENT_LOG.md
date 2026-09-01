# Experiment Log

For the consolidated interpretation, implementation inventory, open risks, and
prioritized next possibilities, see `docs/RESEARCH_LEDGER.md`.

This file records completed, reproducible experiments. Generated arrays and
weights live under `outputs/research/` and are intentionally ignored by Git.
Numbers here are development results, not final paper claims, until the full
multi-seed protocol and all prespecified baselines are complete.

## 2026-09-01 — Benchmark validation

Both benchmark pairs loaded successfully and matched their canonical metadata:

- Indian Pines corrected: 145 x 145 pixels, 200 bands, 16 non-background
  ground-truth classes, 10,249 labeled pixels.
- Pavia University: 610 x 340 pixels, 103 bands, 9 non-background
  ground-truth classes, 42,776 labeled pixels.

Hashes and download provenance are in `docs/DATA.md`.

## 2026-09-01 — Oracle-k KMeans baselines

Configuration: robust normalization, seed 42, KMeans with the canonical class
count, transductive whole-scene fitting. Semantic metrics are computed only on
labeled pixels; labels are used only after clustering for evaluation and
Hungarian matching. Silhouette and Davies-Bouldin are computed in the fitted
representation.

| Dataset | Representation | ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette | DBI |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Indian Pines | Raw spectrum | 0.2238 | 0.4423 | 0.3751 | 0.3527 | 0.2431 | 0.2271 | 1.2963 |
| Indian Pines | PCA-30 | 0.2208 | 0.4390 | 0.3684 | 0.3524 | 0.2400 | 0.2526 | 1.2040 |
| Pavia University | Raw spectrum | 0.3125 | 0.5466 | 0.5363 | 0.5389 | 0.4418 | 0.4075 | 0.8022 |
| Pavia University | PCA-30 | 0.3121 | 0.5463 | 0.5360 | 0.5389 | 0.4418 | 0.4086 | 0.8003 |

Finding: PCA-30 changes the internal geometry diagnostics slightly but does not
improve semantic clustering on either scene. The learned method therefore must
beat the raw baseline, not merely PCA, and performance must be reported
separately by scene because Indian Pines is markedly harder under this setup.

Run IDs:

- `ip-raw-kmeans-6a5f40f8`
- `ip-pca-kmeans-cc4955fb`
- `pu-raw-kmeans-e50d42b3`
- `pu-pca-kmeans-816fc0bc`

## 2026-09-01 — Full-model real-data smoke run

Configuration: Indian Pines, full proposed objective, one epoch, batch size
512, seed 42, oracle k=16, automatic device selection. The run trained on all
valid scene pixels without consulting ground-truth labels and completed model,
checkpoint, normalization, embedding, coordinate, cluster-map, manifest, and
summary artifact generation.

| ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette | Parameters | Train time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2478 | 0.4202 | 0.4049 | 0.2749 | 0.1816 | 0.2701 | 73,960 | 82.55 s |

Run ID: `ip-aligned-full-949aff52`.

Interpretation: after only one epoch, the embedding improves ARI and matched
accuracy over raw KMeans, but lowers NMI, Macro-F1, and mIoU. This is a pipeline
validation and an early indication of class imbalance or prototype-collapse
pressure, not evidence that the proposed method is superior. The next decisive
experiment is a multi-epoch, five-seed comparison against reconstruction-only
and no-prototype/no-spatial ablations.

Post-run diagnosis: the prototype term was 2.7730, essentially `log(16)`, and
the implementation's uniform soft self-target was a stationary point. The
prototype objective was subsequently changed to swapped prediction of
sharpened Sinkhorn-balanced assignments. Therefore this smoke run validates the
artifact path but is explicitly a superseded model result; subsequent learned
runs use the corrected objective.

## 2026-09-01 — Corrected-prototype smoke run

The same one-epoch configuration was rerun after replacing soft self-targets
with sharpened Sinkhorn-balanced assignments.

| ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette | Cluster entropy | Train time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2519 | 0.4008 | 0.3816 | 0.2171 | 0.1456 | 0.4669 | 0.7112 | 87.95 s |

Run ID: `ip-aligned-full-7335fede`. Prototype loss fell to 2.3242 instead of
remaining at `log(16)`, confirming that the new target supplies a nontrivial
optimization signal. ARI and silhouette improved slightly over the superseded
smoke run, but most semantic metrics and final KMeans cluster-size entropy
worsened after one epoch. Balanced minibatch targets do not guarantee balanced
KMeans outputs and may conflict with Indian Pines' strongly imbalanced semantic
classes. Prototype usage entropy, maximum share, and active fraction are now
logged per epoch so longer runs can distinguish transient specialization from
collapse. The prototype prior and its weight remain ablation questions.

## 2026-09-01 — Three-epoch objective pilot

Configuration: Indian Pines, robust normalization, oracle k=16, seed 42, batch
size 512, three epochs, matched architecture and optimizer. This is a screening
experiment and has no uncertainty estimate.

| Method | ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw spectrum + KMeans | 0.2238 | 0.4423 | 0.3751 | 0.3527 | 0.2431 | 0.2271 |
| Reconstruction only | 0.2264 | 0.4196 | 0.3366 | 0.1837 | 0.1179 | 0.5277 |
| No prototype | 0.2054 | 0.4084 | 0.3686 | 0.2749 | 0.1798 | 0.3070 |
| No spatial | 0.2634 | 0.4193 | 0.4057 | 0.2683 | 0.1867 | 0.4654 |
| Full | 0.2635 | 0.4189 | 0.4116 | 0.2810 | 0.1951 | 0.4750 |
| Full, 2x prototypes | 0.2500 | 0.4153 | 0.3840 | 0.2778 | 0.1838 | 0.4368 |

Run IDs are `ip-aligned-reconstruction-2994a385`,
`ip-aligned-no_prototype-d0edcd8b`, `ip-aligned-no_spatial-b70676ae`,
`ip-aligned-full-9679dfd8`, and `ip-aligned-full-7b6969ce`. The generated
comparison is under `outputs/research/pilot-ip-3epoch-20260901/` and includes
JSON, CSV, Markdown, and per-class CSV.

Findings:

- The full method and no-spatial variant beat raw KMeans on ARI and ACC, but
  not on NMI, Macro-F1, or mIoU. The present method is therefore not yet a
  general baseline winner.
- The spatial graph gives small consistent gains over no-spatial in this seed:
  +0.0001 ARI, +0.0059 ACC, +0.0127 Macro-F1, and +0.0084 mIoU. This effect is
  too small to claim without paired multi-seed uncertainty.
- Reconstruction-only has the highest silhouette (0.5277) while producing the
  worst semantic Macro-F1 and mIoU. This empirically confirms that internal
  embedding metrics cannot select the semantic model.
- Per-class diagnostics show the full model has zero F1 for classes 1, 3, and
  16. It weakens several agricultural classes relative to raw spectra while
  recovering a small amount of rare-class structure for classes 7 and 9.
- Doubling internal prototypes does not solve the imbalance at three epochs;
  it lowers every primary semantic metric relative to the 1x full model. Keep
  overclustering as an ablation, not the default.

Decision: retain the full and no-spatial objectives for a longer convergence
pilot, keep raw spectra as the mandatory reference, and do not promote the 2x
prototype setting. The final five-seed experiment should begin only after the
longer pilot establishes a stable epoch budget.

## 2026-09-01 — Eight-epoch convergence and stopping study

Indian Pines is declared the development scene for this study. Its checkpoint
labels are used only for diagnosis and protocol selection; the selected
label-free stopping rule must be frozen before Pavia University evaluation.
Both fixed-pressure curves used an eight-epoch cosine horizon and seed 42.

### Checkpoint curves

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

Run IDs: `ip-aligned-full-22365d46` and
`ip-aligned-no_spatial-f7642d11`.

Prototype-use entropy rises from approximately 0.67 at epoch 1 to 0.86 at
epoch 2 and above 0.93 after epoch 4. Semantic ARI and ACC degrade as continued
training forces assignments closer to uniform usage. Because no-spatial shows
the same degradation, the graph is not its cause. Under the matched schedule,
no-spatial dominates full at the useful early checkpoints and is selected.

Changing the requested epoch count previously also changed cosine `T_max`, so
short and long runs followed different early learning-rate trajectories. The
implementation now exposes `scheduler_epochs` separately and requires it to be
at least the stopping budget. All probed checkpoints are preserved.

### Label-free stopping response

Two responses were tested with no-spatial:

- Disabling only the prototype loss after usage entropy reached 0.85 did not
  preserve the partition. Continued reconstruction/view training reduced
  prototype usage and ended at ARI 0.2254 (`ip-aligned-no_spatial-f4234507`).
- Stopping all training when usage entropy first reached 0.85 halted at epoch
  2 and exactly preserved ARI 0.2713, NMI 0.4322, ACC 0.4287, Macro-F1 0.2690,
  and mIoU 0.1897 (`ip-aligned-no_spatial-f7adc0c8`). This rule uses no class
  labels and is therefore suitable for frozen transfer to Pavia.

The selected development protocol is: no-spatial objective, maximum eight
epochs, cosine horizon eight, and early stopping at normalized hard-prototype
usage entropy 0.85. Raw KMeans still has higher NMI, Macro-F1, and mIoU, so the
learned method is not yet claimed to dominate the baseline.

Generated comparison and learning curves are under
`outputs/research/convergence-ip-20260901/`.

## 2026-09-01 — Frozen held Pavia evaluation

The Indian Pines-selected protocol was executed without semantic checkpoint
probes or Pavia-dependent tuning: `no_spatial`, maximum eight epochs, cosine
horizon eight, label-free early stop at prototype-use entropy 0.85, robust
normalization, patch size 7, embedding dimension 64, oracle k=9, and seed 42.

Pavia has 207,400 valid training pixels and 42,776 labeled evaluation pixels.
Usage entropy reached 0.9433 after the first epoch, so the frozen rule stopped
training at epoch 1. Only then were labels used for final evaluation.

| Method | ARI | NMI | ACC | Macro-F1 | mIoU | Silhouette | DBI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw + KMeans | **0.3125** | **0.5466** | **0.5363** | **0.5389** | **0.4418** | 0.4075 | 0.8022 |
| PCA-30 + KMeans | 0.3121 | 0.5463 | 0.5360 | 0.5389 | 0.4418 | 0.4086 | 0.8003 |
| Frozen learned protocol | 0.2475 | 0.4863 | 0.4446 | 0.4865 | 0.3606 | **0.4595** | **0.7484** |

Run ID: `pu-aligned-no_spatial-52a89c4b`. Training took 749.35 seconds;
embedding inference and KMeans took 61.77 seconds. The model has 60,999
parameters. Generated comparisons are under
`outputs/research/held-pavia-20260901/`.

This is a failed held transfer. The learned method loses 0.0650 ARI, 0.0603
NMI, 0.0916 ACC, 0.0523 Macro-F1, and 0.0812 mIoU against raw KMeans. Its
better silhouette and DBI again do not imply better semantic mapping.

Per-class analysis is mixed. The learned representation recovers class 7 with
F1 0.4311 where raw KMeans scores zero, and keeps class 9 near-perfect, but it
misses class 3 entirely and substantially weakens classes 1, 2, 5, and 8.

Protocol conclusion: do not tune the 0.85 threshold on Pavia or rerun variants
selected using these labels. The rule is scene-size dependent at epoch
granularity: one Pavia epoch contains roughly ten times the optimization steps
of Indian Pines and overshoots the target to 0.943. Future development should
monitor usage within epochs in optimizer-step or sample units and validate a
revised rule on a new untouched scene.
