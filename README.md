# Unsupervised Hyperspectral Land-Cover Mapping

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

This project studies unsupervised land-cover mapping from hyperspectral imagery using PCA baselines, CNN autoencoder embeddings, clustering, and spatial post-processing. Experiments cover Indian Pines, Pavia University, and EnMAP L2A scenes, with an additional HyperAttnRes transformer extension for learned spectral-spatial representations.

> **Research status:** the legacy week-based pipeline is being rebuilt after a
> repository-wide reproducibility and methodology audit. Existing headline
> numbers are historical and should not yet be treated as paper results. See
> the comprehensive [Master Research Ledger](docs/RESEARCH_LEDGER.md), plus the
> focused [Project Audit](docs/PROJECT_AUDIT.md),
> [Research Plan](docs/RESEARCH_PLAN.md), and
> [Experiment Log](docs/EXPERIMENT_LOG.md).

## Current Development Results

The first controlled Indian Pines pilot uses one seed and three epochs. It is a
model-selection diagnostic, not a final paper table.

| Method | ARI | NMI | ACC | Macro-F1 | mIoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw spectrum + KMeans | 0.2238 | 0.4423 | 0.3751 | 0.3527 | 0.2431 |
| Reconstruction-only embedding | 0.2264 | 0.4196 | 0.3366 | 0.1837 | 0.1179 |
| Full objective without spatial term | 0.2634 | 0.4193 | 0.4057 | 0.2683 | 0.1867 |
| Full objective | 0.2635 | 0.4189 | 0.4116 | 0.2810 | 0.1951 |

The learned objective currently improves ARI and matched accuracy, but not NMI,
Macro-F1, or mIoU. See the experiment log for failure analysis and limitations.

The first frozen Pavia transfer has now been completed. It stopped after one
epoch and underperformed raw/PCA KMeans on all primary semantic metrics. This
negative held result is retained in the master ledger; Pavia will not be used
for post-hoc stopping-rule tuning.

## Pipeline

![Pipeline_Diagram](/docs/image.png)
## Repository Structure

```text
.
├── 01_dataset_exploration.py
├── 02_preprocessing.py
├── 03_pca_baseline.py
├── 04_cnn_autoencoder.py
├── 05_cnn_clustering.py
├── 06_embedding_analysis.py
├── 07_spatial_smoothing.py
├── 08_final_report_assets.py
├── 10_analysis_plots.py
├── Advanced/                  # HyperAttnRes and transformer ablations
├── EnMap/                     # EnMAP-specific EDA, preprocessing, and clustering
├── datasets/mat_to_csv.py
├── docs/DATA.md
├── notebooks/                 # Exploratory notebooks
├── config.py                  # Shared labels, colors, plotting style
├── preprocessing.py           # Reusable preprocessing functions
├── run_pipeline.py
└── run_enmap_pipeline.py
```

Large raw imagery, processed arrays, trained weights, and generated outputs are intentionally excluded from Git. See [docs/DATA.md](docs/DATA.md) for the expected local data layout.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

For exact replication of the tested macOS arm64/Python 3.9 environment, use
`requirements-lock.txt`. The looser `requirements.txt` remains the portable
cross-platform specification.

## Usage

Run a manifest-producing research baseline (recommended):

```bash
python research_pipeline.py baseline --dataset ip --representation pca --clusterer kmeans
python research_pipeline.py baseline --dataset pu --representation pca --clusterer kmeans
```

Each run writes its configuration, dataset hashes, Git commit, metrics, cluster
map, and embeddings under `outputs/research/<run-id>/`.

Train the clustering-aligned model with the default five seeds:

```bash
python research_pipeline.py train --dataset ip --ablation full
python research_pipeline.py train --dataset pu --ablation full
```

Run the development-selected label-free stopping protocol. `scheduler-epochs`
fixes the learning-rate trajectory even when different scenes stop at different
epochs:

```bash
python research_pipeline.py train --dataset pu --ablation no_spatial \
  --epochs 8 --scheduler-epochs 8 --early-stop-usage-entropy 0.85
```

Use `--evaluation-every 1` only on a declared development scene; it reads
ground truth after checkpoints and would bias a held-scene result.

Estimate the cluster count without labels instead of using oracle `k`:

```bash
python research_pipeline.py train --dataset pu --k-range 4 14
```

Run a controlled objective ablation:

```bash
python research_pipeline.py train --dataset pu --ablation no_spatial --seeds 42 43 44 45 46
```

Available ablations are `reconstruction`, `no_spectral_angle`, `no_view`,
`no_prototype`, `no_spatial`, and `full`.

Use internal overclustering without changing final evaluation `k`:

```bash
python research_pipeline.py train --dataset ip --ablation full --prototype-multiplier 2
```

Generate JSON, CSV, per-class CSV, and paper-ready Markdown from completed runs:

```bash
python research_pipeline.py compare \
  --runs outputs/research/<baseline-run> outputs/research/<training-group> \
  --output-dir outputs/research/comparison
```

The original week-based workflows remain available during the migration:

Run the Indian Pines and Pavia University workflow:

```bash
python run_pipeline.py
```

Run the EnMAP workflow:

```bash
python run_enmap_pipeline.py
```

Outputs are written to `outputs/`, trained models to `models/`, and generated NumPy arrays to `processed_data/`.

## Methodology

The rebuilt research pipeline robustly normalizes valid pixels, uses all 200
bands of the already-corrected Indian Pines cube, extracts reflect-padded
patches lazily, learns normalized spectral-spatial embeddings, and evaluates
clusters with permutation-safe semantic metrics. The original majority filter
and advanced architectures remain comparison candidates rather than assumed
improvements.

## Advanced: HyperAttnRes

The `Advanced/` module implements a transformer-style spectral-spatial autoencoder with block attention residuals. It compares HyperAttnRes against a standard transformer autoencoder and the CNN autoencoder using clustering metrics, embedding plots, and ablation visualizations.

## Results and Visualizations

Generated figures are not committed because they are reproducible artifacts. After running the pipeline, inspect:

```text
outputs/week1/   # dataset exploration
outputs/week3/   # PCA baseline metrics and plots
outputs/week5/   # CNN clustering metrics
outputs/week6/   # embedding visualizations
outputs/week7/   # spatial smoothing outputs
outputs/week8/   # summary tables and advanced analysis plots
```

## References

- Indian Pines dataset: Purdue MultiSpec hyperspectral data page.
- Pavia University dataset: Hyperspectral Remote Sensing Scenes, Computational Intelligence Group, University of the Basque Country.
- EnMAP mission and L2A products: German Aerospace Center (DLR) EnMAP documentation.
