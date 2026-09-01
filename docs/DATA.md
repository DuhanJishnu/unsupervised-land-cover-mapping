# Data and Artifacts

The master scientific context and next-step decisions are maintained in
`docs/RESEARCH_LEDGER.md`.

Large datasets and generated artifacts are intentionally not committed to Git.

## Expected Local Paths

```text
datasets/mat_files/       # Indian Pines and Pavia University .mat files
datasets/csv_files/       # Optional converted CSV files
datas/                    # EnMAP L2A scene folders or compressed archives
processed_data/           # Generated patches, labels, coordinates, embeddings
models/                   # Trained model weights
outputs/                  # Generated plots, maps, metrics, and reports
```

New research baselines write self-contained run directories to
`outputs/research/<run-id>/`, including a manifest with configuration, dataset
hashes, Git commit, executable source-tree hash, metrics, embeddings, and the
cluster map. The source-tree hash includes uncommitted Python/configuration
files, which prevents a dirty working tree from being represented only by its
base commit.

The EnMAP preprocessor now preserves mapping information in:

```text
processed_data/enmap_train_patches.npy
processed_data/enmap_train_coords.npy
processed_data/enmap_train_scene_ids.npy
processed_data/enmap_scenes.json
```

Coordinates are scene-local `(row, column)` indices. A label array must never
be reshaped into a map without using these coordinates and its corresponding
scene ID.

## Dataset Sources

| Dataset | Source | Expected files |
| --- | --- | --- |
| Indian Pines | [Purdue MultiSpec hyperspectral data](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html) | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` |
| Pavia University | [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | `PaviaU.mat`, `PaviaU_gt.mat` |
| EnMAP L2A | EnMAP product exports from DLR/official access portals | `ENMAP*/ *-SPECTRAL_IMAGE.TIF` scene folders or compressed archives |

The benchmark files used for the first real-data validation were downloaded
on 2026-09-01 from the public `gokriznastic/HybridSN` GitHub mirror because the
UPV/EHU file endpoints returned HTTP 403. Dataset identity was checked against
the UPV/EHU catalog's published dimensions, band counts, and class counts. The
local files are ignored by Git; their content hashes are recorded below so a
future download can be verified independently.

| File | Array key and shape | SHA-256 |
| --- | --- | --- |
| `Indian_pines_corrected.mat` | `indian_pines_corrected`, 145 x 145 x 200 | `ec2f8808710919d566f70f0d4aa885aae1ddfd42b734aba71c5e12ca65450939` |
| `Indian_pines_gt.mat` | `indian_pines_gt`, 145 x 145 | `65c4687a8ab04f6da4789799bc3bc4f6e88bccac3ed6a2e6ae367e5e6b9e429c` |
| `PaviaU.mat` | `paviaU`, 610 x 340 x 103 | `28447fa87f7a5797845e9a189c0da85e23b1d06a4ba7361e5ff44efbf834d2fb` |
| `PaviaU_gt.mat` | `paviaU_gt`, 610 x 340 | `23f6a426928f9b32984adffe659e29f554f9fb6c93b5a107528d308d5087a829` |

The Indian Pines cube hash also matches the SHA-256 published by the
`danaroth/indian_pines` dataset card. These hashes establish byte identity, not
license terms; a release should retain attribution to the original sensor and
scene providers.

## Size Expectations

- Indian Pines `.mat` files are small enough for local development.
- Pavia University `.mat` and converted CSV files can be tens to hundreds of MB.
- EnMAP L2A scenes and extracted archives can be multiple GB.
- `processed_data/` can exceed 10 GB because patch tensors are stored as `.npy`.
- `models/` and `outputs/` are reproducible and should be regenerated locally.

To reproduce results, place the datasets in the paths above, install dependencies from `requirements.txt`, and run `python run_pipeline.py` or `python run_enmap_pipeline.py`.
