# Data and Artifacts

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

## Dataset Sources

| Dataset | Source | Expected files |
| --- | --- | --- |
| Indian Pines | [Purdue MultiSpec hyperspectral data](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html) | `Indian_pines_corrected.mat`, `Indian_pines_gt.mat` |
| Pavia University | [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes) | `PaviaU.mat`, `PaviaU_gt.mat` |
| EnMAP L2A | EnMAP product exports from DLR/official access portals | `ENMAP*/ *-SPECTRAL_IMAGE.TIF` scene folders or compressed archives |

## Size Expectations

- Indian Pines `.mat` files are small enough for local development.
- Pavia University `.mat` and converted CSV files can be tens to hundreds of MB.
- EnMAP L2A scenes and extracted archives can be multiple GB.
- `processed_data/` can exceed 10 GB because patch tensors are stored as `.npy`.
- `models/` and `outputs/` are reproducible and should be regenerated locally.

To reproduce results, place the datasets in the paths above, install dependencies from `requirements.txt`, and run `python run_pipeline.py` or `python run_enmap_pipeline.py`.
