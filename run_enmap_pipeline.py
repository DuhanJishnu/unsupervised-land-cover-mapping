"""Run the EnMAP-specific unsupervised land-cover mapping pipeline."""

import os
import subprocess
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENMAP_DIR = os.path.join(BASE_DIR, "EnMap")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")
PATCHES_FILE = os.path.join(PROCESSED_DIR, "enmap_train_patches.npy")
ENCODER_FILE = os.path.join(BASE_DIR, "models", "enmap_encoder.pth")

ALL_STAGES = [
    ("EnMAP EDA & Visualization", "01b_enmap_eda.py"),
    ("EnMAP Preprocessing & Sampling", "02b_enmap_preprocessing.py"),
    ("EnMAP CNN Autoencoder Training", "04b_enmap_autoencoder.py"),
    ("EnMAP Clustering & Metrics", "05b_enmap_clustering.py"),
]

SKIP_CONDITIONS = {
    "02b_enmap_preprocessing.py": lambda: os.path.exists(PATCHES_FILE),
    "04b_enmap_autoencoder.py": lambda: os.path.exists(ENCODER_FILE),
}


def run_pipeline(skip_preprocess=False):
    print("=" * 70)
    print("ENMAP DATASET PIPELINE - Full Unsupervised Workflow")
    print("=" * 70)
    print(f"Project directory : {BASE_DIR}")
    print(f"EnMAP scripts     : {ENMAP_DIR}")
    print(f"Patches exist     : {os.path.exists(PATCHES_FILE)}")
    print(f"Encoder exists    : {os.path.exists(ENCODER_FILE)}")
    print()

    t_start = time.time()

    for stage_name, script_name in ALL_STAGES:
        skip_fn = SKIP_CONDITIONS.get(script_name) if skip_preprocess else None
        if skip_fn and skip_fn():
            print(f"SKIPPED (output exists): {stage_name} ({script_name})")
            continue

        print(f"\nSTARTING: {stage_name}")
        print(f"Script : {script_name}")
        print("-" * 50)
        t0 = time.time()

        result = subprocess.run(
            [sys.executable, script_name],
            cwd=ENMAP_DIR,
            capture_output=False
        )

        elapsed = time.time() - t0
        if result.returncode != 0:
            print(f"\nERROR in {script_name} (exit code {result.returncode}).")
            print("Pipeline aborted.")
            return

        print(f"\nCOMPLETED: {script_name} ({elapsed / 60:.1f} min)")
        print("-" * 50)

    t_total = time.time() - t_start
    mins, secs = divmod(int(t_total), 60)
    print("\n" + "=" * 70)
    print(f"ENMAP PIPELINE COMPLETE. Total time: {mins}m {secs}s")
    print("=" * 70)


if __name__ == "__main__":
    skip = "--skip-preprocess" in sys.argv
    run_pipeline(skip_preprocess=skip)
