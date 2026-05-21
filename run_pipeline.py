"""Run the complete Indian Pines and Pavia University pipeline."""

import subprocess
import sys
import time

SCRIPTS = [
    ("Week 1 - Dataset Exploration", "01_dataset_exploration.py"),
    ("Week 2 - Preprocessing Pipeline", "02_preprocessing.py"),
    ("Week 3 - PCA Baseline", "03_pca_baseline.py"),
    ("Week 4 - CNN Autoencoder Training", "04_cnn_autoencoder.py"),
    ("Week 5 - CNN Embedding Extraction & Clustering", "05_cnn_clustering.py"),
    ("Week 6 - Embedding Space Visualization", "06_embedding_analysis.py"),
    ("Week 7 - Spatial Smoothing Refinement", "07_spatial_smoothing.py"),
    ("Week 8 - Final Report Structuring", "08_final_report_assets.py")
]

def run_pipeline():
    print("=" * 70)
    print("STARTING FULL MINOR PROJECT PIPELINE (WEEKS 1-8)")
    print("=" * 70)

    t_start = time.time()

    for stage_name, script_name in SCRIPTS:
        print(f"\n\nSTARTING: {stage_name} ({script_name})")
        print("-" * 50)
        t0 = time.time()

        result = subprocess.run([sys.executable, script_name], capture_output=False)

        if result.returncode != 0:
            print(f"\nERROR in {script_name}. Pipeline aborted.")
            return

        print(f"\nCOMPLETED {script_name} in {time.time() - t0:.1f}s")
        print("-" * 50)

    t_total = time.time() - t_start
    mins = int(t_total // 60)
    secs = int(t_total % 60)

    print("=" * 70)
    print(f"FULL PIPELINE SUCCESSFUL. TOTAL TIME: {mins}m {secs}s")
    print("=" * 70)

if __name__ == "__main__":
    run_pipeline()
