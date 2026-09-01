import tempfile
import unittest
from pathlib import Path

import numpy as np

from landcover.config import DATASETS, ExperimentConfig
from landcover.data import Scene
from landcover.training import TrainingConfig, train_and_evaluate


class TrainingIntegrationTests(unittest.TestCase):
    def test_one_epoch_creates_complete_run(self):
        rng = np.random.default_rng(12)
        cube = rng.random((10, 10, 8), dtype=np.float32)
        gt = np.ones((10, 10), dtype=np.int32)
        gt[:, 5:] = 2
        # Give the two synthetic regions a learnable spectral difference.
        cube[:, 5:, 4:] += 1.0
        scene = Scene(cube, gt, np.ones((10, 10), dtype=bool), DATASETS["ip"])
        experiment = ExperimentConfig(
            dataset="ip",
            seed=3,
            patch_size=5,
            embedding_dim=8,
            cluster_count_protocol="explicit",
            n_clusters=2,
        )
        training = TrainingConfig(
            epochs=3,
            scheduler_epochs=3,
            batch_size=32,
            tile_size=5,
            spatial_group_size=8,
            checkpoint_every=1,
            evaluation_every=1,
            early_stop_usage_entropy=0.1,
            device="cpu",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            spec = DATASETS["ip"]
            (data_dir / spec.data_file).write_bytes(b"synthetic cube")
            (data_dir / spec.gt_file).write_bytes(b"synthetic labels")
            run_dir = root / "run"
            result = train_and_evaluate(
                scene, experiment, training, run_dir, data_dir=data_dir
            )

            self.assertEqual(result["run_id"], "run")
            for filename in (
                "checkpoint.pt",
                "checkpoint_epoch_001.pt",
                "model.pt",
                "history.json",
                "normalization.npz",
                "embeddings.npy",
                "coordinates.npy",
                "cluster_map.npy",
                "semantic_diagnostics.json",
                "manifest.json",
            ):
                self.assertTrue((run_dir / filename).exists(), filename)
            self.assertEqual(np.load(run_dir / "embeddings.npy").shape, (100, 8))
            self.assertIn("ari", result["metrics"])
            self.assertEqual(result["metrics"]["completed_epochs"], 1)
            history = __import__("json").loads((run_dir / "history.json").read_text())
            self.assertIn("probe_ari", history[-1])


if __name__ == "__main__":
    unittest.main()
