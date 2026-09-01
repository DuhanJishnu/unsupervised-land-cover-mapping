import unittest

import numpy as np

from landcover.config import DatasetSpec
from landcover.data import Scene, extract_patch, fit_normalization, normalize_scene


class DataTests(unittest.TestCase):
    def setUp(self):
        cube = np.arange(5 * 6 * 4, dtype=np.float32).reshape(5, 6, 4)
        gt = np.zeros((5, 6), dtype=np.int32)
        valid = np.ones((5, 6), dtype=bool)
        spec = DatasetSpec("x", "Synthetic", "", "", "", "", 2, "synthetic")
        self.scene = Scene(cube, gt, valid, spec)

    def test_patch_preserves_center_and_shape(self):
        patch = extract_patch(self.scene.cube, 0, 0, patch_size=3)
        self.assertEqual(patch.shape, (4, 3, 3))
        np.testing.assert_array_equal(patch[:, 1, 1], self.scene.cube[0, 0])

    def test_robust_normalization_is_finite_and_bounded(self):
        normalized = normalize_scene(self.scene, "robust")
        self.assertTrue(np.isfinite(normalized.cube).all())
        self.assertGreaterEqual(float(normalized.cube.min()), 0.0)
        self.assertLessEqual(float(normalized.cube.max()), 1.0)

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_normalization(self.scene.cube, np.zeros((5, 6), dtype=bool))


if __name__ == "__main__":
    unittest.main()

