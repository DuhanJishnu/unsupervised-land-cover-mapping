import unittest

try:
    import numpy as np
    import torch

    from landcover.spatial import local_spatial_graph
    from landcover.torch_data import SpatialTileBatchSampler

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class SpatialTrainingTests(unittest.TestCase):
    def test_sampler_emits_every_index_once(self):
        rows, cols = np.indices((12, 12))
        coordinates = np.column_stack((rows.ravel(), cols.ravel()))
        sampler = SpatialTileBatchSampler(
            coordinates,
            batch_size=32,
            tile_size=6,
            spatial_group_size=8,
            seed=9,
        )
        batches = list(sampler)
        emitted = [index for batch in batches for index in batch]
        self.assertEqual(sorted(emitted), list(range(len(coordinates))))
        self.assertTrue(all(len(batch) <= 32 for batch in batches))

    def test_local_graph_uses_only_neighbors(self):
        coordinates = torch.tensor([[0, 0], [0, 1], [1, 1], [4, 4]])
        spectra = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
            dtype=torch.float32,
        )
        edges, weights = local_spatial_graph(
            coordinates, spectra, connectivity=4, minimum_weight=0.0
        )
        self.assertEqual({tuple(edge) for edge in edges.tolist()}, {(0, 1), (1, 2)})
        self.assertTrue(torch.all(weights > 0))


if __name__ == "__main__":
    unittest.main()
