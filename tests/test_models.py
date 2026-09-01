import unittest

try:
    import torch
    from landcover.models import (
        ClusteringAlignedModel,
        apply_band_mask,
        clustering_aligned_loss,
        random_band_mask,
        sinkhorn_balanced_assignments,
    )
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_sinkhorn_targets_are_normalized_and_balanced(self):
        torch.manual_seed(4)
        embeddings = torch.nn.functional.normalize(torch.randn(128, 32), dim=1)
        prototypes = torch.nn.functional.normalize(torch.randn(8, 32), dim=1)
        targets = sinkhorn_balanced_assignments(embeddings @ prototypes.t())
        self.assertTrue(torch.allclose(targets.sum(1), torch.ones(128), atol=1e-4))
        expected = torch.full((8,), 1.0 / 8.0)
        self.assertTrue(torch.allclose(targets.mean(0), expected, atol=1e-3))

    def test_forward_and_loss_are_finite(self):
        torch.manual_seed(7)
        model = ClusteringAlignedModel(in_bands=20, embedding_dim=16, n_prototypes=4)
        patches = torch.rand(8, 20, 7, 7)
        mask = random_band_mask(8, 20, 0.3, patches.device)
        first = model(apply_band_mask(patches, mask))
        second = model(patches + 0.01 * torch.randn_like(patches))
        center = patches[:, :, 3, 3]
        edges = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        total, terms = clustering_aligned_loss(first, second, center, mask, edges)
        self.assertEqual(first.embedding.shape, (8, 16))
        self.assertEqual(first.reconstruction.shape, (8, 20))
        self.assertEqual(first.prototype_logits.shape, (8, 4))
        self.assertTrue(torch.isfinite(total))
        self.assertEqual(set(terms), {"masked", "spectral_angle", "view", "prototype", "spatial"})


if __name__ == "__main__":
    unittest.main()
