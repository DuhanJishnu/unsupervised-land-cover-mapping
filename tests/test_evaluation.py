import unittest

try:
    import numpy as np
    from landcover.evaluation import evaluate_clustering, hungarian_match, semantic_diagnostics
    EVALUATION_AVAILABLE = True
except ModuleNotFoundError:
    EVALUATION_AVAILABLE = False


@unittest.skipUnless(EVALUATION_AVAILABLE, "numpy/scipy/scikit-learn are not installed")
class EvaluationTests(unittest.TestCase):
    def test_semantic_diagnostics_report_per_class_failures(self):
        truth = np.array([1, 1, 2, 2])
        clusters = np.array([9, 9, 8, 9])
        diagnostics = semantic_diagnostics(truth, clusters)
        self.assertEqual([row["support"] for row in diagnostics["per_class"]], [2, 2])
        self.assertLess(diagnostics["per_class"][1]["recall"], 1.0)

    def test_hungarian_matching_is_permutation_invariant(self):
        truth = np.array([1, 1, 2, 2, 3, 3])
        clusters = np.array([8, 8, 4, 4, 9, 9])
        mapped, mapping = hungarian_match(truth, clusters)
        np.testing.assert_array_equal(mapped, truth)
        self.assertEqual(set(mapping.values()), {1, 2, 3})

    def test_perfect_permuted_clustering_scores_one(self):
        truth = np.array([1, 1, 2, 2, 3, 3])
        clusters = np.array([8, 8, 4, 4, 9, 9])
        result = evaluate_clustering(truth, clusters, labeled_mask=np.ones(6, dtype=bool))
        for key in ("ari", "nmi", "accuracy", "macro_f1", "miou", "coverage"):
            self.assertAlmostEqual(result[key], 1.0)

    def test_rejections_reduce_coverage_and_accuracy(self):
        truth = np.array([1, 1, 2, 2])
        clusters = np.array([5, -1, 7, -1])
        result = evaluate_clustering(truth, clusters, labeled_mask=np.ones(4, dtype=bool))
        self.assertAlmostEqual(result["coverage"], 0.5)
        self.assertAlmostEqual(result["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
