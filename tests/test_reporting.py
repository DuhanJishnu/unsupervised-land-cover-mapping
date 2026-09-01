import json
import tempfile
import unittest
from pathlib import Path

from landcover.reporting import write_comparison


class ReportingTests(unittest.TestCase):
    def test_training_summary_becomes_three_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "group"
            seed = run / "seed-42"
            seed.mkdir(parents=True)
            (run / "summary.json").write_text(
                json.dumps({"summary": {"ari": {"mean": 0.25, "std": 0.0, "n": 1}}})
            )
            (seed / "manifest.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "experiment": {"dataset": "ip", "cluster_count_protocol": "oracle"},
                            "training": {"ablation": "full", "epochs": 3},
                        }
                    }
                )
            )
            (seed / "history.json").write_text(
                json.dumps([{"prototype_usage_entropy": 0.9}])
            )
            output = root / "comparison"
            rows = write_comparison([run], output)
            self.assertEqual(rows[0]["ari"], 0.25)
            self.assertEqual(rows[0]["prototype_usage_entropy"], 0.9)
            for filename in (
                "comparison.json",
                "comparison.csv",
                "comparison.md",
                "per_class.csv",
                "learning_curves.csv",
            ):
                self.assertTrue((output / filename).exists())


if __name__ == "__main__":
    unittest.main()
