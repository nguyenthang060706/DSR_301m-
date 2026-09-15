from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from dsr.data import compute_class_weights
from dsr.metrics import classification_metrics


class PerClassMetricsTests(unittest.TestCase):
    def test_minority_class_failure_is_visible_in_per_class_f1(self) -> None:
        """Macro-F1 alone can look fine while a minority class is completely
        mispredicted — this test locks in that the per-class breakdown catches it."""
        y_true = [0] * 5 + [1] * 5 + [2] * 2
        y_pred = [0] * 5 + [1] * 5 + [0, 1]  # class 2 always mispredicted

        metrics = classification_metrics(y_true, y_pred, num_classes=3)

        self.assertEqual(metrics["per_class_f1"][2], 0.0)
        self.assertEqual(metrics["per_class_support"], [5, 5, 2])
        # macro_f1 stays moderately high even though class 2 is a total failure
        self.assertGreater(metrics["macro_f1"], 0.5)

    def test_per_class_f1_perfect_predictions(self) -> None:
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]
        metrics = classification_metrics(y_true, y_pred, num_classes=3)
        self.assertEqual(metrics["per_class_f1"], [1.0, 1.0, 1.0])
        self.assertEqual(metrics["macro_f1"], 1.0)


class ComputeClassWeightsTests(unittest.TestCase):
    def _write_fake_split(self, split_dir: Path, counts: dict[str, int]) -> None:
        split_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {"path": f"{class_name}/{i}.jpg", "class": class_name}
            for class_name, n in counts.items()
            for i in range(n)
        ]
        with (split_dir / "trashnet_cv_folds.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "class"])
            writer.writeheader()
            writer.writerows(rows)

    def test_minority_class_gets_largest_weight(self) -> None:
        counts = {
            "cardboard": 340,
            "glass": 420,
            "metal": 350,
            "paper": 500,
            "plastic": 410,
            "trash": 116,
        }
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            self._write_fake_split(split_dir, counts)
            config = {"classes": list(counts.keys()), "data": {"split_dir": str(split_dir)}}

            weights = compute_class_weights(config)

            self.assertEqual(len(weights), len(counts))
            # trash has the fewest examples -> must get the largest weight
            self.assertEqual(weights[-1], max(weights))
            # inverse-frequency: weight * count should be ~constant (== total / num_classes)
            total = sum(counts.values())
            expected_product = total / len(counts)
            for weight, count in zip(weights, counts.values()):
                self.assertAlmostEqual(weight * count, expected_product, places=6)

    def test_raises_on_zero_count_class(self) -> None:
        counts = {"cardboard": 10, "glass": 0}
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp)
            self._write_fake_split(split_dir, counts)
            config = {"classes": list(counts.keys()), "data": {"split_dir": str(split_dir)}}
            with self.assertRaises(ValueError):
                compute_class_weights(config)


if __name__ == "__main__":
    unittest.main()
