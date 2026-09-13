from __future__ import annotations

import unittest

from scripts.make_trashnet_splits import assign_stratified_folds, stratified_holdout


class SplitTests(unittest.TestCase):
    def test_stratified_holdout_keeps_each_class_represented(self) -> None:
        rows = [
            {"path": f"{class_name}/{i}.jpg", "class": class_name}
            for class_name in ("cardboard", "glass", "metal", "paper", "plastic", "trash")
            for i in range(10)
        ]

        holdout, cv_pool = stratified_holdout(rows, holdout_ratio=0.15, seed=1)

        self.assertEqual(len(holdout), 12)
        self.assertEqual(len(cv_pool), 48)
        self.assertEqual(
            {row["class"] for row in holdout},
            {"cardboard", "glass", "metal", "paper", "plastic", "trash"},
        )

    def test_assign_stratified_folds_is_deterministic(self) -> None:
        rows = [
            {"path": f"{class_name}/{i}.jpg", "class": class_name}
            for class_name in ("cardboard", "glass", "metal", "paper", "plastic", "trash")
            for i in range(10)
        ]

        first = assign_stratified_folds(rows, folds=5, seed=20260913)
        second = assign_stratified_folds(rows, folds=5, seed=20260913)

        self.assertEqual(first, second)
        self.assertEqual({row["fold"] for row in first}, {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
