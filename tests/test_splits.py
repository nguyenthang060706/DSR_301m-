from pathlib import Path
import csv
import unittest


class SplitTests(unittest.TestCase):
    def test_public_split_csvs(self) -> None:
        train_file = Path("data/splits/public_train.csv")
        dev_file = Path("data/splits/public_dev.csv")
        split_file = Path("data/splits/public_split.csv")

        self.assertTrue(train_file.exists(), "public_train.csv missing")
        self.assertTrue(dev_file.exists(), "public_dev.csv missing")
        self.assertTrue(split_file.exists(), "public_split.csv missing")

        with train_file.open("r", encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with dev_file.open("r", encoding="utf-8") as f:
            dev_rows = list(csv.DictReader(f))

        self.assertEqual(len(train_rows), 12750)
        self.assertEqual(len(dev_rows), 2250)

        # Leakage check
        train_paths = {r["path"] for r in train_rows}
        dev_paths = {r["path"] for r in dev_rows}
        self.assertEqual(len(train_paths & dev_paths), 0, "No overlap between train and dev")

        # Class coverage
        train_classes = {r["class"] for r in train_rows}
        dev_classes = {r["class"] for r in dev_rows}
        self.assertEqual(len(train_classes), 30)
        self.assertEqual(len(dev_classes), 30)


if __name__ == "__main__":
    unittest.main()
