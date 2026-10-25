"""tests/test_synthesize_trays.py

Unit tests for scripts/synthesize_trays.py (D2-C1 / Spec 8.4).
Verifies:
  1. Box IoU computation
  2. Determinism: 2 runs with same seed produce identical images/labels
  3. No public_dev.csv images in training cutouts
  4. Label formats: class-agnostic (class 0) and 3cls (classes in {0, 1, 2})
"""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Ensure repo root and scripts/ are importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

from synthesize_trays import (
    compute_box_iou,
    generate_synthetic_tray,
    load_public_train,
)


class TestSynthesizeTraysIoU:
    def test_exact_overlap(self):
        box1 = [10.0, 10.0, 50.0, 50.0]
        box2 = [10.0, 10.0, 50.0, 50.0]
        assert pytest.approx(compute_box_iou(box1, box2), abs=1e-5) == 1.0

    def test_disjoint_boxes(self):
        box1 = [0.0, 0.0, 10.0, 10.0]
        box2 = [20.0, 20.0, 30.0, 30.0]
        assert compute_box_iou(box1, box2) == 0.0


class TestSynthesizeTraysGeneration:
    @pytest.fixture
    def dummy_background(self):
        # 640x640 gray background
        return Image.new("RGB", (640, 640), color=(180, 180, 180))

    @pytest.fixture
    def dummy_cutouts(self):
        # Create 3 RGBA cutouts: red (fine 1, coarse 0), green (fine 5, coarse 1), blue (fine 10, coarse 2)
        cutouts = []
        for color, fine_id, coarse_id in [
            ((220, 30, 30, 255), 1, 0),
            ((30, 220, 30, 255), 5, 1),
            ((30, 30, 220, 255), 10, 2),
        ]:
            img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
            # Put a solid circle in the middle
            for x in range(25, 75):
                for y in range(25, 75):
                    img.putpixel((x, y), color)
            cutouts.append((img, fine_id, coarse_id))
        return cutouts

    def test_seed_determinism(self, dummy_background, dummy_cutouts):
        """Spec 8.4: 2 runs with same seed produce identical outputs."""
        rng1 = random.Random(20261012)
        img1, agn1, cls1 = generate_synthetic_tray(
            bg_img=dummy_background,
            cutout_list=dummy_cutouts,
            output_size=(320, 320),
            num_objects=3,
            rng=rng1,
        )

        rng2 = random.Random(20261012)
        img2, agn2, cls2 = generate_synthetic_tray(
            bg_img=dummy_background,
            cutout_list=dummy_cutouts,
            output_size=(320, 320),
            num_objects=3,
            rng=rng2,
        )

        # Labels must be exactly identical
        assert agn1 == agn2
        assert cls1 == cls2

        # Images must be pixel-for-pixel identical
        arr1 = np.array(img1)
        arr2 = np.array(img2)
        assert np.array_equal(arr1, arr2)

    def test_label_format_and_classes(self, dummy_background, dummy_cutouts):
        """Agnostic labels must have class 0; 3cls labels must have class in {0, 1, 2}."""
        rng = random.Random(42)
        _, agnostic_labels, threecls_labels = generate_synthetic_tray(
            bg_img=dummy_background,
            cutout_list=dummy_cutouts,
            output_size=(320, 320),
            num_objects=2,
            rng=rng,
        )

        assert len(agnostic_labels) == len(threecls_labels)
        assert len(agnostic_labels) > 0

        for line in agnostic_labels:
            parts = line.strip().split()
            assert len(parts) == 5
            cid = int(parts[0])
            assert cid == 0  # Class-agnostic is strictly 0
            xc, yc, w, h = [float(x) for x in parts[1:]]
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 < w <= 1.0
            assert 0.0 < h <= 1.0

        for line in threecls_labels:
            parts = line.strip().split()
            assert len(parts) == 5
            cid = int(parts[0])
            assert cid in {0, 1, 2}  # 3 coarse classes


class TestPublicSplitLeakagePrevention:
    def test_load_public_train_filters_dev(self, tmp_path):
        """Spec 8.4: Verify that public_dev images are never included in train set."""
        train_csv = tmp_path / "public_train.csv"
        with open(train_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerow(["images/train_obj_01.jpg", 1])
            writer.writerow(["images/train_obj_02.jpg", 2])

        dev_csv = tmp_path / "public_dev.csv"
        with open(dev_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerow(["images/dev_obj_99.jpg", 3])

        train_stems = load_public_train(train_csv)
        dev_stems = load_public_train(dev_csv)

        assert "train_obj_01" in train_stems
        assert "train_obj_02" in train_stems
        assert "dev_obj_99" not in train_stems
        assert train_stems.isdisjoint(dev_stems)
