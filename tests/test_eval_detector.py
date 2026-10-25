"""tests/test_eval_detector.py

Unit tests for scripts/eval_detector.py (D2-C3 / Spec 8.5).
Tests greedy matching, IoU computation, recall calculation, and edge cases.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure repo root and scripts/ are importable
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

from eval_detector import compute_iou, greedy_match, evaluate_detections_at_conf


class TestComputeIoU:
    def test_identical_boxes(self):
        b1 = np.array([10.0, 10.0, 50.0, 50.0])
        b2 = np.array([10.0, 10.0, 50.0, 50.0])
        assert pytest.approx(compute_iou(b1, b2), abs=1e-5) == 1.0

    def test_non_overlapping_boxes(self):
        b1 = np.array([0.0, 0.0, 10.0, 10.0])
        b2 = np.array([20.0, 20.0, 30.0, 30.0])
        assert compute_iou(b1, b2) == 0.0

    def test_partial_overlap(self):
        # 10x10 each, overlap is 5x10 = 50 area
        # b1: [0, 0, 10, 10] -> area 100
        # b2: [5, 0, 15, 10] -> area 100
        # intersection: [5, 0, 10, 10] -> area 50
        # union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
        b1 = np.array([0.0, 0.0, 10.0, 10.0])
        b2 = np.array([5.0, 0.0, 15.0, 10.0])
        assert pytest.approx(compute_iou(b1, b2), abs=1e-5) == 1.0 / 3.0


class TestGreedyMatch:
    def test_empty_predictions(self):
        gt_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        pred_boxes = np.empty((0, 4))
        pred_scores = np.empty((0,))
        tp, fp, fn = greedy_match(pred_boxes, pred_scores, gt_boxes)
        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_empty_ground_truth(self):
        gt_boxes = np.empty((0, 4))
        pred_boxes = np.array([[10, 10, 50, 50]])
        pred_scores = np.array([0.9])
        tp, fp, fn = greedy_match(pred_boxes, pred_scores, gt_boxes)
        assert tp == 0
        assert fp == 1
        assert fn == 0

    def test_perfect_matches(self):
        gt_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100]])
        pred_scores = np.array([0.85, 0.95])
        tp, fp, fn = greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5)
        assert tp == 2
        assert fp == 0
        assert fn == 0

    def test_greedy_highest_confidence_priority(self):
        # Two predictions for the same single GT box
        # Pred 1: high conf (0.9), good match (IoU=0.9)
        # Pred 2: lower conf (0.7), also good match (IoU=0.85)
        # Only pred 1 should match, pred 2 should be marked FP
        gt_boxes = np.array([[10.0, 10.0, 50.0, 50.0]])
        pred_boxes = np.array([[10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 50.0, 50.0]])
        pred_scores = np.array([0.90, 0.70])
        tp, fp, fn = greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5)
        assert tp == 1
        assert fp == 1
        assert fn == 0

    def test_manual_5_images_recall(self):
        """Spec 8.5 requirement: recall matches manual calculation on 5 images."""
        # 5 images with known boxes:
        # Img 1: 1 GT, 1 matched Pred -> TP=1, FP=0, FN=0
        # Img 2: 2 GT, 1 matched Pred, 1 missed -> TP=1, FP=0, FN=1
        # Img 3: 1 GT, 0 Pred -> TP=0, FP=0, FN=1
        # Img 4: 2 GT, 2 matched Pred, 1 false alarm -> TP=2, FP=1, FN=0
        # Img 5: 0 GT, 1 false alarm -> TP=0, FP=1, FN=0
        # Total GT = 1 + 2 + 1 + 2 + 0 = 6
        # Total TP = 1 + 1 + 0 + 2 + 0 = 4
        # Total FP = 0 + 0 + 0 + 1 + 1 = 2
        # Total FN = 0 + 1 + 1 + 0 + 0 = 2
        # Recall = 4 / 6 = 2/3 ~ 0.6667
        # Precision = 4 / (4 + 2) = 4/6 ~ 0.6667
        predictions = {
            "img1": (np.array([[10, 10, 50, 50]]), np.array([0.8])),
            "img2": (np.array([[10, 10, 50, 50]]), np.array([0.7])),
            "img3": (np.empty((0, 4)), np.empty((0,))),
            "img4": (np.array([[10, 10, 50, 50], [60, 60, 100, 100], [200, 200, 250, 250]]), np.array([0.9, 0.85, 0.3])),
            "img5": (np.array([[50, 50, 80, 80]]), np.array([0.6])),
        }
        ground_truth = {
            "img1": np.array([[10, 10, 50, 50]]),
            "img2": np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
            "img3": np.array([[10, 10, 50, 50]]),
            "img4": np.array([[10, 10, 50, 50], [60, 60, 100, 100]]),
            "img5": np.empty((0, 4)),
        }

        # Test at conf_thresh = 0.25 (all preds included)
        metrics = evaluate_detections_at_conf(predictions, ground_truth, conf_thresh=0.25, iou_thresh=0.5)
        assert metrics["tp"] == 4
        assert metrics["fp"] == 2
        assert metrics["fn"] == 2
        assert pytest.approx(metrics["recall"], abs=1e-4) == 4.0 / 6.0
        assert pytest.approx(metrics["precision"], abs=1e-4) == 4.0 / 6.0

        # Test at conf_thresh = 0.5 (pred with conf 0.3 on img4 is filtered out)
        # img4 now has 2 preds (0.9, 0.85), both match GT -> TP=2, FP=0
        # Total TP = 4, Total FP = 1 (on img5), Total FN = 2
        metrics_high_conf = evaluate_detections_at_conf(predictions, ground_truth, conf_thresh=0.5, iou_thresh=0.5)
        assert metrics_high_conf["tp"] == 4
        assert metrics_high_conf["fp"] == 1
        assert metrics_high_conf["fn"] == 2
        assert pytest.approx(metrics_high_conf["precision"], abs=1e-4) == 4.0 / 5.0
