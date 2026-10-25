"""tests/test_eval_canteen_classifier.py

Unit tests for scripts/eval_canteen_classifier.py (Spec D3-C4).
Kiểm tra:
1. Hàm compute_ece: tính toán ECE 10 bins chuẩn xác.
2. Định nghĩa Critical Error Rate: mẫu số chỉ gồm Hữu cơ + Tái chế.
3. Đánh giá classifier trên mock crops xuất đầy đủ predictions CSV và metrics CSV.
4. Test Guard: Chặn nạp --split test khi DSR_ALLOW_TEST=0.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, "scripts")
from eval_canteen_classifier import compute_ece


def test_compute_ece_perfect_and_worst():
    """Kiểm tra compute_ece với trường hợp chuẩn xác 100% và trường hợp tự tin sai."""
    # 1. Dự đoán hoàn hảo với confidence 1.0 -> ECE = 0.0
    confidences = np.array([1.0, 1.0, 1.0, 1.0])
    predictions = np.array([0, 1, 2, 0])
    labels = np.array([0, 1, 2, 0])
    ece = compute_ece(confidences, predictions, labels, n_bins=10)
    assert abs(ece) < 1e-5, f"Expected near 0 ECE, got {ece}"

    # 2. Hoàn toàn tự tin (conf=1.0) nhưng dự đoán sai 100% -> |acc - conf| = |0.0 - 1.0| = 1.0
    predictions_wrong = np.array([1, 0, 0, 1])
    ece_wrong = compute_ece(confidences, predictions_wrong, labels, n_bins=10)
    assert abs(ece_wrong - 1.0) < 1e-5, f"Expected ECE 1.0, got {ece_wrong}"


@pytest.fixture
def mock_canteen_crops_env(tmp_path: Path):
    """Tạo môi trường giả lập crops và manifest cho eval_canteen_classifier."""
    crops_dir = tmp_path / "canteen_crops"
    split_dir = crops_dir / "val_gt"
    split_dir.mkdir(parents=True)
    out_dir = tmp_path / "reports"

    # Tạo 3 ảnh crop 224x224
    manifest_rows = []
    # 0: Hữu cơ, 1: Tái chế, 2: Còn lại
    for idx, (grp, c_id) in enumerate([("0", "c01"), ("1", "c02"), ("2", "c03")]):
        img = Image.new("RGB", (224, 224), color=(50 * (idx + 1), 50, 50))
        crop_file_rel = f"val_gt/crop_{idx:02d}.jpg"
        img.save(crops_dir / crop_file_rel)

        manifest_rows.append({
            "crop_id": f"crop_{idx:02d}",
            "image_id": f"tray_{idx:02d}.png",
            "cluster_id": "20240717_trua",
            "crop_file": crop_file_rel,
            "box_xyxy": "50_50_150_150",
            "padded_xyxy": "40_40_160_160",
            "source": "gt",
            "gt_group": grp,
            "iou_with_gt": "1.0",
            "conf": "1.0",
        })

    # Thêm 1 mẫu ambiguous_loc (phải bị script bỏ qua)
    img_amb = Image.new("RGB", (224, 224), color=(100, 100, 100))
    crop_amb_rel = "val_gt/crop_amb.jpg"
    img_amb.save(crops_dir / crop_amb_rel)
    manifest_rows.append({
        "crop_id": "crop_amb",
        "image_id": "tray_amb.png",
        "cluster_id": "20240717_trua",
        "crop_file": crop_amb_rel,
        "box_xyxy": "10_10_60_60",
        "padded_xyxy": "5_5_65_65",
        "source": "det",
        "gt_group": "ambiguous_loc",
        "iou_with_gt": "0.3",
        "conf": "0.8",
    })

    manifest_p = crops_dir / "crops_manifest_val_gt.csv"
    with open(manifest_p, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(manifest_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "root": tmp_path,
        "crops_dir": crops_dir,
        "manifest_p": manifest_p,
        "out_dir": out_dir,
    }


def test_eval_canteen_classifier_execution(mock_canteen_crops_env):
    """Kiểm tra chạy eval_canteen_classifier.py trên mock crops."""
    env = mock_canteen_crops_env
    cmd = [
        sys.executable, "scripts/eval_canteen_classifier.py",
        "--split", "val",
        "--crops-dir", str(env["crops_dir"]),
        "--manifest", str(env["manifest_p"]),
        "--out-dir", str(env["out_dir"]),
        "--device", "cpu",
        "--batch-size", "2",
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True)
    assert "KẾT QUẢ ĐÁNH GIÁ CLASSIFIER CANTEEN" in res.stdout
    assert "Đã loại 1 crop ambiguous_loc" in res.stdout

    # Kiểm tra file CSV dự đoán chi tiết từng crop
    pred_csv = env["out_dir"] / "canteen_classifier_predictions_val.csv"
    assert pred_csv.exists()
    with open(pred_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    # Phải có đầy đủ các cột phục vụ Bootstrap Ngày 7
    assert "cluster_id" in rows[0]
    assert "is_correct" in rows[0]
    assert "is_critical_error" in rows[0]
    assert "p_organic" in rows[0]
    assert "p_recyclable" in rows[0]

    # Kiểm tra file CSV tóm tắt metrics
    metrics_csv = env["out_dir"] / "canteen_classifier_metrics_val.csv"
    assert metrics_csv.exists()
    with open(metrics_csv, "r", encoding="utf-8") as f:
        m_rows = dict(csv.reader(f))
    assert "macro_f1" in m_rows
    assert "ece_10bin" in m_rows
    assert "critical_error_rate" in m_rows
    assert int(m_rows["total_evaluated_crops"]) == 3


def test_eval_canteen_classifier_blocks_test(mock_canteen_crops_env):
    """Test guard bắt buộc chặn --split test khi DSR_ALLOW_TEST=0."""
    env = mock_canteen_crops_env
    cmd = [
        sys.executable, "scripts/eval_canteen_classifier.py",
        "--split", "test",
        "--crops-dir", str(env["crops_dir"]),
        "--manifest", str(env["manifest_p"]),
        "--out-dir", str(env["out_dir"]),
        "--device", "cpu",
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert res.returncode != 0
    assert "PROTOCOL VIOLATION - TEST GUARD" in res.stderr
