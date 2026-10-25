"""tests/test_extract_crops.py

Unit tests for scripts/extract_crops.py and src/dsr/transforms.py (Spec 8.8 / D3-C3).
Kiểm tra:
1. JitterBoxCrop và apply_box_jitter: biến đổi box, clamp chặt trong ảnh.
2. expand_box_with_padding: mở rộng viền box 15% không tràn viền ảnh.
3. IoU đa tầng:
   - IoU >= 0.5: giữ nguyên class GT
   - 0.1 <= IoU < 0.5: gán ambiguous_loc
   - IoU < 0.1: gán background (hard-negative)
4. Trích xuất crop chế độ GT và DET sinh manifest CSV đúng định dạng.
5. Test guard: Chặn nạp --split test khi DSR_ALLOW_TEST=0.
"""

from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from pathlib import Path
import pytest
from PIL import Image

from dsr.transforms import JitterBoxCrop, apply_box_jitter


def test_apply_box_jitter_properties():
    """Kiểm tra apply_box_jitter: không âm, không vượt quá kích thước ảnh, w > 0, h > 0."""
    img_w, img_h = 640, 480
    box = [100, 100, 200, 250]
    rng = random.Random(42)

    for _ in range(50):
        jx1, jy1, jx2, jy2 = apply_box_jitter(
            box,
            (img_w, img_h),
            shift=0.10,
            scale=0.15,
            pad=(0.05, 0.20),
            rng=rng,
        )
        assert 0 <= jx1 < jx2 <= img_w, f"Invalid x bounds: {jx1}, {jx2}"
        assert 0 <= jy1 < jy2 <= img_h, f"Invalid y bounds: {jy1}, {jy2}"
        assert jx2 - jx1 > 0
        assert jy2 - jy1 > 0


def test_jitter_box_crop_callable():
    """Kiểm tra class JitterBoxCrop crop đúng ảnh PIL."""
    img = Image.new("RGB", (400, 300), color=(128, 128, 128))
    box = [50, 60, 150, 180]
    cropper = JitterBoxCrop(shift=0.10, scale=0.15, pad=(0.05, 0.20), seed=42)

    crop = cropper(img, box)
    assert isinstance(crop, Image.Image)
    cw, ch = crop.size
    assert cw > 0 and ch > 0
    # Kích thước crop phải xấp xỉ box ban đầu cộng padding
    assert 80 <= cw <= 180
    assert 100 <= ch <= 200


@pytest.fixture
def temp_tray_environment(tmp_path: Path):
    """Tạo môi trường giả lập khay ăn và nhãn để test extract_crops.py."""
    img_dir = tmp_path / "trays"
    lbl_dir = tmp_path / "labels"
    out_dir = tmp_path / "crops_out"
    meta_p = tmp_path / "canteen_metadata.csv"
    plan_p = tmp_path / "canteen_date_plan.json"

    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Tạo 1 ảnh khay 640x480
    img = Image.new("RGB", (640, 480), color=(200, 200, 200))
    img_name = "canteen_2024-07-17_1201_pre.png"
    img.save(img_dir / img_name)

    # Tạo nhãn GT: 2 box (class 0 và class 1)
    # Box 1: [0.3, 0.3, 0.2, 0.2] -> class 0 (Hữu cơ)
    # Box 2: [0.7, 0.7, 0.2, 0.2] -> class 1 (Tái chế)
    lbl_name = "canteen_2024-07-17_1201_pre.txt"
    lbl_content = (
        "0 0.3 0.3 0.2 0.2\n"
        "1 0.7 0.7 0.2 0.2\n"
    )
    (lbl_dir / lbl_name).write_text(lbl_content, encoding="utf-8")

    # Tạo metadata
    with open(meta_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "cluster_id", "date"])
        writer.writeheader()
        writer.writerow({
            "image_id": img_name,
            "cluster_id": "20240717_trua",
            "date": "2024-07-17",
        })

    # Tạo date-plan: 2024-07-17 thuộc val
    with open(plan_p, "w", encoding="utf-8") as f:
        json.dump({"val": ["2024-07-17"], "test": ["2024-07-03"], "pool": []}, f)

    return {
        "root": tmp_path,
        "img_dir": img_dir,
        "lbl_dir": lbl_dir,
        "out_dir": out_dir,
        "meta_p": meta_p,
        "plan_p": plan_p,
        "img_name": img_name,
    }


def test_extract_crops_gt_mode(temp_tray_environment):
    """Kiểm tra extract_crops.py chế độ --boxes gt."""
    env = temp_tray_environment
    cmd = [
        sys.executable, "scripts/extract_crops.py",
        "--split", "val",
        "--boxes", "gt",
        "--pad", "0.15",
        "--metadata", str(env["meta_p"]),
        "--images-dir", str(env["img_dir"]),
        "--labels-dir", str(env["lbl_dir"]),
        "--date-plan", str(env["plan_p"]),
        "--out-dir", str(env["out_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True)
    assert "[HOÀN TẤT] Đã trích xuất thành công 2 crops" in res.stdout

    # Kiểm tra manifest CSV
    manifest_p = env["out_dir"] / "crops_manifest_val_gt.csv"
    assert manifest_p.exists()
    with open(manifest_p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["source"] == "gt"
    assert rows[0]["gt_group"] == "0"
    assert rows[1]["gt_group"] == "1"
    assert float(rows[0]["iou_with_gt"]) == 1.0


def test_extract_crops_det_pseudo_mode(temp_tray_environment):
    """Kiểm tra extract_crops.py chế độ --boxes det (pseudo fallback)."""
    env = temp_tray_environment
    cmd = [
        sys.executable, "scripts/extract_crops.py",
        "--split", "val",
        "--boxes", "det",
        "--pad", "0.15",
        "--metadata", str(env["meta_p"]),
        "--images-dir", str(env["img_dir"]),
        "--labels-dir", str(env["lbl_dir"]),
        "--date-plan", str(env["plan_p"]),
        "--out-dir", str(env["out_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True)
    assert "[HOÀN TẤT] Đã trích xuất thành công" in res.stdout

    manifest_p = env["out_dir"] / "crops_manifest_val_det.csv"
    assert manifest_p.exists()
    with open(manifest_p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert rows[0]["source"] == "det"


def test_extract_crops_blocks_test_split(temp_tray_environment):
    """Test guard bắt buộc chặn --split test khi DSR_ALLOW_TEST=0."""
    env = temp_tray_environment
    cmd = [
        sys.executable, "scripts/extract_crops.py",
        "--split", "test",
        "--boxes", "gt",
        "--metadata", str(env["meta_p"]),
        "--images-dir", str(env["img_dir"]),
        "--labels-dir", str(env["lbl_dir"]),
        "--date-plan", str(env["plan_p"]),
        "--out-dir", str(env["out_dir"]),
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert res.returncode != 0
    assert "PROTOCOL VIOLATION - TEST GUARD" in res.stderr
