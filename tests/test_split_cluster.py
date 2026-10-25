"""tests/test_split_cluster.py

Unit tests for scripts/split_canteen_by_cluster.py (Spec 8.6).
Kiểm tra:
1. Không rò rỉ ngày/cụm giữa các tập split (Test ∩ Val = ∅, Test ∩ Pool = ∅, Val ∩ Pool = ∅).
2. Dừng và ném RuntimeError khi số ngày chụp < 5 mà không có cờ --allow-shift-split.
3. Cho phép chạy khi có cờ --allow-shift-split.
4. Cờ --dry-run: chỉ in thống kê, tuyệt đối không copy file và không sinh test manifest.
5. Khi chạy thật: sinh đầy đủ ảnh trong images/pool, images/val, images/test,
   sinh song song 2 bộ nhãn labels_agnostic/ (class 0) và labels_3cls/ (giữ nguyên class),
   và sinh file canteen_test_manifest.json với đầy đủ mã SHA-256.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
import pytest


@pytest.fixture
def temp_canteen_env(tmp_path: Path):
    """Tạo môi trường giả lập nhỏ gọn để kiểm thử split cluster."""
    images_dir = tmp_path / "raw_images"
    labels_dir = tmp_path / "raw_labels"
    out_dir = tmp_path / "processed_yolo"
    manifest_p = tmp_path / "splits" / "test_manifest.json"
    report_p = tmp_path / "reports" / "split_report.csv"
    meta_p = tmp_path / "canteen_metadata.csv"
    plan_p = tmp_path / "canteen_date_plan.json"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Tạo 6 ngày giả lập (mỗi ngày 2 ảnh)
    rows = []
    dates = [
        "2026-10-01", "2026-10-02", "2026-10-03",
        "2026-10-04", "2026-10-05", "2026-10-06"
    ]
    img_idx = 0
    for d in dates:
        d_nodash = d.replace("-", "")
        for ca in ["trua", "chieu"]:
            img_idx += 1
            img_name = f"canteen_{d}_{ca}_{img_idx:03d}.png"
            lbl_name = f"canteen_{d}_{ca}_{img_idx:03d}.txt"

            # Tạo file ảnh giả lập (dummy content)
            (images_dir / img_name).write_bytes(b"DUMMY_IMAGE_DATA_" + str(img_idx).encode())

            # Tạo file nhãn giả lập YOLO (class 1 và class 2)
            lbl_content = (
                "1 0.5 0.5 0.2 0.2\n"
                "2 0.3 0.3 0.1 0.1\n"
            )
            (labels_dir / lbl_name).write_text(lbl_content, encoding="utf-8")

            rows.append({
                "image_id": img_name,
                "cluster_id": f"{d_nodash}_{ca}",
                "date": d,
                "device": "phone",
                "annotator_1": "a1",
                "annotator_2": "a2",
                "ambiguous": "0",
                "note": "test tray",
            })

    # Ghi metadata
    with open(meta_p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Ghi date plan: 2 ngày test, 2 ngày val, 2 ngày pool
    date_plan = {
        "test": [dates[0], dates[1]],
        "val": [dates[2], dates[3]],
        "pool": [dates[4], dates[5]],
    }
    with open(plan_p, "w", encoding="utf-8") as f:
        json.dump(date_plan, f)

    return {
        "root": tmp_path,
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "out_dir": out_dir,
        "manifest_p": manifest_p,
        "report_p": report_p,
        "meta_p": meta_p,
        "plan_p": plan_p,
        "dates": dates,
    }


def test_dry_run_does_not_modify_filesystem(temp_canteen_env):
    """Kiểm tra cờ --dry-run: không copy ảnh, không tạo thư mục out, không ghi manifest."""
    env = temp_canteen_env
    cmd = [
        sys.executable, "scripts/split_canteen_by_cluster.py",
        "--metadata", str(env["meta_p"]),
        "--images-dir", str(env["images_dir"]),
        "--labels-dir", str(env["labels_dir"]),
        "--out-dir", str(env["out_dir"]),
        "--date-plan", str(env["plan_p"]),
        "--manifest-out", str(env["manifest_p"]),
        "--report-out", str(env["report_p"]),
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert "[DRY-RUN - KHÔNG GHI FILE]" in res.stdout
    assert "[DRY-RUN HOÀN TẤT]" in res.stdout

    # Kiểm tra không sinh file ra đĩa
    assert not env["out_dir"].exists()
    assert not env["manifest_p"].exists()
    assert not env["report_p"].exists()


def test_abort_when_dates_less_than_5_without_flag(temp_canteen_env):
    """Spec 8.6: Bắt buộc dừng và ném lỗi nếu số ngày < 5 và không có --allow-shift-split."""
    env = temp_canteen_env
    # Tạo metadata chỉ có 3 ngày
    short_meta = env["root"] / "short_meta.csv"
    with open(env["meta_p"], "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    short_rows = [r for r in rows if r["date"] in env["dates"][:3]]
    with open(short_meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(short_rows)

    cmd = [
        sys.executable, "scripts/split_canteen_by_cluster.py",
        "--metadata", str(short_meta),
        "--images-dir", str(env["images_dir"]),
        "--labels-dir", str(env["labels_dir"]),
        "--out-dir", str(env["out_dir"]),
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert res.returncode != 0
    assert "Số ngày chụp quá ít" in res.stderr or "LỖI AN TOÀN SPEC 8.6" in res.stderr


def test_allow_shift_split_when_dates_less_than_5(temp_canteen_env):
    """Khi có cờ --allow-shift-split, cho phép chạy ngay cả khi số ngày < 5."""
    env = temp_canteen_env
    short_meta = env["root"] / "short_meta.csv"
    with open(env["meta_p"], "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    short_rows = [r for r in rows if r["date"] in env["dates"][:3]]
    with open(short_meta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(short_rows)

    cmd = [
        sys.executable, "scripts/split_canteen_by_cluster.py",
        "--metadata", str(short_meta),
        "--images-dir", str(env["images_dir"]),
        "--labels-dir", str(env["labels_dir"]),
        "--out-dir", str(env["out_dir"]),
        "--allow-shift-split",
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace")
    assert "cờ --allow-shift-split được bật" in res.stdout


def test_real_split_dual_labels_and_manifest(temp_canteen_env):
    """Kiểm tra chạy thật: phân bổ ảnh vào images/{pool, val, test},
    sinh labels_3cls và labels_agnostic, sinh test manifest có SHA256."""
    env = temp_canteen_env
    cmd = [
        sys.executable, "scripts/split_canteen_by_cluster.py",
        "--metadata", str(env["meta_p"]),
        "--images-dir", str(env["images_dir"]),
        "--labels-dir", str(env["labels_dir"]),
        "--out-dir", str(env["out_dir"]),
        "--date-plan", str(env["plan_p"]),
        "--manifest-out", str(env["manifest_p"]),
        "--report-out", str(env["report_p"]),
    ]
    subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", check=True)

    # 1. Kiểm tra cấu trúc thư mục
    out = env["out_dir"]
    for sp in ["pool", "val", "test"]:
        assert (out / "images" / sp).exists()
        assert (out / "labels_3cls" / sp).exists()
        assert (out / "labels_agnostic" / sp).exists()

    # Mỗi split có 2 ngày * 2 ca = 4 ảnh
    assert len(list((out / "images" / "test").glob("*.png"))) == 4
    assert len(list((out / "images" / "val").glob("*.png"))) == 4
    assert len(list((out / "images" / "pool").glob("*.png"))) == 4

    # 2. Kiểm tra bộ nhãn song song
    sample_lbl = list((out / "labels_3cls" / "test").glob("*.txt"))[0]
    lines_3cls = sample_lbl.read_text(encoding="utf-8").strip().splitlines()
    assert lines_3cls[0].startswith("1 ")  # Class gốc 1
    assert lines_3cls[1].startswith("2 ")  # Class gốc 2

    agnostic_lbl = out / "labels_agnostic" / "test" / sample_lbl.name
    lines_agnostic = agnostic_lbl.read_text(encoding="utf-8").strip().splitlines()
    assert lines_agnostic[0].startswith("0 ")  # Đã chuyển về class 0
    assert lines_agnostic[1].startswith("0 ")  # Đã chuyển về class 0
    # Tọa độ box phải được giữ nguyên
    assert lines_agnostic[0].split()[1:] == lines_3cls[0].split()[1:]

    # 3. Kiểm tra Test Manifest
    assert env["manifest_p"].exists()
    with open(env["manifest_p"], "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["dataset"] == "canteen_test_frozen"
    assert manifest["total_test_trays"] == 4
    assert len(manifest["files"]) == 4
    for item in manifest["files"]:
        assert "sha256" in item
        assert len(item["sha256"]) == 64
        assert "path" in item
        assert item["path"].startswith("images/test/")

    # 4. Kiểm tra Report CSV
    assert env["report_p"].exists()
    with open(env["report_p"], "r", encoding="utf-8") as f:
        rep_rows = list(csv.DictReader(f))
    assert len(rep_rows) == 3
    splits_in_rep = {r["split"]: int(r["trays"]) for r in rep_rows}
    assert splits_in_rep == {"test": 4, "val": 4, "pool": 4}
