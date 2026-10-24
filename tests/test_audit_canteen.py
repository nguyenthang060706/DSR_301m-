from pathlib import Path
import pytest
from scripts.audit_canteen_data import audit_canteen


def test_audit_empty_directory(tmp_path: Path):
    empty_images = tmp_path / "images"
    empty_labels = tmp_path / "labels"
    out_csv = tmp_path / "reports" / "audit.csv"

    # Should run on non-existent or empty dir without crashing
    rows = audit_canteen(
        metadata_path=None,
        images_dir=empty_images,
        labels_dir=empty_labels,
        out_csv=out_csv,
        no_hash=True,
    )

    assert out_csv.exists()
    status_row = next((r for r in rows if r["metric"] == "status"), None)
    assert status_row is not None
    assert status_row["value"] == "EMPTY"


def test_audit_mock_canteen_data(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Tạo ảnh giả
    (images_dir / "20261024_trua_001.jpg").write_bytes(b"dummy_image_1")
    (images_dir / "20261024_trua_002.jpg").write_bytes(b"dummy_image_2")
    (images_dir / "20261025_chieu_001.jpg").write_bytes(b"dummy_image_3")

    # Tạo nhãn YOLO giả
    # 20261024_trua_001 có 2 box: 1 organic (0), 1 recyclable (1)
    (labels_dir / "20261024_trua_001.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", encoding="utf-8")
    # 20261024_trua_002 có 1 box: other_landfill (2)
    (labels_dir / "20261024_trua_002.txt").write_text("2 0.4 0.4 0.2 0.2\n", encoding="utf-8")
    # 20261025_chieu_001 chưa có nhãn (missing)

    # Tạo metadata giả
    meta_csv = tmp_path / "metadata.csv"
    meta_csv.write_text(
        "image_id,cluster_id,date,annotator_1,annotator_2,ambiguous\n"
        "20261024_trua_001.jpg,20261024_trua,2026-10-24,ann_a,ann_b,0\n"
        "20261024_trua_002.jpg,20261024_trua,2026-10-24,ann_a,,0\n"  # missing annotator_2
        "20261025_chieu_001.jpg,invalid_cluster,2026-10-25,ann_a,ann_b,1\n",  # invalid cluster + ambiguous
        encoding="utf-8"
    )

    out_csv = tmp_path / "reports" / "audit.csv"

    rows = audit_canteen(
        metadata_path=meta_csv,
        images_dir=images_dir,
        labels_dir=labels_dir,
        out_csv=out_csv,
        no_hash=True,
    )

    assert out_csv.exists()
    row_dict = {r["metric"]: r["value"] for r in rows}

    assert row_dict["total_images"] == "3"
    assert row_dict["images_with_labels"] == "2"
    assert row_dict["missing_labels"] == "1"
    assert row_dict["total_boxes"] == "3"
    assert "group_0:1" in row_dict["boxes_by_group"]
    assert "group_1:1" in row_dict["boxes_by_group"]
    assert "group_2:1" in row_dict["boxes_by_group"]
    assert row_dict["invalid_cluster_ids"] == "1"
    assert row_dict["missing_annotators"] == "1"
    assert row_dict["ambiguous_count"] == "1"
