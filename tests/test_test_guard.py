"""tests/test_test_guard.py

Unit tests for src/dsr/test_guard.py (D3-C2 / Spec 8.7).
Verifies:
  1. Accessing test image paths raises RuntimeError.
  2. YAML pointing to test images raises RuntimeError.
  3. Non-test images pass silently.
  4. Bypass with DSR_ALLOW_TEST=1 works properly.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import pytest

from dsr.test_guard import (
    assert_not_test,
    assert_no_test_in_yaml,
    get_test_stems_and_paths,
)


@pytest.fixture
def mock_test_manifest(tmp_path):
    manifest_file = tmp_path / "canteen_test_manifest.json"
    data = {
        "files": [
            {"path": "data/raw/canteen_trays/canteen_2024-07-20_1201_pre.png", "sha256": "abcdef123456"},
            {"path": "data/raw/canteen_trays/canteen_2024-07-20_1202_pre.png", "sha256": "123456abcdef"},
        ]
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return manifest_file


def test_test_guard_blocks_test_path(mock_test_manifest):
    # Should raise RuntimeError when trying to load a test path
    with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION - TEST GUARD"):
        assert_not_test(
            "data/raw/canteen_trays/canteen_2024-07-20_1201_pre.png",
            manifest_path=mock_test_manifest,
        )

    # Should also detect by stem
    with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION - TEST GUARD"):
        assert_not_test(
            "some/other/path/canteen_2024-07-20_1202_pre.jpg",
            manifest_path=mock_test_manifest,
        )


def test_test_guard_allows_train_and_val_paths(mock_test_manifest):
    # Should pass without error for pool/train/val paths
    assert_not_test(
        [
            "data/raw/canteen_trays/canteen_2024-06-27_1201_pre.png",
            "data/raw/canteen_trays/canteen_2024-06-28_1201_pre.png",
        ],
        manifest_path=mock_test_manifest,
    )


def test_test_guard_blocks_yaml_with_test(mock_test_manifest, tmp_path):
    bad_yaml = {
        "train": "data/processed/canteen_yolo/images/train",
        "val": "data/processed/canteen_yolo/images/test",
    }
    with pytest.raises(RuntimeError, match="PROTOCOL VIOLATION - TEST GUARD"):
        assert_no_test_in_yaml(bad_yaml, manifest_path=mock_test_manifest)

    good_yaml = {
        "train": "data/processed/canteen_yolo/images/train",
        "val": "data/processed/canteen_yolo/images/val",
    }
    # Should pass
    assert_no_test_in_yaml(good_yaml, manifest_path=mock_test_manifest)


def test_test_guard_bypass_with_env(mock_test_manifest, monkeypatch):
    monkeypatch.setenv("DSR_ALLOW_TEST", "1")
    # Even if accessing test path, with DSR_ALLOW_TEST=1 it should pass
    assert_not_test(
        "data/raw/canteen_trays/canteen_2024-07-20_1201_pre.png",
        manifest_path=mock_test_manifest,
    )


def test_test_guard_blocks_date_plan_test_dates(tmp_path):
    # Lớp 1: Chặn theo canteen_date_plan.json ngay cả khi chưa có manifest Test
    plan_file = tmp_path / "canteen_date_plan.json"
    plan_data = {
        "test": ["2024-07-03", "2024-07-04"],
        "val": ["2024-07-17"],
        "pool": ["2024-06-27"],
    }
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump(plan_data, f)

    # Thử nạp ảnh thuộc ngày test 2024-07-03 -> phải raise RuntimeError
    with pytest.raises(RuntimeError, match="thuộc ngày 2024-07-03 được chỉ định trong date-plan"):
        assert_not_test(
            "data/raw/canteen_trays/canteen_2024-07-03_1201_pre.png",
            manifest_path=tmp_path / "nonexistent_manifest.json",
            date_plan_path=plan_file,
        )

    # Ảnh thuộc ngày pool 2024-06-27 -> cho phép
    assert_not_test(
        "data/raw/canteen_trays/canteen_2024-06-27_1201_pre.png",
        manifest_path=tmp_path / "nonexistent_manifest.json",
        date_plan_path=plan_file,
    )


def test_test_guard_allows_metadata_and_dryrun_paths(tmp_path):
    # Không chặn đọc file metadata
    plan_file = tmp_path / "canteen_date_plan.json"
    with open(plan_file, "w", encoding="utf-8") as f:
        json.dump({"test": ["2024-07-03"]}, f)

    assert_not_test(
        "data/canteen_metadata.csv",
        manifest_path=tmp_path / "nonexistent_manifest.json",
        date_plan_path=plan_file,
    )

