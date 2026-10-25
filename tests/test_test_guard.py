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
