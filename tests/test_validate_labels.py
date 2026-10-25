"""tests/test_validate_labels.py

Unit tests cho scripts/validate_labels.py
"""
import pytest
from pathlib import Path

# Import validate function
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_labels import validate_label_file, validate_directory


class TestValidateLabelFile:
    def test_valid_3class(self, tmp_path):
        """File nhãn hợp lệ với 3 class."""
        label = tmp_path / "test.txt"
        label.write_text(
            "0 0.5 0.5 0.2 0.3\n"
            "1 0.3 0.7 0.15 0.1\n"
            "2 0.8 0.2 0.1 0.15\n"
        )
        errors, warnings, n = validate_label_file(label, None, None, {0, 1, 2})
        assert n == 3
        assert errors == []

    def test_valid_agnostic(self, tmp_path):
        """File nhãn agnostic hợp lệ."""
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.2 0.3\n0 0.3 0.7 0.15 0.1\n")
        errors, warnings, n = validate_label_file(label, None, None, {0})
        assert n == 2
        assert errors == []

    def test_invalid_class_id(self, tmp_path):
        """class_id ngoài tập cho phép."""
        label = tmp_path / "test.txt"
        label.write_text("5 0.5 0.5 0.2 0.3\n")
        errors, warnings, n = validate_label_file(label, None, None, {0, 1, 2})
        assert n == 1
        assert any("class_id=5" in e for e in errors)

    def test_coord_out_of_range(self, tmp_path):
        """Tọa độ ngoài [0,1]."""
        label = tmp_path / "test.txt"
        label.write_text("0 1.5 0.5 0.2 0.3\n")
        errors, warnings, n = validate_label_file(label, None, None, {0, 1, 2})
        assert any("ngoài [0,1]" in e for e in errors)

    def test_zero_width(self, tmp_path):
        """w = 0."""
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.0 0.3\n")
        errors, warnings, n = validate_label_file(label, None, None, {0, 1, 2})
        assert any("≤ 0" in e for e in errors)

    def test_small_box_warning(self, tmp_path):
        """Box < 16x16 px cảnh báo."""
        label = tmp_path / "test.txt"
        label.write_text("0 0.5 0.5 0.01 0.01\n")
        errors, warnings, n = validate_label_file(label, 640, 640, {0, 1, 2})
        # 0.01 * 640 = 6.4 px < 16 → cảnh báo
        assert len(warnings) > 0

    def test_empty_file(self, tmp_path):
        """File rỗng."""
        label = tmp_path / "test.txt"
        label.write_text("")
        errors, warnings, n = validate_label_file(label, None, None, {0, 1, 2})
        assert n == 0
        assert errors == []


class TestValidateDirectory:
    def test_all_class_0_fail(self, tmp_path):
        """100% class 0 trong chế độ 3-class → FAIL."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        (label_dir / "img2.txt").write_text("0 0.3 0.7 0.15 0.1\n")

        result = validate_directory(label_dir, image_dir=None, agnostic=False)
        assert result["status"] == "FAIL"
        assert result["all_class_0"] is True

    def test_all_class_0_ok_agnostic(self, tmp_path):
        """100% class 0 trong chế độ agnostic → PASS."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.3\n")
        (label_dir / "img2.txt").write_text("0 0.3 0.7 0.15 0.1\n")

        result = validate_directory(label_dir, image_dir=None, agnostic=True)
        assert result["status"] == "PASS"

    def test_mixed_classes_pass(self, tmp_path):
        """Nhãn có đủ nhóm → PASS."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()
        (label_dir / "img1.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.7 0.15 0.1\n")
        (label_dir / "img2.txt").write_text("2 0.8 0.2 0.1 0.15\n")

        result = validate_directory(label_dir, image_dir=None, agnostic=False)
        assert result["status"] == "PASS"
        assert result["class_counts"] == {0: 1, 1: 1, 2: 1}

    def test_empty_directory_warn(self, tmp_path):
        """Thư mục rỗng → WARN."""
        label_dir = tmp_path / "labels"
        label_dir.mkdir()

        result = validate_directory(label_dir, image_dir=None)
        assert result["status"] == "WARN"
        assert result["total_files"] == 0
