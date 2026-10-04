from pathlib import Path


def test_protocol_lock_exists_and_valid():
    lock_file = Path("docs/protocol_lock.md")
    assert lock_file.exists(), "docs/protocol_lock.md must exist"
    content = lock_file.read_text(encoding="utf-8")

    # Check key locked assertions from v9.5
    assert "household_waste_30" in content, "Must reference household_waste_30"
    assert "15.000 ảnh" in content, "Must record 15.000 images"
    assert "20261001" in content, "Must record seed 20261001"
    assert "21.60" in content, "Must record Teacher RER 21.60"
    assert "SGD" in content, "Must record locked optimizer SGD"
    print("test_protocol_lock_exists_and_valid PASS")


def test_label_map_valid():
    map_file = Path("data/mappings/label_map.csv")
    assert map_file.exists(), "label_map.csv must exist"
    lines = [l for l in map_file.read_text(encoding="utf-8").splitlines() if not l.startswith("#") and l.strip()]
    header = lines[0].split(",")
    assert "fine_id" in header
    assert "fine_label" in header
    assert "coarse_id" in header
    assert "coarse_label" in header
    # 30 data lines + 1 header = 31 lines
    assert len(lines) == 31, f"Expected 31 lines (1 header + 30 classes), got {len(lines)}"
    print("test_label_map_valid PASS")


if __name__ == "__main__":
    test_protocol_lock_exists_and_valid()
    test_label_map_valid()
