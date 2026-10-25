"""src/dsr/test_guard.py

D3-C2 / Spec 8.7: Cơ chế chặn máy (machine-enforced guard) ngăn việc rò rỉ hoặc đọc
tập Test Canteen ngoài scripts/eval_test_once.py.

Nguyên tắc:
- Mọi script huấn luyện, validation, tuning, trích crop, hoặc sinh yaml đều phải gọi guard.
- Raise RuntimeError nếu phát hiện đường dẫn thuộc Test.
- Chỉ cho phép bypass khi biến môi trường DSR_ALLOW_TEST=1 (chỉ được bật trong eval_test_once.py).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Union


DEFAULT_MANIFEST_PATH = Path("data/splits/canteen_test_manifest.json")


def is_test_allowed() -> bool:
    """Check if test access is explicitly authorized via environment variable."""
    return os.environ.get("DSR_ALLOW_TEST", "0") == "1"


def get_test_stems_and_paths(
    manifest_path: Union[Path, str, None] = None,
) -> tuple[set[str], set[str]]:
    """Load test image file stems and relative paths from test manifest.

    Returns:
        (test_stems, test_paths)
    """
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    if not path.exists():
        return set(), set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set(), set()

    stems: set[str] = set()
    paths: set[str] = set()

    # Support manifest formats:
    # 1. {"files": [{"path": ..., "sha256": ...}, ...]}
    # 2. {"stems": [...], "images": [...]}
    # 3. dict mapping path/stem -> sha256
    # 4. list of paths/dicts
    items = []
    if isinstance(data, dict):
        if "files" in data and isinstance(data["files"], list):
            items = data["files"]
        elif "images" in data and isinstance(data["images"], list):
            items = data["images"]
        else:
            items = list(data.keys())
    elif isinstance(data, list):
        items = data

    for item in items:
        if isinstance(item, dict):
            p = item.get("path") or item.get("file") or item.get("name") or ""
        elif isinstance(item, str):
            p = item
        else:
            continue

        if p:
            clean_p = Path(p).as_posix().lower()
            paths.add(clean_p)
            stems.add(Path(p).stem.lower())

    return stems, paths


def assert_not_test(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    manifest_path: Union[Path, str, None] = None,
) -> None:
    """Raise RuntimeError if any path belongs to the frozen Test set."""
    if is_test_allowed():
        return

    test_stems, test_paths = get_test_stems_and_paths(manifest_path)
    if not test_stems and not test_paths:
        return  # Manifest doesn't exist yet; nothing to guard

    if isinstance(paths, (str, Path)):
        check_list = [paths]
    else:
        check_list = list(paths)

    for item in check_list:
        p_obj = Path(item)
        stem = p_obj.stem.lower()
        posix_p = p_obj.as_posix().lower()

        if stem in test_stems or posix_p in test_paths:
            raise RuntimeError(
                f"[PROTOCOL VIOLATION - TEST GUARD] Tuyệt đối không được nạp dữ liệu Test ({item}) "
                f"vào huấn luyện, validation hay tuning! Tập Test đã được đóng băng."
            )


def assert_no_test_in_yaml(
    yaml_target: Union[str, Path, dict[str, Any]],
    manifest_path: Union[Path, str, None] = None,
) -> None:
    """Verify that a YOLO dataset YAML configuration does not point to Test images."""
    if is_test_allowed():
        return

    test_stems, test_paths = get_test_stems_and_paths(manifest_path)
    if not test_stems and not test_paths:
        return

    if isinstance(yaml_target, (str, Path)):
        import yaml
        with open(yaml_target, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif isinstance(yaml_target, dict):
        cfg = yaml_target
    else:
        raise ValueError("yaml_target must be a Path, str or dict")

    # Check 'train' and 'val' entries in YAML
    for key in ["train", "val"]:
        val = cfg.get(key)
        if not val:
            continue
        # val can be string or list of strings
        subpaths = [val] if isinstance(val, str) else list(val)
        for sp in subpaths:
            sp_lower = str(sp).lower().replace("\\", "/")
            # If the path points directly to test or contains test keywords associated with test split
            if "test" in sp_lower.split("/") or "canteen_test" in sp_lower:
                raise RuntimeError(
                    f"[PROTOCOL VIOLATION - TEST GUARD] File YAML trỏ trường '{key}' "
                    f"vào đường dẫn Test ({sp})!"
                )
