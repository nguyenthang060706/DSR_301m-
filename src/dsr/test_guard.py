"""src/dsr/test_guard.py

D3-C2 / Spec 8.7: Cơ chế chặn máy 2 lớp (Two-Stage Machine-Enforced Guard)
ngăn chặn tuyệt đối việc rò rỉ hoặc đọc tập Test Canteen ngoài scripts/eval_test_once.py.

Nguyên tắc:
1. Lớp 1 (Ngày 3–4, trước khi có manifest): Đọc configs/canteen_date_plan.json.
   Mọi đường dẫn ảnh/crop thuộc danh sách ngày "test" bị chặn ngay lập tức.
2. Lớp 2 (Ngày 4–6, sau khi đóng băng): Đọc data/splits/canteen_test_manifest.json (mã SHA-256 và paths).
3. Cho phép đọc file metadata/nhãn để thống kê số lượng hoặc chạy --dry-run (không chặn đếm metadata).
4. Chỉ cho phép bypass khi biến môi trường DSR_ALLOW_TEST=1 (chỉ được bật trong eval_test_once.py).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Union


DEFAULT_MANIFEST_PATH = Path("data/splits/canteen_test_manifest.json")
DEFAULT_DATE_PLAN_PATH = Path("configs/canteen_date_plan.json")


def is_test_allowed() -> bool:
    """Check if test access is explicitly authorized via environment variable."""
    return os.environ.get("DSR_ALLOW_TEST", "0") == "1"


def normalize_date_token(val: str) -> str:
    """Chuẩn hóa ngày sang chuỗi YYYYMMDD và YYYY-MM-DD."""
    val = val.strip().replace("/", "-")
    parts = val.split("_")[0].split()
    date_str = parts[0]
    if len(date_str) == 8 and date_str.isdigit():
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def get_test_dates_from_plan(
    date_plan_path: Union[Path, str, None] = None,
) -> set[str]:
    """Lấy danh sách các ngày thuộc tập Test từ configs/canteen_date_plan.json.
    
    Trả về tập các chuỗi ngày ở cả 2 định dạng (YYYY-MM-DD và YYYYMMDD).
    """
    path = Path(date_plan_path) if date_plan_path else DEFAULT_DATE_PLAN_PATH
    if not path.exists():
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    test_dates = set()
    raw_dates = data.get("test", [])
    for d in raw_dates:
        norm = normalize_date_token(str(d))
        test_dates.add(norm)
        test_dates.add(norm.replace("-", ""))
    return test_dates


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


# Regex nhận diện ngày trong tên file ảnh hoặc đường dẫn (YYYY-MM-DD hoặc YYYYMMDD)
DATE_REGEX = re.compile(r"(\d{4}-\d{2}-\d{2}|\d{8})")


def assert_not_test(
    paths: Union[str, Path, Iterable[Union[str, Path]]],
    manifest_path: Union[Path, str, None] = None,
    date_plan_path: Union[Path, str, None] = None,
) -> None:
    """Raise RuntimeError if any path belongs to the frozen Test set (Lớp 1: Date Plan, Lớp 2: Manifest)."""
    if is_test_allowed():
        return

    test_stems, test_paths = get_test_stems_and_paths(manifest_path)
    test_dates = get_test_dates_from_plan(date_plan_path)

    # Nếu cả manifest lẫn date-plan đều chưa có thì không có gì để chặn
    if not test_stems and not test_paths and not test_dates:
        return

    if isinstance(paths, (str, Path)):
        check_list = [paths]
    else:
        check_list = list(paths)

    for item in check_list:
        p_obj = Path(item)
        stem = p_obj.stem.lower()
        posix_p = p_obj.as_posix().lower()

        # Bỏ qua nếu đường dẫn chỉ là file kế hoạch hoặc metadata tổng thể
        if "canteen_date_plan.json" in posix_p or "canteen_metadata" in posix_p:
            continue

        # Lớp 2: Kiểm tra theo Manifest đã đóng băng
        if stem in test_stems or posix_p in test_paths:
            raise RuntimeError(
                f"[PROTOCOL VIOLATION - TEST GUARD] Tuyệt đối không được nạp dữ liệu Test ({item}) "
                f"vào huấn luyện, validation hay tuning! Tập Test đã được đóng băng trong manifest."
            )

        # Lớp 1: Kiểm tra theo Date Plan
        if test_dates:
            matches = DATE_REGEX.findall(stem) + DATE_REGEX.findall(posix_p)
            for m in matches:
                norm_m = normalize_date_token(m)
                if norm_m in test_dates or m in test_dates:
                    raise RuntimeError(
                        f"[PROTOCOL VIOLATION - TEST GUARD] Tuyệt đối không được nạp dữ liệu Test ({item}) "
                        f"thuộc ngày {norm_m} được chỉ định trong date-plan! Tập Test được bảo vệ."
                    )


def assert_no_test_in_yaml(
    yaml_target: Union[str, Path, dict[str, Any]],
    manifest_path: Union[Path, str, None] = None,
    date_plan_path: Union[Path, str, None] = None,
) -> None:
    """Verify that a YOLO dataset YAML configuration does not point to Test images."""
    if is_test_allowed():
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
        subpaths = [val] if isinstance(val, str) else list(val)
        for sp in subpaths:
            sp_lower = str(sp).lower().replace("\\", "/")
            if "test" in sp_lower.split("/") or "canteen_test" in sp_lower:
                raise RuntimeError(
                    f"[PROTOCOL VIOLATION - TEST GUARD] File YAML trỏ trường '{key}' "
                    f"vào đường dẫn Test ({sp})!"
                )
            # Kiểm tra theo ngày Test nếu có
            assert_not_test(sp, manifest_path=manifest_path, date_plan_path=date_plan_path)
