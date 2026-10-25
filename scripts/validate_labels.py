"""scripts/validate_labels.py

D2-C4: Kiểm tra chất lượng nhãn YOLO trong một thư mục.
Kiểm tra:
  - class_id ∈ {0,1,2} (hoặc {0} với bản agnostic)
  - Tọa độ ∈ [0,1], w > 0, h > 0
  - Box < 16x16 px → cảnh báo
  - FAIL nếu 100% box đều là class 0 (dấu hiệu quên gán nhóm sau pre-label)

Sử dụng:
    python scripts/validate_labels.py --label-dir data/raw/canteen_labels \
        --image-dir data/raw/canteen_trays \
        [--agnostic]  # nếu chỉ cho phép class 0
"""
from __future__ import annotations

import sys
import io
import argparse
from pathlib import Path

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def validate_label_file(
    label_path: Path,
    image_w: int | None,
    image_h: int | None,
    valid_classes: set[int],
    min_box_px: int = 16,
) -> tuple[list[str], list[str], int]:
    """Validate a single YOLO label file.

    Returns (errors, warnings, num_boxes).
    """
    errors = []
    warnings = []
    num_boxes = 0

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(f"L{i}: thiếu trường (cần ≥5, có {len(parts)})")
            continue

        try:
            class_id = int(parts[0])
        except ValueError:
            errors.append(f"L{i}: class_id không phải số nguyên: '{parts[0]}'")
            continue

        if class_id not in valid_classes:
            errors.append(f"L{i}: class_id={class_id} ngoài {valid_classes}")

        try:
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append(f"L{i}: tọa độ không hợp lệ")
            continue

        # Tọa độ trong [0,1]
        for name, val in [("xc", xc), ("yc", yc), ("w", bw), ("h", bh)]:
            if val < 0.0 or val > 1.0:
                errors.append(f"L{i}: {name}={val:.4f} ngoài [0,1]")

        if bw <= 0:
            errors.append(f"L{i}: w={bw} ≤ 0")
        if bh <= 0:
            errors.append(f"L{i}: h={bh} ≤ 0")

        # Cảnh báo box quá nhỏ
        if image_w is not None and image_h is not None:
            box_w_px = bw * image_w
            box_h_px = bh * image_h
            if box_w_px < min_box_px and box_h_px < min_box_px:
                warnings.append(f"L{i}: box {box_w_px:.0f}x{box_h_px:.0f} px < {min_box_px}x{min_box_px}")

        num_boxes += 1

    return errors, warnings, num_boxes


def validate_directory(
    label_dir: Path,
    image_dir: Path | None,
    agnostic: bool = False,
    min_box_px: int = 16,
) -> dict:
    """Validate all YOLO label files in a directory.

    Returns summary dict with errors, warnings, stats.
    """
    valid_classes = {0} if agnostic else {0, 1, 2}

    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        return {
            "status": "WARN",
            "message": f"Không tìm thấy file .txt nào trong {label_dir}",
            "total_files": 0,
            "total_boxes": 0,
            "class_counts": {},
            "errors": [],
            "warnings": [],
        }

    total_files = len(label_files)
    total_boxes = 0
    total_errors = []
    total_warnings = []
    class_counts: dict[int, int] = {}
    files_with_errors = 0

    for lbl_file in label_files:
        stem = lbl_file.stem

        # Try to get image dimensions
        img_w, img_h = None, None
        if image_dir is not None:
            for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                img_path = image_dir / f"{stem}{ext}"
                if img_path.exists():
                    try:
                        from PIL import Image
                        img = Image.open(img_path)
                        img_w, img_h = img.size
                        img.close()
                    except Exception:
                        pass
                    break

        errors, warnings, num_boxes = validate_label_file(
            lbl_file, img_w, img_h, valid_classes, min_box_px
        )

        if errors:
            files_with_errors += 1
            for e in errors:
                total_errors.append(f"{lbl_file.name}: {e}")

        for w in warnings:
            total_warnings.append(f"{lbl_file.name}: {w}")

        total_boxes += num_boxes

        # Count classes
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cid = int(parts[0])
                        class_counts[cid] = class_counts.get(cid, 0) + 1
                    except ValueError:
                        pass

    # Check: 100% box là class 0 khi KHÔNG phải agnostic → FAIL
    all_class_0 = False
    if not agnostic and total_boxes > 0:
        if set(class_counts.keys()) == {0}:
            all_class_0 = True
            total_errors.append(
                f"[CRITICAL] 100% box ({total_boxes}) đều là class 0 — "
                f"có thể quên gán nhóm sau pre-label!"
            )

    status = "PASS"
    if total_errors:
        status = "FAIL"
    elif total_warnings:
        status = "WARN"

    return {
        "status": status,
        "total_files": total_files,
        "total_boxes": total_boxes,
        "class_counts": class_counts,
        "files_with_errors": files_with_errors,
        "errors": total_errors,
        "warnings": total_warnings,
        "all_class_0": all_class_0,
    }


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra chất lượng nhãn YOLO (D2-C4)")
    parser.add_argument("--label-dir", type=str, required=True, help="Thư mục chứa file .txt nhãn YOLO")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Thư mục ảnh tương ứng (để kiểm tra kích thước box pixel)")
    parser.add_argument("--agnostic", action="store_true",
                        help="Cho phép chỉ class 0 (chế độ class-agnostic)")
    parser.add_argument("--min-box-px", type=int, default=16,
                        help="Ngưỡng cạnh nhỏ nhất (px) để cảnh báo")
    args = parser.parse_args()

    label_dir = Path(args.label_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None

    if not label_dir.exists():
        print(f"[LỖI] Thư mục nhãn không tồn tại: {label_dir}")
        sys.exit(1)

    print(f"[*] Kiểm tra nhãn trong: {label_dir}")
    print(f"    Chế độ: {'agnostic (class 0)' if args.agnostic else '3 nhóm (class 0,1,2)'}")

    result = validate_directory(label_dir, image_dir, args.agnostic, args.min_box_px)

    print(f"\n{'='*60}")
    print(f"  KẾT QUẢ KIỂM TRA: [{result['status']}]")
    print(f"{'='*60}")
    print(f"  Tổng file:      {result['total_files']}")
    print(f"  Tổng box:        {result['total_boxes']}")
    print(f"  Class counts:    {result['class_counts']}")

    if result.get("files_with_errors", 0) > 0:
        print(f"  File có lỗi:     {result['files_with_errors']}")

    if result["warnings"]:
        print(f"\n  ⚠ Cảnh báo ({len(result['warnings'])}):")
        for w in result["warnings"][:20]:
            print(f"    - {w}")
        if len(result["warnings"]) > 20:
            print(f"    ... và {len(result['warnings']) - 20} cảnh báo khác")

    if result["errors"]:
        print(f"\n  ✗ Lỗi ({len(result['errors'])}):")
        for e in result["errors"][:20]:
            print(f"    - {e}")
        if len(result["errors"]) > 20:
            print(f"    ... và {len(result['errors']) - 20} lỗi khác")

    if result["status"] == "FAIL":
        print(f"\n  [FAIL] Nhãn KHÔNG đạt yêu cầu chất lượng!")
        sys.exit(1)
    elif result["status"] == "WARN":
        print(f"\n  [WARN] Nhãn có cảnh báo nhưng không có lỗi nghiêm trọng.")
    else:
        print(f"\n  [PASS] Tất cả nhãn hợp lệ!")


if __name__ == "__main__":
    main()
