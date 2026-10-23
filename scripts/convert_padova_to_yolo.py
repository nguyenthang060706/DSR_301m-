"""scripts/convert_padova_to_yolo.py

Chuyển đổi tập dữ liệu khay ăn & thức ăn thừa Đại học Padova
(Food-recognition-and-leftover-estimation-main) sang chuẩn YOLO class-agnostic.
Bổ sung khẩn cấp dữ liệu thức ăn thừa (Hữu cơ) cho Detector D1.
"""

import sys
import io
import re
from pathlib import Path
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def parse_padova_bbox(line):
    # Định dạng: ID: 6; [xmin, ymin, w, h]
    match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', line)
    if match:
        return [int(v) for v in match.groups()]
    return None

def main():
    dataset_dir = Path("data/raw/Food-recognition-and-leftover-estimation-main/dataset")
    if not dataset_dir.exists():
        print(f"[LỖI] Không tìm thấy thư mục Padova: {dataset_dir}")
        sys.exit(1)

    out_dir = Path("data/processed/padova_yolo")
    train_img_dir = out_dir / "images" / "train"
    train_lbl_dir = out_dir / "labels" / "train"
    val_img_dir = out_dir / "images" / "val"
    val_lbl_dir = out_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tray_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("tray")])
    print(f"[*] Tìm thấy {len(tray_dirs)} khay ăn Padova: {[d.name for d in tray_dirs]}")

    total_converted = 0
    total_boxes = 0

    for tray_idx, tray_dir in enumerate(tray_dirs):
        # Khay cuối (tray8 hoặc tray9) để làm val, còn lại làm train
        split = "val" if tray_idx >= len(tray_dirs) - 2 else "train"
        tgt_img_dir = val_img_dir if split == "val" else train_img_dir
        tgt_lbl_dir = val_lbl_dir if split == "val" else train_lbl_dir

        bbox_dir = tray_dir / "bounding_boxes"
        if not bbox_dir.exists():
            continue

        for img_path in sorted(list(tray_dir.glob("*.jpg")) + list(tray_dir.glob("*.JPG"))):
            bbox_file = bbox_dir / f"{img_path.stem}_bounding_box.txt"
            if not bbox_file.exists():
                continue

            try:
                with Image.open(img_path) as im:
                    w_img, h_img = im.size
            except Exception as e:
                print(f"[!] Lỗi đọc ảnh {img_path}: {e}")
                continue

            yolo_labels = []
            with open(bbox_file, "r", encoding="utf-8") as bf:
                for line in bf:
                    box = parse_padova_bbox(line)
                    if box:
                        xmin, ymin, bw, bh = box
                        if bw <= 0 or bh <= 0:
                            continue
                        xc = (xmin + bw / 2.0) / float(w_img)
                        yc = (ymin + bh / 2.0) / float(h_img)
                        norm_w = bw / float(w_img)
                        norm_h = bh / float(h_img)

                        # Giới hạn trong khoảng [0, 1]
                        xc = max(0.0, min(1.0, xc))
                        yc = max(0.0, min(1.0, yc))
                        norm_w = max(0.0, min(1.0, norm_w))
                        norm_h = max(0.0, min(1.0, norm_h))

                        # Class 0: waste (class-agnostic)
                        yolo_labels.append(f"0 {xc:.6f} {yc:.6f} {norm_w:.6f} {norm_h:.6f}")

            if yolo_labels:
                out_name = f"padova_{tray_dir.name}_{img_path.name}"
                # Sao chép ảnh và lưu nhãn
                out_img_path = tgt_img_dir / out_name
                try:
                    with Image.open(img_path) as im:
                        im.convert("RGB").save(out_img_path, quality=95)
                except Exception:
                    continue

                out_lbl_path = tgt_lbl_dir / f"{out_img_path.stem}.txt"
                with open(out_lbl_path, "w", encoding="utf-8") as lf:
                    lf.write("\n".join(yolo_labels) + "\n")

                total_converted += 1
                total_boxes += len(yolo_labels)

    # Ghi file data.yaml
    yaml_content = f"""# Padova Canteen Trays Dataset Configuration
path: {out_dir.resolve().as_posix()}
train: images/train
val: images/val

# Classes
names:
  0: waste
"""
    with open(out_dir / "data.yaml", "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"[THÀNH CÔNG] Đã chuyển đổi {total_converted} ảnh khay Padova với {total_boxes} bounding boxes.")
    print(f"[*] Cấu hình lưu tại: {out_dir / 'data.yaml'}")

if __name__ == "__main__":
    main()
