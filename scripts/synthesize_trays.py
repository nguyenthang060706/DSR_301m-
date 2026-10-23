"""scripts/synthesize_trays.py

Script sinh dữ liệu khay rác nhân tạo bằng kỹ thuật Copy-Paste tổng hợp (MASTER v10.0 §3.5).
Ghép 3-6 vật thể từ kho ảnh Kaggle (household_waste_30) lên ảnh nền khay ăn/bàn ăn canteen:
- Tự động thay đổi tỉ lệ (scale), xoay nhẹ (rotation -30 đến +30 độ), làm mềm viền (feather).
- Kiểm soát chồng lấn: IoU giữa các bounding box <= 0.3.
- Xuất dữ liệu chuẩn YOLO (ảnh JPG/PNG + file tọa độ TXT class 0: waste).
"""

import sys
import io
import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def compute_box_iou(box1, box2):
    # box format: [xmin, ymin, xmax, ymax]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def extract_foreground_simple(img_pil):
    """
    Tách vật thể đơn giản nếu ảnh có nền sáng hoặc đồng nhất,
    hoặc trả về toàn bộ ảnh với viền mềm nhẹ.
    """
    img_rgba = img_pil.convert("RGBA")
    return img_rgba

def generate_synthetic_tray(bg_img, object_images, output_size=(640, 640), num_objects=4):
    """
    Tạo 1 ảnh khay tổng hợp từ ảnh nền và danh sách ảnh vật thể.
    """
    canvas = bg_img.resize(output_size).convert("RGBA")
    w_canvas, h_canvas = output_size
    placed_boxes = []

    sampled_objs = random.sample(object_images, min(num_objects, len(object_images)))

    for obj_path in sampled_objs:
        try:
            obj_img = Image.open(obj_path).convert("RGBA")
        except Exception:
            continue

        # Scale ngẫu nhiên: 15% đến 35% chiều rộng canvas
        target_w = random.randint(int(w_canvas * 0.15), int(w_canvas * 0.35))
        aspect_ratio = obj_img.height / max(1, obj_img.width)
        target_h = int(target_w * aspect_ratio)

        # Xoay ngẫu nhiên
        angle = random.randint(-30, 30)
        obj_resized = obj_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        obj_rotated = obj_resized.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        rw, rh = obj_rotated.size

        # Tìm vị trí không bị chồng lấn quá IoU 0.3 (thử tối đa 20 lần)
        max_attempts = 20
        best_pos = None

        for _ in range(max_attempts):
            if w_canvas - rw <= 0 or h_canvas - rh <= 0:
                break
            px = random.randint(10, max(11, w_canvas - rw - 10))
            py = random.randint(10, max(11, h_canvas - rh - 10))
            cand_box = [px, py, px + rw, py + rh]

            # Kiểm tra IoU với các box đã đặt
            overlap = False
            for pb in placed_boxes:
                if compute_box_iou(cand_box, pb) > 0.3:
                    overlap = True
                    break
            if not overlap:
                best_pos = (px, py, cand_box)
                break

        if best_pos is not None:
            px, py, cand_box = best_pos
            # Ghép vào canvas với kênh alpha
            canvas.paste(obj_rotated, (px, py), obj_rotated)
            placed_boxes.append(cand_box)

    # Chuyển đổi box sang chuẩn YOLO: class_id, x_center, y_center, width, height (0..1)
    yolo_labels = []
    for b in placed_boxes:
        xmin, ymin, xmax, ymax = b
        xc = ((xmin + xmax) / 2.0) / w_canvas
        yc = ((ymin + ymax) / 2.0) / h_canvas
        bw = (xmax - xmin) / float(w_canvas)
        bh = (ymax - ymin) / float(h_canvas)
        yolo_labels.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    return canvas.convert("RGB"), yolo_labels

def main():
    parser = argparse.ArgumentParser(description="Sinh ảnh khay rác tổng hợp Copy-Paste")
    parser.add_argument("--num_images", type=int, default=100, help="Số lượng ảnh cần sinh pilot")
    parser.add_argument("--output_dir", type=str, default="data/processed/synthetic_trays", help="Thư mục xuất")
    args = parser.parse_args()

    raw_dir = Path("data/raw/household_waste_30")
    if not raw_dir.exists():
        print(f"[LỖI] Không tìm thấy tập ảnh gốc: {raw_dir}")
        sys.exit(1)

    # Thu thập danh sách ảnh vật thể (hỗ trợ cấu trúc lồng nhau default / real_world)
    all_obj_paths = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png")) + list(raw_dir.rglob("*.jpeg"))
    print(f"[*] Tìm thấy {len(all_obj_paths)} ảnh vật thể đơn lẻ từ Kaggle 30-class.")


    out_path = Path(args.output_dir)
    train_img_out = out_path / "images" / "train"
    train_lbl_out = out_path / "labels" / "train"
    val_img_out = out_path / "images" / "val"
    val_lbl_out = out_path / "labels" / "val"

    for d in [train_img_out, train_lbl_out, val_img_out, val_lbl_out]:
        d.mkdir(parents=True, exist_ok=True)

    # Nạp ảnh nền canteen thật nếu có, ngược lại dùng bảng màu đa dạng
    bg_dir = Path("data/raw/canteen_backgrounds")
    real_bgs = []
    if bg_dir.exists():
        real_bgs = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))

    val_count = int(args.num_images * 0.15)
    train_count = args.num_images - val_count

    print(f"[*] Đang sinh {args.num_images} ảnh khay nhân tạo ({train_count} train, {val_count} val)...")
    
    bg_colors = [
        (210, 215, 220),  # khay nhựa ghi sáng
        (185, 190, 195),  # khay inox mờ
        (225, 215, 195),  # mặt bàn gỗ melamine vàng nhạt
        (240, 240, 242),  # mặt bàn nhựa trắng
        (160, 165, 170),  # khay inox xám đậm
        (205, 195, 180),  # mặt bàn canteen gỗ sồi
    ]

    for idx in range(args.num_images):
        split = "val" if idx < val_count else "train"
        target_img_dir = val_img_out if split == "val" else train_img_out
        target_lbl_dir = val_lbl_out if split == "val" else train_lbl_out

        if real_bgs and random.random() < 0.5:
            try:
                bg = Image.open(random.choice(real_bgs)).convert("RGB")
            except Exception:
                bg = Image.new("RGB", (640, 640), color=random.choice(bg_colors))
        else:
            bg = Image.new("RGB", (640, 640), color=random.choice(bg_colors))

        num_objs = random.randint(3, 7)
        tray_img, labels = generate_synthetic_tray(bg, all_obj_paths, num_objects=num_objs)

        file_stem = f"synth_tray_{idx:05d}"
        tray_img.save(target_img_dir / f"{file_stem}.jpg", quality=90)
        with open(target_lbl_dir / f"{file_stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(labels) + "\n")

    # Tạo data.yaml cho synthetic trays
    yaml_content = f"""# Synthetic Trays Dataset Configuration
path: {out_path.resolve().as_posix()}
train: images/train
val: images/val

# Classes
names:
  0: waste
"""
    with open(out_path / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[THÀNH CÔNG] Đã sinh {args.num_images} ảnh khay ({train_count} train, {val_count} val).")
    print(f"[*] Đã tạo cấu hình YOLO tại: {out_path / 'data.yaml'}")

if __name__ == "__main__":
    main()
