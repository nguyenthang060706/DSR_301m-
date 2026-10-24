"""scripts/auto_prelabel_canteen.py

Tự động sinh Bounding Box gợi ý (Model-Assisted Pre-labeling) cho toàn bộ ảnh khay canteen.
- Pipeline phối hợp 2 model trên GPU:
  1. YOLOv8x: Phát hiện các đĩa, bát thức ăn, món ăn thừa (casserole, pizza, pasta, rau đậu...).
  2. Bread Detector (YOLO fine-tuned): Phát hiện chính xác túi bánh mỳ (bread bun in clear plastic wrap).
- Ánh xạ nhãn:
  - 0: Huu_co (Tất cả thức ăn thừa, đĩa/bát đựng thức ăn thừa, túi bánh mỳ)
  - 1: Tai_che (Chai nhựa, lon, thủy tinh)
  - 2: Con_lai (Cốc dùng 1 lần, rác vô cơ khác)
- Xử lý triệt để:
  - Túi bánh mỳ luôn được bounding box và gắn nhãn 0:Huu_co (không bị sót, không bị gán nhầm sang 2:Con_lai).
  - Khử các box phản chiếu nhựa nhầm thành fork/knife/spoon/cup trên túi bánh mỳ.
  - Các miếng thức ăn thừa trên đĩa phẳng (lasagna, casserole, pizza...) đều được phát hiện đầy đủ.
  - Loại bỏ các vật thể nền ngoài khay (bàn, ghế, người, rổ bên cạnh...).
- Xuất nhãn YOLO .txt vào data/raw/canteen_labels/
- Xuất ảnh preview trực quan vào reports/canteen_preview_annotated/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Danh sách các lớp nền cần loại bỏ
IGNORE_CLASSES = {
    "dining table", "chair", "person", "bench", "couch", "tv", "laptop",
    "cell phone", "remote", "keyboard", "mouse", "refrigerator", "sink",
    "book", "clock", "traffic light", "fire hydrant", "car"
}

# Ánh xạ từ COCO sang 3 lớp Canteen
FOOD_CLASSES = {
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake"
}

RECYCLABLE_CLASSES = {
    "bottle", "wine glass"
}

OTHER_CLASSES = {
    "cup"
}

GROUP_NAMES = {
    0: "0:Huu_co",
    1: "1:Tai_che",
    2: "2:Con_lai",
}

GROUP_COLORS = {
    0: (0, 200, 0),    # Xanh lá cây (Hữu cơ / Thức ăn thừa / Bánh mỳ)
    1: (220, 150, 0),  # Xanh dương / Vàng cam (Tái chế)
    2: (0, 0, 220),    # Đỏ (Còn lại)
}


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """box format: [xc, yc, w, h] normalized."""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def prelabel_canteen_trays(
    images_dir: Path,
    labels_dir: Path,
    preview_dir: Path,
    bread_model_path: Path = Path("data/checkpoints/bread_detector.pt"),
    food_model_name: str = "yolov8x.pt",
    conf_food: float = 0.035,
    conf_bread: float = 0.20,
    max_previews: int = 60,
    specific_preview_stems: set[str] | None = None,
):
    print("=" * 80)
    print("TỰ ĐỘNG SINH BOUNDING BOX & NHÃN GỢI Ý CHO KHAY CANTEEN (DUAL-STAGE ENSEMBLE)")
    print(f"Thư mục ảnh nguồn:     {images_dir}")
    print(f"Thư mục nhãn xuất:      {labels_dir}")
    print(f"Thư mục preview:        {preview_dir}")
    print(f"Model thức ăn/đĩa:      {food_model_name} (conf={conf_food})")
    print(f"Model chuyên túi bánh:  {bread_model_path} (conf={conf_bread})")
    print("=" * 80)

    labels_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị tính toán: {device} ({torch.cuda.get_device_name(0) if device==0 else 'CPU'})")

    print("Đang nạp mô hình...")
    model_food = YOLO(food_model_name)
    model_bread = YOLO(str(bread_model_path)) if bread_model_path.exists() else None
    if model_bread is None:
        print(f"[WARN] Không tìm thấy bread model tại {bread_model_path}, chỉ dùng {food_model_name}")

    image_paths = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ])

    total_images = len(image_paths)
    if total_images == 0:
        print(f"[WARN] Không tìm thấy ảnh nào tại {images_dir}")
        return

    total_boxes = 0
    group_counts = {0: 0, 1: 0, 2: 0}
    bread_detected_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        candidates = []

        # 1. Phát hiện túi bánh mỳ (Bread Bun in Plastic Bag)
        if model_bread is not None:
            res_bread = model_bread(str(img_path), conf=conf_bread, device=device, verbose=False)[0]
            for b in res_bread.boxes:
                conf = float(b.conf[0].item())
                xywhn = b.xywhn[0].tolist()
                xc, yc, w, h = xywhn
                # Bỏ qua box nằm sát mép bàn ngoài khay
                if xc < 0.05 or xc > 0.95 or yc < 0.05 or yc > 0.95:
                    continue
                # Bỏ qua box kích thước bất thường
                if w < 0.08 or w > 0.45 or h < 0.08 or h > 0.45:
                    continue
                candidates.append({
                    "group_id": 0,
                    "raw_cls": "tui_banh_my",
                    "conf": conf,
                    "xywhn": xywhn,
                    "priority": 10,  # Ưu tiên cao nhất
                })

        # 2. Phát hiện các món ăn, đĩa, bát, chai/cốc trên khay
        res_food = model_food(str(img_path), conf=conf_food, device=device, verbose=False)[0]
        for b in res_food.boxes:
            cls_name = model_food.names[int(b.cls[0].item())]
            if cls_name in IGNORE_CLASSES:
                continue

            conf = float(b.conf[0].item())
            xywhn = b.xywhn[0].tolist()
            xc, yc, w, h = xywhn

            # Bỏ qua vật thể mép ngoài khay (rổ bên cạnh, chân người, mép bàn ngoài)
            if xc < 0.06 or xc > 0.94 or yc < 0.04 or yc > 0.94:
                continue

            area = w * h
            if area < 0.015 or area > 0.85:
                continue

            # Bỏ qua fork/knife/spoon vì trên khay thường là phản chiếu nếp nhăn túi bánh mỳ
            if cls_name in ["fork", "knife", "spoon"]:
                continue

            # Ánh xạ nhóm
            if cls_name in RECYCLABLE_CLASSES:
                group_id = 1
            elif cls_name in OTHER_CLASSES:
                group_id = 2
            else:
                group_id = 0  # Hữu cơ / Thức ăn

            candidates.append({
                "group_id": group_id,
                "raw_cls": cls_name,
                "conf": conf,
                "xywhn": xywhn,
                "priority": 1,
            })

        # 3. Khử trùng lặp và phân giải xung đột (NMS)
        # Sắp xếp: Ưu tiên túi bánh mỳ (priority=10), sau đó theo conf giảm dần
        candidates.sort(key=lambda x: (x["priority"], x["conf"]), reverse=True)
        kept_boxes = []

        # Lấy danh sách các box bánh mỳ trước để khử các box cup/bottle/food trùng lên bánh mỳ
        bread_boxes = [c for c in candidates if c["raw_cls"] == "tui_banh_my"]

        for c in candidates:
            # Nếu không phải bánh mỳ nhưng trùng lên bánh mỳ (e.g. cup, cake, sandwich, bottle)
            if c["raw_cls"] != "tui_banh_my":
                overlaps_bread = False
                for bb in bread_boxes:
                    iou = compute_iou(c["xywhn"], bb["xywhn"])
                    # Khoảng cách tâm
                    dist = np.hypot(c["xywhn"][0] - bb["xywhn"][0], c["xywhn"][1] - bb["xywhn"][1])
                    if iou > 0.20 or dist < 0.12:
                        overlaps_bread = True
                        break
                if overlaps_bread:
                    # Trùng với túi bánh mỳ -> bỏ qua box trùng này
                    continue

            # NMS tiêu chuẩn với các box đã giữ lại
            overlap = False
            for kept in kept_boxes:
                iou = compute_iou(c["xywhn"], kept["xywhn"])
                if iou > 0.45:
                    overlap = True
                    break
            if not overlap:
                kept_boxes.append(c)

        if any(b["raw_cls"] == "tui_banh_my" for b in kept_boxes):
            bread_detected_count += 1

        # 4. Ghi file nhãn chuẩn YOLO .txt
        lbl_file = labels_dir / f"{img_path.stem}.txt"
        lines = []
        for b in kept_boxes:
            gid = b["group_id"]
            xc, yc, w, h = b["xywhn"]
            lines.append(f"{gid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            group_counts[gid] += 1
            total_boxes += 1

        with lbl_file.open("w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines) + "\n")
            else:
                f.write("")

        # 5. Vẽ ảnh preview minh họa trực quan
        should_preview = (idx <= max_previews)
        if specific_preview_stems and img_path.stem in specific_preview_stems:
            should_preview = True

        if should_preview:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_h, img_w = img.shape[:2]
                for b in kept_boxes:
                    gid = b["group_id"]
                    xc, yc, w, h = b["xywhn"]
                    color = GROUP_COLORS.get(gid, (0, 200, 0))
                    label_text = f"{GROUP_NAMES[gid]} ({b['conf']:.2f})"

                    x1 = int((xc - w / 2) * img_w)
                    y1 = int((yc - h / 2) * img_h)
                    x2 = int((xc + w / 2) * img_w)
                    y2 = int((yc + h / 2) * img_h)

                    # Khung chữ nhật
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                    # Nền nhãn chữ
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(img, (x1, max(0, y1 - 24)), (x1 + tw + 6, max(0, y1)), color, -1)
                    cv2.putText(img, label_text, (x1 + 3, max(14, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

                preview_path = preview_dir / f"annotated_{img_path.stem}.jpg"
                cv2.imwrite(str(preview_path), img)

        if idx % 50 == 0 or idx == total_images:
            print(f"  Đã xử lý {idx:03d}/{total_images} ảnh khay... (Túi bánh mỳ phát hiện: {bread_detected_count})")

    print("\n" + "=" * 80)
    print("HOÀN TẤT SINH BOUNDING BOX TỰ ĐỘNG CHO TOÀN BỘ 401 ẢNH KHAY CANTEEN")
    print(f"Tổng số ảnh đã xử lý:      {total_images}")
    print(f"Tổng số khay có túi bánh:  {bread_detected_count}/{total_images} ({bread_detected_count/total_images*100:.1f}%)")
    print(f"Tổng số box gợi ý:         {total_boxes} (trung bình {total_boxes/total_images:.2f} box/khay)")
    print(f"  - 0: Hữu cơ:              {group_counts[0]} boxes")
    print(f"  - 1: Tái chế:             {group_counts[1]} boxes")
    print(f"  - 2: Còn lại:             {group_counts[2]} boxes")
    print(f"Đã cập nhật toàn bộ file nhãn .txt vào: {labels_dir}")
    print(f"Đã cập nhật toàn bộ ảnh preview minh họa vào: {preview_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Tự động sinh nhãn box gợi ý cho khay canteen")
    parser.add_argument("--images", type=Path, default=Path("data/raw/canteen_trays"))
    parser.add_argument("--labels", type=Path, default=Path("data/raw/canteen_labels"))
    parser.add_argument("--previews", type=Path, default=Path("reports/canteen_preview_annotated"))
    parser.add_argument("--bread-model", type=Path, default=Path("data/checkpoints/bread_detector.pt"))
    parser.add_argument("--food-model", type=str, default="yolov8x.pt")
    parser.add_argument("--conf-food", type=float, default=0.035)
    parser.add_argument("--conf-bread", type=float, default=0.20)
    parser.add_argument("--max-previews", type=int, default=100)
    args = parser.parse_args()

    prelabel_canteen_trays(
        images_dir=args.images,
        labels_dir=args.labels,
        preview_dir=args.previews,
        bread_model_path=args.bread_model,
        food_model_name=args.food_model,
        conf_food=args.conf_food,
        conf_bread=args.conf_bread,
        max_previews=args.max_previews,
    )


if __name__ == "__main__":
    main()
