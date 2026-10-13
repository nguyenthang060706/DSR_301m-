"""scripts/calc_agreement_multiobject.py

Kiểm định độ đồng thuận gán nhãn khay canteen 2 tầng (Two-Tier Inter-Annotator Agreement)
theo Kế hoạch MASTER v10.0 (§3.4) và docs/label_guide.md v2.0:
- Tầng 1: Đồng thuận vị trí Bounding Box (Box IoU >= 0.5, yêu cầu Box F1 >= 0.80).
- Tầng 2: Đồng thuận phân loại rác trên các box đã khớp (Cohen's Kappa >= 0.70).
"""

import sys
import io
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def yolo_to_xyxy(xc, yc, w, h):
    return [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0]

def compute_box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area

def load_yolo_labels(file_path):
    boxes = []
    if not file_path.exists():
        return boxes
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append({"cls": cls_id, "box": yolo_to_xyxy(xc, yc, w, h)})
    return boxes

def evaluate_agreement(dir1: Path, dir2: Path, iou_thresh=0.5):
    print("=" * 65)
    print("KIỂM ĐỊNH ĐỘ ĐỒNG THUẬN GÁN NHÃN KHAY CANTEEN 2 TẦNG (MASTER v10.0)")
    print(f"Người 1 (Annotator 1): {dir1}")
    print(f"Người 2 (Annotator 2): {dir2}")
    print(f"Ngưỡng Box IoU:         {iou_thresh}")
    print("=" * 65)

    files1 = set([p.name for p in dir1.glob("*.txt")])
    files2 = set([p.name for p in dir2.glob("*.txt")])
    common_files = sorted(list(files1.intersection(files2)))

    if not common_files:
        print("[LỖI] Không tìm thấy file nhãn trùng khớp nào giữa 2 thư mục!")
        return

    print(f"[*] Số khay đánh giá chung: {len(common_files)} file")

    total_boxes1 = 0
    total_boxes2 = 0
    matched_pairs = 0

    classes_p1 = []
    classes_p2 = []

    for fname in common_files:
        b1_list = load_yolo_labels(dir1 / fname)
        b2_list = load_yolo_labels(dir2 / fname)

        total_boxes1 += len(b1_list)
        total_boxes2 += len(b2_list)

        # Khớp box theo greedy IoU
        used2 = set()
        for item1 in b1_list:
            best_iou = 0.0
            best_idx2 = -1
            for idx2, item2 in enumerate(b2_list):
                if idx2 in used2:
                    continue
                iou = compute_box_iou(item1["box"], item2["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx2 = idx2

            if best_iou >= iou_thresh and best_idx2 >= 0:
                used2.add(best_idx2)
                matched_pairs += 1
                classes_p1.append(item1["cls"])
                classes_p2.append(b2_list[best_idx2]["cls"])

    # TẦNG 1: ĐỒNG THUẬN VỊ TRÍ BOX
    precision = matched_pairs / max(1, total_boxes1)
    recall = matched_pairs / max(1, total_boxes2)
    box_f1 = 2 * matched_pairs / max(1, total_boxes1 + total_boxes2)

    print("\n--- [TẦNG 1: ĐỒNG THUẬN VỊ TRÍ BOUNDING BOX] ---")
    print(f"Tổng số box Người 1: {total_boxes1}")
    print(f"Tổng số box Người 2: {total_boxes2}")
    print(f"Số cặp box khớp (IoU >= {iou_thresh}): {matched_pairs}")
    print(f"Box Precision (P1 match P2): {precision * 100:.2f}%")
    print(f"Box Recall    (P2 match P1): {recall * 100:.2f}%")
    print(f"Box F1-score:                {box_f1:.4f} (Mục tiêu: >= 0.80)")
    status_t1 = "ĐẠT CHUẨN" if box_f1 >= 0.80 else "KHÔNG ĐẠT (Cần họp quy ước box)"
    print(f"-> ĐÁNH GIÁ TẦNG 1: {status_t1}")

    # TẦNG 2: ĐỒNG THUẬN PHÂN LOẠI NHÓM
    print("\n--- [TẦNG 2: ĐỒNG THUẬN PHÂN LOẠI NHÓM TRÊN BOX KHỚP] ---")
    if matched_pairs > 0:
        kappa = cohen_kappa_score(classes_p1, classes_p2)
        acc = np.mean(np.array(classes_p1) == np.array(classes_p2)) * 100.0
        print(f"Độ chính xác thô (Raw Agreement): {acc:.2f}%")
        print(f"Hệ số Cohen's Kappa:               {kappa:.4f} (Mục tiêu: >= 0.70)")
        status_t2 = "ĐẠT CHUẨN" if kappa >= 0.70 else "KHÔNG ĐẠT (Cần rà soát quy ước nhãn)"
        print(f"-> ĐÁNH GIÁ TẦNG 2: {status_t2}")
    else:
        print("[CẢNH BÁO] Không có box nào khớp để tính Kappa.")

    print("=" * 65)

def main():
    parser = argparse.ArgumentParser(description="Tính độ đồng thuận gán nhãn khay canteen 2 tầng")
    parser.add_argument("--dir1", type=str, required=True, help="Thư mục labels của Annotator 1")
    parser.add_argument("--dir2", type=str, required=True, help="Thư mục labels của Annotator 2")
    parser.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU khớp box (mặc định 0.5)")
    args = parser.parse_args()

    evaluate_agreement(Path(args.dir1), Path(args.dir2), iou_thresh=args.iou)

if __name__ == "__main__":
    main()
