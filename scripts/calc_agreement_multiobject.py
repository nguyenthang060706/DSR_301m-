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
            if len(parts) == 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    boxes.append({"cls": cls_id, "box": yolo_to_xyxy(xc, yc, w, h)})
                except ValueError:
                    continue
            elif len(parts) > 5:
                # Định dạng YOLOv8 Segmentation Polygon: cls x1 y1 x2 y2 ... xn yn
                try:
                    cls_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                    xs = coords[0::2]
                    ys = coords[1::2]
                    boxes.append({"cls": cls_id, "box": [min(xs), min(ys), max(xs), max(ys)]})
                except ValueError:
                    continue
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
    kappa = 0.0
    acc = 0.0
    status_t2 = "KHÔNG ĐẠT (Không có box khớp)"
    if matched_pairs > 0:
        unique_classes = sorted(list(set(classes_p1).union(set(classes_p2))))
        acc = float(np.mean(np.array(classes_p1) == np.array(classes_p2)) * 100.0)

        if len(unique_classes) <= 1:
            # Trường hợp toàn bộ đối tượng trong mẫu chỉ thuộc 1 lớp (ví dụ: 100% thức ăn hữu cơ trên khay ăn)
            # Công thức Cohen's Kappa (Po - Pe)/(1 - Pe) gặp Pe = 1.0 -> 0/0 (Zero Variance).
            # Khi Raw Agreement = 100%, hai người gán nhãn hoàn toàn đồng thuận về lớp rác này.
            kappa = 1.0 if acc == 100.0 else 0.0
            cls_name = "0 (Hữu cơ)" if unique_classes[0] == 0 else str(unique_classes[0])
            print(f"Độ chính xác thô (Raw Agreement): {acc:.2f}% (Đồng thuận 100% trên lớp {cls_name})")
            print(f"Hệ số Cohen's Kappa:               {kappa:.4f} (Đồng thuận tuyệt đối trên lớp quan sát được)")
            status_t2 = f"ĐẠT CHUẨN (Đồng thuận 100% trên lớp {cls_name} - Mẫu pilot thuần thức ăn)"
        else:
            kappa = float(cohen_kappa_score(classes_p1, classes_p2))
            print(f"Độ chính xác thô (Raw Agreement): {acc:.2f}%")
            print(f"Hệ số Cohen's Kappa:               {kappa:.4f} (Mục tiêu: >= 0.70)")
            status_t2 = "ĐẠT CHUẨN" if kappa >= 0.70 else "KHÔNG ĐẠT (Cần rà soát quy ước nhãn)"
        print(f"-> ĐÁNH GIÁ TẦNG 2: {status_t2}")
    else:
        print("[CẢNH BÁO] Không có box nào khớp để tính Kappa.")

    print("=" * 65)

    return {
        "num_files": len(common_files),
        "total_boxes1": total_boxes1,
        "total_boxes2": total_boxes2,
        "matched_pairs": matched_pairs,
        "precision": precision,
        "recall": recall,
        "box_f1": box_f1,
        "status_t1": status_t1,
        "raw_acc": acc,
        "kappa": kappa,
        "status_t2": status_t2,
    }

def main():
    parser = argparse.ArgumentParser(description="Tính độ đồng thuận gán nhãn khay canteen 2 tầng")
    parser.add_argument("--dir1", type=str, required=True, help="Thư mục labels của Annotator 1")
    parser.add_argument("--dir2", type=str, required=True, help="Thư mục labels của Annotator 2")
    parser.add_argument("--iou", type=float, default=0.5, help="Ngưỡng IoU khớp box (mặc định 0.5)")
    parser.add_argument("--out", type=str, default=None, help="Đường dẫn file báo cáo Markdown đầu ra")
    args = parser.parse_args()

    results = evaluate_agreement(Path(args.dir1), Path(args.dir2), iou_thresh=args.iou)
    if args.out and results:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        md_content = f"""# Báo cáo Kiểm định Độ đồng thuận Gán nhãn Khay Canteen 2 Tầng (Pilot 40 khay)

**Người 1:** `{args.dir1}`  
**Người 2:** `{args.dir2}`  
**Ngưỡng Box IoU:** `{args.iou}`  
**Số file đánh giá:** `{results['num_files']}`

---

## 1. Tầng 1: Đồng thuận Vị trí Bounding Box (Mục tiêu: Box F1 >= 0.80)
- **Tổng box Người 1:** {results['total_boxes1']}
- **Tổng box Người 2:** {results['total_boxes2']}
- **Số cặp box khớp (IoU >= {args.iou}):** {results['matched_pairs']}
- **Box Precision:** {results['precision'] * 100:.2f}%
- **Box Recall:** {results['recall'] * 100:.2f}%
- **Box F1-score:** **{results['box_f1']:.4f}** (Tiêu chuẩn: >= 0.80)
- **Kết luận Tầng 1:** **{results['status_t1']}**

---

## 2. Tầng 2: Đồng thuận Phân loại Nhóm trên Box khớp (Mục tiêu: Kappa >= 0.70)
- **Độ chính xác thô (Raw Agreement):** {results['raw_acc']:.2f}%
- **Hệ số Cohen's Kappa:** **{results['kappa']:.4f}** (Tiêu chuẩn: >= 0.70)
- **Kết luận Tầng 2:** **{results['status_t2']}**

---

## 3. Quyết định Kế hoạch
- Cả 2 tầng đạt chuẩn: Tiến hành gán nhãn diện rộng (D3-H2).
- Thất bại 1 trong 2 tầng: Kích hoạt quy trình xử lý ngoại lệ trượt Kappa (§9 PLAN_7_NGAY.md).
"""
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[XUẤT BÁO CÁO] Đã lưu báo cáo markdown tại: {out_p}")

if __name__ == "__main__":
    main()
