"""scripts/eval_detector.py

D2-C3: Đánh giá detector class-agnostic (spec 8.5).
Báo cáo: AP50, AP50-95, precision, recall@IoU0.5 tại conf cố định
(mặc định 0.25; thêm 0.05 và 0.001).

Sử dụng:
    python scripts/eval_detector.py --weights runs/detect/d1a_taco/weights/best.pt \
        --data-yaml data/processed/taco_yolo_agnostic/data.yaml \
        --split val --imgsz 416
"""
from __future__ import annotations

import sys
import io
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes in [x1,y1,x2,y2] format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> tuple[int, int, int]:
    """Greedy matching: sort preds by confidence, match each to best unmatched GT.

    Returns (TP, FP, FN).
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    order = np.argsort(-pred_scores)
    matched_gt = set()
    tp = 0
    fp = 0

    for idx in order:
        best_iou = 0.0
        best_gt = -1
        for g, gt_box in enumerate(gt_boxes):
            if g in matched_gt:
                continue
            iou = compute_iou(pred_boxes[idx], gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = g
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def load_yolo_labels(label_dir: Path, img_dir: Path) -> dict[str, np.ndarray]:
    """Load YOLO label files, converting normalized coords to absolute xyxy."""
    from PIL import Image

    labels = {}
    for txt_file in sorted(label_dir.glob("*.txt")):
        stem = txt_file.stem
        # Find corresponding image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        img = Image.open(img_path)
        w, h = img.size
        img.close()

        boxes = []
        with open(txt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # class_id, xc, yc, bw, bh (normalized)
                xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                x2 = (xc + bw / 2) * w
                y2 = (yc + bh / 2) * h
                boxes.append([x1, y1, x2, y2])
        labels[stem] = np.array(boxes) if boxes else np.empty((0, 4))
    return labels


def eval_recall_at_conf(
    model,
    data_yaml: str,
    split: str,
    imgsz: int,
    conf_thresholds: list[float],
    iou_thresh: float = 0.5,
) -> dict[float, dict[str, float]]:
    """Evaluate recall/precision at fixed conf thresholds using greedy matching."""
    import yaml

    with open(data_yaml, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get("path", Path(data_yaml).parent))
    split_rel = data_cfg.get(split, split)
    img_dir = base_path / split_rel
    # Labels dir: sibling 'labels' directory
    lbl_dir = img_dir.parent.parent / "labels" / img_dir.name

    gt_labels = load_yolo_labels(lbl_dir, img_dir)

    results = {}
    for conf in conf_thresholds:
        # Run inference
        preds = model.predict(
            source=str(img_dir),
            conf=conf,
            iou=0.7,  # NMS IoU
            imgsz=imgsz,
            verbose=False,
            save=False,
        )

        total_tp, total_fp, total_fn = 0, 0, 0
        for pred in preds:
            stem = Path(pred.path).stem
            gt_boxes = gt_labels.get(stem, np.empty((0, 4)))
            pred_boxes = pred.boxes.xyxy.cpu().numpy()
            pred_scores = pred.boxes.conf.cpu().numpy()

            tp, fp, fn = greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thresh)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[conf] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "TP": total_tp,
            "FP": total_fp,
            "FN": total_fn,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Đánh giá detector class-agnostic (D2-C3)")
    parser.add_argument("--weights", type=str, required=True, help="Đường dẫn đến checkpoint detector (.pt)")
    parser.add_argument("--data-yaml", type=str, required=True, help="File data.yaml cho dataset")
    parser.add_argument("--split", type=str, default="val", choices=["val", "heldout", "test"],
                        help="Tập dữ liệu để đánh giá")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.05, 0.001],
                        help="Ngưỡng confidence cố định")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh đầu vào")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="Ngưỡng IoU để ghép box")
    parser.add_argument("--out-csv", type=str, default=None, help="Đường dẫn xuất CSV kết quả")
    args = parser.parse_args()

    # Test guard: chặn đọc Test ngoài eval_test_once.py
    if args.split == "test":
        try:
            from dsr.test_guard import assert_not_test_split
            assert_not_test_split("eval_detector.py")
        except ImportError:
            import os
            if os.environ.get("DSR_ALLOW_TEST") != "1":
                print("[LỖI] Không được đánh giá trên split 'test' ngoài eval_test_once.py.")
                print("      Đặt biến môi trường DSR_ALLOW_TEST=1 nếu đang chạy từ eval_test_once.py.")
                sys.exit(1)

    from ultralytics import YOLO

    print(f"[*] Nạp model từ: {args.weights}")
    model = YOLO(args.weights)

    # 1. ultralytics val() cho AP50, AP50-95
    print(f"\n[*] Chạy ultralytics val() trên split '{args.split}'...")
    val_results = model.val(
        data=args.data_yaml,
        split=args.split,
        imgsz=args.imgsz,
        verbose=False,
    )

    ap50 = val_results.box.map50
    ap50_95 = val_results.box.map
    ul_precision = val_results.box.mp
    ul_recall = val_results.box.mr

    print(f"\n{'='*60}")
    print(f"  ULTRALYTICS METRICS (split={args.split})")
    print(f"{'='*60}")
    print(f"  AP50:      {ap50:.4f}")
    print(f"  AP50-95:   {ap50_95:.4f}")
    print(f"  Precision: {ul_precision:.4f}")
    print(f"  Recall:    {ul_recall:.4f}")

    # 2. Recall@conf cố định bằng greedy matching thủ công
    print(f"\n{'='*60}")
    print(f"  RECALL @ FIXED CONF (IoU≥{args.iou_thresh})")
    print(f"{'='*60}")
    print(f"  {'Conf':>8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>6}  {'FP':>6}  {'FN':>6}")
    print(f"  {'-'*62}")

    recall_results = eval_recall_at_conf(
        model, args.data_yaml, args.split, args.imgsz, args.conf, args.iou_thresh
    )

    rows = []
    for conf in sorted(recall_results.keys(), reverse=True):
        r = recall_results[conf]
        print(f"  {conf:>8.3f}  {r['precision']:>10.4f}  {r['recall']:>8.4f}  {r['f1']:>8.4f}  "
              f"{r['TP']:>6d}  {r['FP']:>6d}  {r['FN']:>6d}")
        rows.append({
            "split": args.split,
            "conf": conf,
            "iou_thresh": args.iou_thresh,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "TP": r["TP"],
            "FP": r["FP"],
            "FN": r["FN"],
            "AP50": ap50,
            "AP50_95": ap50_95,
        })

    # 3. Xuất CSV
    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[*] Kết quả đã ghi ra: {out_path}")

    print(f"\n[THÀNH CÔNG] Đánh giá detector hoàn tất.")


if __name__ == "__main__":
    main()
