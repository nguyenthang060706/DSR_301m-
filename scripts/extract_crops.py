"""scripts/extract_crops.py

D3-C3 / Spec 8.8: Trích xuất các box vật thể (crops) từ khay ăn canteen để phục vụ
đánh giá 2 giai đoạn (Detector -> Crop -> Classifier), Pha 2 (hard-negative background) và Few-shot.

Hỗ trợ:
- Hai chế độ cắt box:
  + --boxes gt : Cắt theo ground truth (Oracle Boxes - trần lý thuyết của Classifier)
  + --boxes det: Cắt theo box do Detector phát hiện (Detected Boxes)
- Xử lý IoU đa tầng khi --boxes det:
  + max_IoU >= 0.5: box khớp vật thật, gt_group theo GT
  + 0.1 <= max_IoU < 0.5: định vị kém -> gt_group = "ambiguous_loc" (loại khỏi train/eval)
  + max_IoU < 0.1: dương tính giả -> gt_group = "background" (làm mẫu hard-negative)
- Giới hạn tối đa N_max=8 box trên 1 khay (sắp xếp theo confidence giảm dần).
- Mở rộng viền box (padding) 10-20% (mặc định --pad 0.15).
- Tích hợp test_guard: Chặn nạp tập Test trước Ngày 6.
- Xuất thư mục crop và crops_manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

from PIL import Image

sys.path.insert(0, "src")
from dsr.test_guard import assert_not_test, is_test_allowed


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def yolo_to_pixel_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> list[int]:
    """Chuyển tọa độ YOLO chuẩn hóa sang tọa độ pixel [x1, y1, x2, y2]."""
    px_w = w * img_w
    px_h = h * img_h
    x1 = int(round((xc * img_w) - (px_w / 2.0)))
    y1 = int(round((yc * img_h) - (px_h / 2.0)))
    x2 = int(round((xc * img_w) + (px_w / 2.0)))
    y2 = int(round((yc * img_h) + (px_h / 2.0)))
    return [max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)]


def expand_box_with_padding(box: Sequence[int], img_w: int, img_h: int, pad_ratio: float = 0.15) -> list[int]:
    """Mở rộng viền box theo pad_ratio (mặc định 15%) và clamp vào kích thước ảnh."""
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad_w = int(round(w * pad_ratio))
    pad_h = int(round(h * pad_ratio))
    return [
        max(0, x1 - pad_w),
        max(0, y1 - pad_h),
        min(img_w, x2 + pad_w),
        min(img_h, y2 + pad_h),
    ]


def compute_iou_xyxy(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Tính Intersection over Union (IoU) giữa 2 box [x1, y1, x2, y2]."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def load_gt_boxes_from_txt(lbl_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Nạp danh sách bounding box Ground Truth từ file nhãn YOLO."""
    boxes = []
    if not lbl_path.exists():
        return boxes

    with open(lbl_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    pixel_box = yolo_to_pixel_xyxy(xc, yc, w, h, img_w, img_h)
                    boxes.append({
                        "cls_id": cls_id,
                        "box": pixel_box,
                        "raw_yolo": (xc, yc, w, h),
                    })
                except ValueError:
                    continue
    return boxes


def parse_args():
    parser = argparse.ArgumentParser(description="Trích xuất crops vật thể từ khay ăn (Spec 8.8)")
    parser.add_argument("--split", type=str, required=True, choices=["val", "pool", "pilot", "test"],
                        help="Tập dữ liệu cần cắt crop: val, pool, pilot, hoặc test")
    parser.add_argument("--boxes", type=str, default="gt", choices=["gt", "det"],
                        help="Nguồn bounding box: gt (Ground Truth) hoặc det (Detector)")
    parser.add_argument("--pad", type=float, default=0.15, help="Tỷ lệ padding mở rộng viền box (default: 0.15)")
    parser.add_argument("--nmax", type=int, default=8, help="Số box tối đa trên 1 khay (default: 8)")
    parser.add_argument("--det-weights", type=Path, default=None, help="Đường dẫn weights detector nếu --boxes det")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng confidence detector (default: 0.25)")
    parser.add_argument("--nms", type=float, default=0.45, help="Ngưỡng IoU NMS detector (default: 0.45)")
    parser.add_argument("--metadata", type=Path, default=Path("data/canteen_metadata.csv"), help="File metadata")
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/canteen_trays"), help="Thư mục ảnh khay gốc")
    parser.add_argument("--labels-dir", type=Path, default=Path("data/raw/canteen_labels"), help="Thư mục nhãn GT gốc")
    parser.add_argument("--date-plan", type=Path, default=Path("configs/canteen_date_plan.json"), help="Kế hoạch ngày")
    parser.add_argument("--pilot-list", type=Path, default=Path("data/kappa_pilot_list.txt"), help="Danh sách ảnh pilot")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/canteen_crops"), help="Thư mục lưu crop")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print(f"TRÍCH XUẤT CROPS CANTEEN (Spec 8.8) | Split: {args.split.upper()} | Boxes: {args.boxes.upper()}")
    print(f"Padding: {args.pad:.2f} | N_max: {args.nmax} | Out: {args.out_dir}")
    print("=" * 68)

    # 1. Bảo vệ Test bằng Test Guard (chặn tuyệt đối nạp Test trước Ngày 6)
    if args.split == "test" and not is_test_allowed():
        raise RuntimeError(
            "[PROTOCOL VIOLATION - TEST GUARD] Tập Test canteen chưa được mở! "
            "Trích xuất crop trên Test chỉ được phép thực hiện một lần duy nhất tại Ngày 6."
        )

    # 2. Xác định danh sách ảnh khay thuộc split
    target_rows: list[dict] = []

    if args.split == "pilot":
        # Nạp từ danh sách pilot
        pilot_stems = set()
        if args.pilot_list.exists():
            with open(args.pilot_list, "r", encoding="utf-8") as f:
                pilot_stems = set(ln.strip() for ln in f if ln.strip())

        if args.metadata.exists():
            with open(args.metadata, "r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r["image_id"] in pilot_stems or Path(r["image_id"]).stem in pilot_stems:
                        target_rows.append(r)
        if not target_rows and pilot_stems:
            # Fallback nếu metadata chưa có pilot rows
            for stem in pilot_stems:
                target_rows.append({
                    "image_id": f"{stem}.png" if not stem.endswith(".png") else stem,
                    "cluster_id": "pilot_cluster",
                })
    else:
        # Nạp theo date plan
        if not args.date_plan.exists():
            raise FileNotFoundError(f"Chưa có file kế hoạch ngày: {args.date_plan}")
        with open(args.date_plan, "r", encoding="utf-8") as f:
            plan_data = json.load(f)

        split_dates = set()
        for d in plan_data.get(args.split, []):
            d_str = str(d).strip().replace("/", "-")
            split_dates.add(d_str)
            split_dates.add(d_str.replace("-", ""))

        with open(args.metadata, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = r.get("date", "").strip()
                if not d and "cluster_id" in r:
                    d = r["cluster_id"].split("_")[0]
                d_clean = d.replace("/", "-")
                if d_clean in split_dates or d_clean.replace("-", "") in split_dates:
                    target_rows.append(r)

    print(f"[*] Tìm thấy {len(target_rows)} khay ăn thuộc tập {args.split}")

    # Guard kiểm tra danh sách ảnh
    image_paths_to_check = [r["image_id"] for r in target_rows]
    if args.split != "test":
        assert_not_test(image_paths_to_check, date_plan_path=args.date_plan)

    # 3. Khởi tạo detector nếu --boxes det
    det_model = None
    if args.boxes == "det":
        if args.det_weights and args.det_weights.exists():
            print(f"[*] Nạp trọng số YOLO Detector từ: {args.det_weights}")
            from ultralytics import YOLO
            det_model = YOLO(str(args.det_weights))
        else:
            print("[CẢNH BÁO] Không có --det-weights hoặc file không tồn tại. Chế độ det sẽ sử dụng pseudo detector (mock).")

    # 4. Tạo thư mục đích
    crop_dest_dir = args.out_dir / f"{args.split}_{args.boxes}"
    crop_dest_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    total_crops = 0
    group_stats: dict[str, int] = {}

    for row_idx, r in enumerate(target_rows):
        img_file = r["image_id"]
        cid = r.get("cluster_id", "canteen_cluster")
        img_path = args.images_dir / img_file

        if not img_path.exists():
            continue

        try:
            with Image.open(img_path) as pil_img:
                img_rgb = pil_img.convert("RGB")
                img_w, img_h = img_rgb.size

                lbl_file = Path(img_file).stem + ".txt"
                lbl_path = args.labels_dir / lbl_file
                gt_boxes = load_gt_boxes_from_txt(lbl_path, img_w, img_h)

                candidate_boxes = []

                if args.boxes == "gt":
                    for g_idx, g in enumerate(gt_boxes):
                        candidate_boxes.append({
                            "box": g["box"],
                            "source": "gt",
                            "gt_group": str(g["cls_id"]),
                            "iou_with_gt": 1.0,
                            "conf": 1.0,
                        })
                else:
                    # Chế độ --boxes det
                    raw_dets = []
                    if det_model is not None:
                        results = det_model.predict(img_rgb, conf=args.conf, iou=args.nms, verbose=False)
                        for res in results:
                            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                            confs = res.boxes.conf.cpu().numpy()
                            for b, c in zip(boxes_xyxy, confs):
                                raw_dets.append({
                                    "box": [int(round(x)) for x in b],
                                    "conf": float(c),
                                })
                    else:
                        # Fallback pseudo detector khi test mock
                        for g in gt_boxes:
                            raw_dets.append({
                                "box": g["box"],
                                "conf": 0.95,
                            })

                    # Sắp xếp giảm dần theo confidence, lấy top nmax
                    raw_dets = sorted(raw_dets, key=lambda x: x["conf"], reverse=True)[:args.nmax]

                    for d_item in raw_dets:
                        det_box = d_item["box"]
                        # Tính IoU cực đại với toàn bộ GT boxes
                        max_iou = 0.0
                        best_cls = -1
                        for g in gt_boxes:
                            iou = compute_iou_xyxy(det_box, g["box"])
                            if iou > max_iou:
                                max_iou = iou
                                best_cls = g["cls_id"]

                        # Xử lý đa tầng
                        if max_iou >= 0.5 and best_cls >= 0:
                            gt_group = str(best_cls)
                        elif 0.1 <= max_iou < 0.5:
                            gt_group = "ambiguous_loc"  # Loại khỏi train/eval
                        else:
                            gt_group = "background"    # Mẫu hard-negative

                        candidate_boxes.append({
                            "box": det_box,
                            "source": "det",
                            "gt_group": gt_group,
                            "iou_with_gt": round(max_iou, 4),
                            "conf": round(d_item["conf"], 4),
                        })

                # Thực hiện cắt crop và lưu ảnh
                for c_idx, c_info in enumerate(candidate_boxes):
                    padded_box = expand_box_with_padding(c_info["box"], img_w, img_h, pad_ratio=args.pad)
                    x1, y1, x2, y2 = padded_box
                    if x2 <= x1 or y2 <= y1:
                        continue

                    crop_img = img_rgb.crop((x1, y1, x2, y2))
                    crop_id = f"{Path(img_file).stem}_c{c_idx:02d}"
                    crop_filename = f"{crop_id}.jpg"
                    crop_save_path = crop_dest_dir / crop_filename
                    crop_img.save(crop_save_path, "JPEG", quality=95)

                    grp = c_info["gt_group"]
                    group_stats[grp] = group_stats.get(grp, 0) + 1
                    total_crops += 1

                    manifest_rows.append({
                        "crop_id": crop_id,
                        "image_id": img_file,
                        "cluster_id": cid,
                        "crop_file": f"{args.split}_{args.boxes}/{crop_filename}",
                        "box_xyxy": f"{c_info['box'][0]}_{c_info['box'][1]}_{c_info['box'][2]}_{c_info['box'][3]}",
                        "padded_xyxy": f"{x1}_{y1}_{x2}_{y2}",
                        "source": c_info["source"],
                        "gt_group": grp,
                        "iou_with_gt": c_info["iou_with_gt"],
                        "conf": c_info["conf"],
                    })
        except Exception as e:
            print(f"[CẢNH BÁO] Không thể xử lý {img_file}: {e}")

    # 5. Ghi file manifest
    manifest_csv = args.out_dir / f"crops_manifest_{args.split}_{args.boxes}.csv"
    fieldnames = [
        "crop_id", "image_id", "cluster_id", "crop_file",
        "box_xyxy", "padded_xyxy", "source", "gt_group", "iou_with_gt", "conf"
    ]
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 68)
    print(f"[HOÀN TẤT] Đã trích xuất thành công {total_crops} crops!")
    print(f"Manifest lưu tại: {manifest_csv}")
    print(f"Phân bổ nhóm rác / background:")
    for grp, cnt in sorted(group_stats.items()):
        print(f"  - {grp:15s}: {cnt} crops")
    print("=" * 68)


if __name__ == "__main__":
    main()
