"""scripts/train_d1_taco.py

Huấn luyện chính thức Detector D1(a) (YOLOv8n Class-Agnostic) trên tập dữ liệu TACO sạch.
Val leak fixed: train trên images/train, validation chọn best.pt trên images/val.
Heldout (images/heldout) được giữ làm tập đánh giá OOD độc lập.
"""

import argparse
import io
import shutil
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Detector D1(a) YOLOv8n trên TACO")
    parser.add_argument("--name", type=str, default="d1a_taco", help="Tên run lưu trong runs/detect")
    parser.add_argument("--data", type=str, default="data/processed/taco_yolo_agnostic/data.yaml", help="Đường dẫn tới file data.yaml")
    parser.add_argument("--heldout-data", type=str, default="data/processed/taco_yolo_agnostic/taco_heldout.yaml", help="Đường dẫn tới file taco_heldout.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Trọng số khởi tạo")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[LỖI] Chưa cài đặt thư viện cần thiết: {e}")
        sys.exit(1)

    print("=" * 65)
    print(f"HUẤN LUYỆN DETECTOR D1(a) ({args.epochs} EPOCH TRÊN TACO) — Run: {args.name}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Thiết bị: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[LỖI] Không tìm thấy file cấu hình dữ liệu: {data_yaml}")
        sys.exit(1)

    # Thư mục lưu checkpoint
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Bắt đầu huấn luyện {args.epochs} epoch, imgsz={args.imgsz}, batch={args.batch}...")
    start_time = time.time()

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/detect",
        name=args.name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    total_time = time.time() - start_time
    sec_per_epoch = total_time / args.epochs
    print("\n" + "=" * 65)
    print(f"[HOÀN THÀNH HUẤN LUYỆN] Tổng thời gian: {total_time / 60:.2f} phút ({sec_per_epoch:.2f} s/epoch)")
    print("=" * 65)

    # Đường dẫn best weights
    best_weights = Path(f"runs/detect/{args.name}/weights/best.pt")
    target_weights = ckpt_dir / f"{args.name}_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"[LƯU CHECKPOINT] Đã sao chép best weights tới: {target_weights}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {best_weights}, dùng checkpoint mặc định.")
        target_weights = best_weights

    # Đánh giá trên tập heldout OOD nếu có file heldout-data
    heldout_yaml = Path(args.heldout_data).resolve()
    if heldout_yaml.exists():
        print("\n[*] Đang đánh giá chính xác trên tập TACO Heldout (OOD)...")
        val_model = YOLO(str(target_weights))
        metrics = val_model.val(data=str(heldout_yaml), split="val", imgsz=args.imgsz)

        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        recall50 = float(metrics.box.r[0]) if len(metrics.box.r) > 0 else 0.0
        precision50 = float(metrics.box.p[0]) if len(metrics.box.p) > 0 else 0.0

        print("=" * 65)
        print("KẾT QUẢ TRÊN TẬP NGOẠI LAI (TACO HELDOUT):")
        print(f"  - Precision@0.5: {precision50 * 100:.2f}%")
        print(f"  - Recall@0.5:    {recall50 * 100:.2f}%")
        print(f"  - mAP50:         {map50 * 100:.2f}%")
        print(f"  - mAP50-95:      {map50_95 * 100:.2f}%")
        print("=" * 65)

        # Ghi nhận kết quả
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        report_file = reports_dir / f"{args.name}_metrics.csv"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("model,dataset,epochs,imgsz,total_minutes,sec_per_epoch,precision50,recall50,map50,map50_95\n")
            f.write(f"yolov8n,taco_agnostic,{args.epochs},{args.imgsz},{total_time/60:.2f},{sec_per_epoch:.2f},{precision50:.4f},{recall50:.4f},{map50:.4f},{map50_95:.4f}\n")
        print(f"[*] Báo cáo kết quả chi tiết đã lưu tại: {report_file}")


if __name__ == "__main__":
    main()
