"""scripts/train_d1_taco.py

Huấn luyện chính thức Detector D1(a) (YOLOv8n Class-Agnostic) trên toàn bộ tập dữ liệu TACO (1.200 train, 300 heldout).
Mục tiêu:
1. Huấn luyện 50 epoch để tạo Checkpoint nền tảng (Pre-trained Waste Detector).
2. Lưu checkpoint tại: models/checkpoints/d1_taco_pretrained.pt
3. Xuất file ONNX tại: models/checkpoints/d1_taco_pretrained.onnx
4. Đánh giá Recall@0.5 và mAP50 trên 300 ảnh heldout và ghi vào reports/d1_taco_full_metrics.csv.
"""

import sys
import io
import time
import shutil
from pathlib import Path

# Đảm bảo in tiếng Việt trên console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[LỖI] Chưa cài đặt thư viện cần thiết: {e}")
        sys.exit(1)

    print("=" * 65)
    print("HUẤN LUYỆN CHÍNH THỨC DETECTOR D1(a) (50 EPOCH TRÊN TACO)")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Thiết bị: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 65)

    data_yaml = Path("data/processed/taco_yolo_agnostic/data.yaml").resolve()
    if not data_yaml.exists():
        print(f"[LỖI] Không tìm thấy file cấu hình dữ liệu: {data_yaml}")
        sys.exit(1)

    # Thư mục lưu checkpoint
    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = 50
    imgsz = 416
    batch = 16

    print(f"[*] Bắt đầu huấn luyện {epochs} epoch, imgsz={imgsz}, batch={batch}...")
    start_time = time.time()

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/detect",
        name="d1_taco_full",
        exist_ok=True,
        save=True,
        verbose=True
    )

    total_time = time.time() - start_time
    sec_per_epoch = total_time / epochs
    print("\n" + "=" * 65)
    print(f"[HOÀN THÀNH HUẤN LUYỆN] Tổng thời gian: {total_time / 60:.2f} phút ({sec_per_epoch:.2f} s/epoch)")
    print("=" * 65)

    # Đường dẫn best weights
    best_weights = Path("runs/detect/d1_taco_full/weights/best.pt")
    target_weights = ckpt_dir / "d1_taco_pretrained.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"[LƯU CHECKPOINT] Đã sao chép best weights tới: {target_weights}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy {best_weights}, dùng checkpoint mặc định.")
        target_weights = best_weights

    # Đánh giá trên tập heldout
    print("\n[*] Đang đánh giá chính xác trên tập TACO Heldout (300 ảnh)...")
    val_model = YOLO(str(target_weights))
    metrics = val_model.val(data=str(data_yaml), split="val", imgsz=imgsz)

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

    # Export ONNX
    print("\n[*] Đang xuất mô hình sang ONNX...")
    try:
        onnx_file = val_model.export(format="onnx", imgsz=imgsz)
        target_onnx = ckpt_dir / "d1_taco_pretrained.onnx"
        if Path(onnx_file).exists():
            shutil.copy2(onnx_file, target_onnx)
            print(f"[LƯU ONNX] Đã sao chép ONNX tới: {target_onnx}")
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi khi export ONNX: {e}")

    # Ghi nhận kết quả
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "d1_taco_full_metrics.csv"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("model,dataset,epochs,imgsz,total_minutes,sec_per_epoch,precision50,recall50,map50,map50_95\n")
        f.write(f"yolov8n,taco_agnostic,{epochs},{imgsz},{total_time/60:.2f},{sec_per_epoch:.2f},{precision50:.4f},{recall50:.4f},{map50:.4f},{map50_95:.4f}\n")
    print(f"[*] Báo cáo kết quả chi tiết đã lưu tại: {report_file}")

if __name__ == "__main__":
    main()
