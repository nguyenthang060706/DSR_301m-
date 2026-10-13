"""scripts/pilot_detector.py

Thử nghiệm sơ bộ (Pilot Benchmark) cho Detector D1 (YOLOv8n) class-agnostic trên tập TACO.
Mục tiêu:
1. Đo tốc độ huấn luyện thực tế (giây/epoch) trên GPU RTX 4050 để cập nhật bảng ngân sách GPU §9.1.
2. Kiểm tra quy trình nạp dữ liệu từ data/processed/taco_yolo_agnostic/data.yaml.
3. Kiểm tra tính năng export ONNX của detector.
4. Ghi nhận Recall@0.5 và mAP50 sơ bộ.
"""

import sys
import io
import time
import os
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

    print("=" * 60)
    print("PILOT BENCHMARK DETECTOR D1 (YOLOv8n Class-Agnostic)")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("=" * 60)

    data_yaml_path = Path("data/processed/taco_yolo_agnostic/data.yaml").resolve()
    if not data_yaml_path.exists():
        print(f"[LỖI] Không tìm thấy file cấu hình: {data_yaml_path}")
        sys.exit(1)

    print(f"[*] Sử dụng cấu hình dữ liệu: {data_yaml_path}")
    
    # Khởi tạo mô hình YOLOv8n pretrain COCO
    model = YOLO("yolov8n.pt")
    
    # Chạy pilot 5 epochs
    epochs = 5
    imgsz = 416
    batch = 16
    
    start_time = time.time()
    results = model.train(
        data=str(data_yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/detect",
        name="pilot_d1_taco",
        exist_ok=True,
        verbose=True
    )
    total_train_time = time.time() - start_time
    sec_per_epoch = total_train_time / epochs

    print("\n" + "=" * 60)
    print(f"[*] Hoàn thành {epochs} epoch pilot trong: {total_train_time:.2f} giây")
    print(f"[*] Tốc độ trung bình: {sec_per_epoch:.2f} giây/epoch")
    print(f"[*] Ước tính 50 epoch: {(sec_per_epoch * 50) / 60:.2f} phút (~{(sec_per_epoch * 50) / 3600:.2f} giờ)")
    print("=" * 60)

    # Đánh giá trên tập heldout
    print("\n[*] Đánh giá nhanh trên tập heldout...")
    metrics = model.val(data=str(data_yaml_path), split="val", imgsz=imgsz)
    
    # Trích xuất chỉ số
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    # Recall @ IoU 0.5
    recall50 = metrics.box.r[0] if len(metrics.box.r) > 0 else 0.0

    print(f"[*] Heldout Recall@0.5: {recall50:.4f}")
    print(f"[*] Heldout mAP50:       {map50:.4f}")
    print(f"[*] Heldout mAP50-95:    {map50_95:.4f}")

    # Thử nghiệm export ONNX
    print("\n[*] Thử nghiệm Export ONNX...")
    try:
        onnx_path = model.export(format="onnx", imgsz=imgsz)
        print(f"[THÀNH CÔNG] Export ONNX tại: {onnx_path}")
    except Exception as e:
        print(f"[CẢNH BÁO] Export ONNX gặp lỗi: {e}")

    # Ghi nhận kết quả vào reports/pilot_detector_summary.csv
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    summary_file = reports_dir / "pilot_detector_summary.csv"
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("model,epochs,imgsz,sec_per_epoch,total_sec,heldout_recall50,heldout_map50,heldout_map50_95\n")
        f.write(f"yolov8n,{epochs},{imgsz},{sec_per_epoch:.2f},{total_train_time:.2f},{recall50:.4f},{map50:.4f},{map50_95:.4f}\n")
    
    print(f"[*] Báo cáo đã lưu vào: {summary_file}")

if __name__ == "__main__":
    main()
