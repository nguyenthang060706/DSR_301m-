"""scripts/train_d1b_combined.py

Huấn luyện Detector D1(b) (YOLOv8n Class-Agnostic) trên tập dữ liệu Đa Nguồn Hỗn Hợp:
- TACO (1.200 ảnh)
- Synthetic Copy-Paste Trays (1.275 ảnh)
- Padova Canteen Trays (50 ảnh)
Mục tiêu: Đột phá trần Recall@0.5 từ 45.56% lên trên 70-80% để phục vụ Pipeline Canteen.
"""

import sys
import io
import time
import shutil
import argparse
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Detector D1(b) Hỗn hợp")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch huấn luyện (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=416, help="Kích thước ảnh đầu vào (default: 416)")
    parser.add_argument("--weights", type=str, default="models/checkpoints/d1_taco_pretrained.pt", help="Checkpoint khởi tạo")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[LỖI] Thiếu thư viện: {e}")
        sys.exit(1)

    print("=" * 70)
    print("HUẤN LUYỆN DETECTOR D1(b) ĐA NGUỒN (TACO + SYNTHETIC + PADOVA)")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print(f"Khởi tạo từ: {args.weights}")
    print("=" * 70)

    data_yaml = Path("data/processed/combined_d1b_data.yaml").resolve()
    if not data_yaml.exists():
        print(f"[LỖI] Không tìm thấy file cấu hình: {data_yaml}")
        sys.exit(1)

    ckpt_dir = Path("models/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    weights_init = args.weights if Path(args.weights).exists() else "yolov8n.pt"
    print(f"[*] Sử dụng trọng số ban đầu: {weights_init}")

    start_time = time.time()
    model = YOLO(weights_init)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/detect",
        name="d1b_combined",
        exist_ok=True,
        save=True,
        verbose=True
    )

    total_time = time.time() - start_time
    sec_per_epoch = total_time / args.epochs
    print("\n" + "=" * 70)
    print(f"[HOÀN THÀNH HUẤN LUYỆN] Tổng thời gian: {total_time / 60:.2f} phút ({sec_per_epoch:.2f} s/epoch)")
    print("=" * 70)

    if hasattr(model, "trainer") and getattr(model.trainer, "best", None):
        best_weights = Path(model.trainer.best)
    elif Path("runs/detect/runs/detect/d1b_combined/weights/best.pt").exists():
        best_weights = Path("runs/detect/runs/detect/d1b_combined/weights/best.pt")
    else:
        best_weights = Path("runs/detect/d1b_combined/weights/best.pt")

    target_weights = ckpt_dir / "d1b_combined_best.pt"

    if best_weights.exists():
        shutil.copy2(best_weights, target_weights)
        print(f"[LƯU CHECKPOINT] Đã sao chép best weights tới: {target_weights}")
    else:
        print(f"[CẢNH BÁO] Không tìm thấy file {best_weights}")
        target_weights = best_weights

    # Đánh giá trên tập validation đa nguồn
    print("\n[*] Đang đánh giá trên tập Validation đa nguồn...")
    val_model = YOLO(str(target_weights))
    metrics = val_model.val(data=str(data_yaml), split="val", imgsz=args.imgsz)

    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    recall50 = float(metrics.box.r[0]) if len(metrics.box.r) > 0 else 0.0
    precision50 = float(metrics.box.p[0]) if len(metrics.box.p) > 0 else 0.0

    print("=" * 70)
    print("KẾT QUẢ ĐÁNH GIÁ DETECTOR D1(b):")
    print(f"  - Precision@0.5: {precision50 * 100:.2f}%")
    print(f"  - Recall@0.5:    {recall50 * 100:.2f}%  (Mục tiêu Go/No-go: >= 80.0%)")
    print(f"  - mAP50:         {map50 * 100:.2f}%")
    print(f"  - mAP50-95:      {map50_95 * 100:.2f}%")
    print("=" * 70)

    # Xuất ONNX
    print("\n[*] Đang xuất sang ONNX...")
    try:
        onnx_file = val_model.export(format="onnx", imgsz=args.imgsz)
        target_onnx = ckpt_dir / "d1b_combined_best.onnx"
        if Path(onnx_file).exists():
            shutil.copy2(onnx_file, target_onnx)
            print(f"[LƯU ONNX] Đã sao chép ONNX tới: {target_onnx}")
    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi export ONNX: {e}")

    # Ghi nhận báo cáo
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    report_file = reports_dir / "d1b_combined_metrics.csv"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("model,dataset,epochs,imgsz,total_minutes,sec_per_epoch,precision50,recall50,map50,map50_95\n")
        f.write(f"yolov8n,taco_synth_padova,{args.epochs},{args.imgsz},{total_time/60:.2f},{sec_per_epoch:.2f},{precision50:.4f},{recall50:.4f},{map50:.4f},{map50_95:.4f}\n")
    print(f"[*] Báo cáo kết quả chi tiết đã lưu tại: {report_file}")

if __name__ == "__main__":
    main()
