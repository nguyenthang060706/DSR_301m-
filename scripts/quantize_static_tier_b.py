"""scripts/quantize_static_tier_b.py

Static Post-Training Quantization (Static PTQ INT8) theo chuẩn Kế hoạch MASTER v9.5 (§8).
Sử dụng 200 ảnh phân tầng từ tập Dev làm Calibration Dataset để tính toán chính xác
phạm vi kích hoạt (Activation Scales) cho các lớp Depthwise Separable Convolution của MobileNetV3.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "src")
from dsr.data import (
    WasteDataset,
    create_transforms,
    load_label_map,
    read_split_csv,
)


class DevCalibrationDataReader(CalibrationDataReader):
    """Nạp đúng 200 ảnh phân tầng từ tập Dev để hiệu chuẩn QDQ."""

    def __init__(self, dataloader: DataLoader, max_samples: int = 200):
        self.enum_data = iter(dataloader)
        self.count = 0
        self.max_samples = max_samples

    def get_next(self):
        if self.count >= self.max_samples:
            return None
        try:
            batch = next(self.enum_data)
            images, _, _ = batch
            self.count += images.shape[0]
            return {"input": images.numpy()}
        except StopIteration:
            return None

    def rewind(self):
        self.enum_data = iter(self.enum_data)
        self.count = 0


def evaluate_onnx(onnx_path: Path, dev_loader: DataLoader, fine_to_coarse_id: dict[int, int]):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    y_true_fine = []
    y_pred_fine = []
    y_true_coarse = []
    y_pred_coarse = []

    for images, fine_labels, coarse_labels in dev_loader:
        img_np = images.numpy()
        outputs = session.run(None, {input_name: img_np})
        logits = outputs[0]
        preds_fine = logits.argmax(axis=1)
        preds_coarse = np.array([fine_to_coarse_id[p] for p in preds_fine])

        y_true_fine.extend(fine_labels.numpy())
        y_pred_fine.extend(preds_fine)
        y_true_coarse.extend(coarse_labels.numpy())
        y_pred_coarse.extend(preds_coarse)

    f1_fine = f1_score(y_true_fine, y_pred_fine, average="macro") * 100.0
    acc_fine = accuracy_score(y_true_fine, y_pred_fine) * 100.0
    f1_coarse = f1_score(y_true_coarse, y_pred_coarse, average="macro") * 100.0
    acc_coarse = accuracy_score(y_true_coarse, y_pred_coarse) * 100.0

    return {
        "fine_f1": f1_fine,
        "fine_acc": acc_fine,
        "coarse_f1": f1_coarse,
        "coarse_acc": acc_coarse,
    }


def benchmark_latency(onnx_path: Path, warmup_runs: int = 50, test_runs: int = 200):
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})

    latencies_ms = []
    for _ in range(test_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / median_lat if median_lat > 0 else 0.0

    return {
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "fps": fps,
    }


def main():
    print("=" * 80)
    print("STATIC INT8 QUANTIZATION (QDQ) VỚI 200 ẢNH CALIBRATION (MASTER V9.5 §8)")
    print("=" * 80)

    map_file = Path("data/mappings/label_map.csv")
    dev_file = Path("data/splits/public_dev.csv")
    data_root = Path("data/raw/household_waste_30")
    onnx_dir = Path("deploy/onnx")

    fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)
    dev_rows = read_split_csv(dev_file)

    dev_transform = create_transforms(image_size=224, train=False)
    dev_ds = WasteDataset(data_root, dev_rows, fine_name_to_id, fine_to_coarse_id, dev_transform)

    # Loader cho đánh giá (batch 32)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=2)
    # Loader cho Calibration (batch 1, 200 ảnh)
    calib_loader = DataLoader(dev_ds, batch_size=1, shuffle=False)

    models = [
        "tier_b_m0_mobilenet_v3_small",
        "tier_b_m1b_dkd_mobilenet_v3_small",
    ]

    results = []

    for name in models:
        fp32_path = onnx_dir / f"{name}_fp32.onnx"
        static_int8_path = onnx_dir / f"{name}_static_int8.onnx"

        print(f"\n" + "-" * 70)
        print(f"LƯỢNG TỬ HÓA STATIC INT8: {name.upper()}")
        print("-" * 70)

        # 1. Khởi tạo Calibration Reader với đúng 200 ảnh
        calib_reader = DevCalibrationDataReader(calib_loader, max_samples=200)

        # 2. Static Quantization
        print(f"  Đang chạy Static Quantization QDQ với 200 ảnh hiệu chuẩn...")
        t0 = time.perf_counter()
        quantize_static(
            model_input=str(fp32_path),
            model_output=str(static_int8_path),
            calibration_data_reader=calib_reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
            calibrate_method=CalibrationMethod.MinMax,
        )
        quant_time = time.perf_counter() - t0
        size_int8 = static_int8_path.stat().st_size / (1024 * 1024)
        size_fp32 = fp32_path.stat().st_size / (1024 * 1024)
        print(f"  [PASS] Hoàn thành trong {quant_time:.1f}s | Dung lượng: {size_int8:.2f} MB (gốc: {size_fp32:.2f} MB, nén {size_fp32/size_int8:.2f}x)")

        # 3. Đánh giá độ chính xác trên tập Dev
        print(f"  Đánh giá độ chính xác Static INT8 trên 2.250 ảnh Dev...")
        metrics_int8 = evaluate_onnx(static_int8_path, dev_loader, fine_to_coarse_id)

        # Đánh giá FP32 đối chiếu
        metrics_fp32 = evaluate_onnx(fp32_path, dev_loader, fine_to_coarse_id)

        drop = metrics_fp32["fine_f1"] - metrics_int8["fine_f1"]
        gate_status = "PASS (<= 2.0%)" if drop <= 2.0 else f"FAIL ({drop:.2f}%)"

        print(f"    FP32: Dev Fine F1 = {metrics_fp32['fine_f1']:.2f}% | 3-Class F1 = {metrics_fp32['coarse_f1']:.2f}%")
        print(f"    Static INT8: Dev Fine F1 = {metrics_int8['fine_f1']:.2f}% | 3-Class F1 = {metrics_int8['coarse_f1']:.2f}%")
        print(f"    Mức sụt giảm (Delta Fine F1): {drop:.2f}% [{gate_status}]")

        # 4. Đo Latency
        print(f"  Đo Latency CPU suy luận (Warmup 50, Test 200 lần)...")
        lat = benchmark_latency(static_int8_path)
        print(f"    Static INT8 Latency: {lat['median_ms']:.2f} ms | P95: {lat['p95_ms']:.2f} ms | {lat['fps']:.1f} FPS")

        results.append({
            "model_name": name,
            "format": "Static_INT8",
            "size_mb": f"{size_int8:.2f}",
            "fp32_fine_f1": f"{metrics_fp32['fine_f1']:.2f}",
            "int8_fine_f1": f"{metrics_int8['fine_f1']:.2f}",
            "drop_f1": f"{drop:.2f}",
            "canteen_f1": f"{metrics_int8['coarse_f1']:.2f}",
            "median_lat_ms": f"{lat['median_ms']:.2f}",
            "fps": f"{lat['fps']:.1f}",
            "gate_status": gate_status,
        })

    print("\n" + "=" * 90)
    print("BẢNG TỔNG HỢP NGHIỆM THU STATIC INT8 (QDQ FORMAT - §8)")
    print("=" * 90)
    print(f"{'Mô hình':<32} | {'Dung lượng':<10} | {'FP32 F1':<8} | {'INT8 F1':<8} | {'Tụt F1':<7} | {'Canteen F1':<10} | {'Latency':<8} | {'Cổng INT8'}")
    print("-" * 90)
    for r in results:
        print(f"{r['model_name']:<32} | {r['size_mb']:<6} MB | {r['fp32_fine_f1']:<8} | {r['int8_fine_f1']:<8} | {r['drop_f1']:<7} | {r['canteen_f1']:<10} | {r['median_lat_ms']:<5} ms | {r['gate_status']}")
    print("=" * 90)

    # Lưu báo cáo
    out_csv = Path("reports/static_int8_benchmark.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[PASS] Báo cáo đã lưu tại: {out_csv}")


if __name__ == "__main__":
    main()
