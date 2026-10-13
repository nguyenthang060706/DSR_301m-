"""scripts/export_and_quantize_tier_b.py

Quy trình Xuất mô hình & Lượng tử hóa INT8 (PTQ) cho Tier B (MobileNetV3-Small)
Tuân thủ Kế hoạch MASTER v9.5 (§4.1 & §8):
  1. Xuất mô hình PyTorch sang ONNX FP32.
  2. Lượng tử hóa Post-Training Quantization (INT8) bằng ONNX Runtime.
  3. Đo lường chính xác mức sụt giảm Macro-F1 trên tập Dev (Cổng yêu cầu tụt <= 2.0%).
  4. Đo kiểm Latency chuẩn hóa (Warm-up 50 lần, đo 200 lần: Median, P95, FPS).
  5. Đánh giá dung lượng file (MB) và tỷ lệ nén.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_dynamic, quantize_static
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
from dsr.models import create_model


class DevCalibrationDataReader(CalibrationDataReader):
    """Nạp 200 ảnh phân tầng từ tập Dev để hiệu chuẩn static quantization."""

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
            # ONNX Runtime yêu cầu dict {input_name: numpy_array}
            return {"input": images.numpy()}
        except StopIteration:
            return None


def export_to_onnx(model: torch.nn.Module, onnx_path: Path):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"  [PASS] Đã xuất ONNX FP32: {onnx_path} ({size_mb:.2f} MB)")
    return size_mb


def evaluate_onnx_model(onnx_path: Path, dev_loader: DataLoader, fine_to_coarse_id: dict[int, int]):
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

    # 1. Warm-up
    for _ in range(warmup_runs):
        _ = session.run(None, {input_name: dummy_input})

    # 2. Đo kiểm lặp lại
    latencies_ms = []
    for _ in range(test_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_lat = float(np.median(latencies_ms))
    p95_lat = float(np.percentile(latencies_ms, 95))
    mean_lat = float(np.mean(latencies_ms))
    std_lat = float(np.std(latencies_ms))
    fps = 1000.0 / median_lat if median_lat > 0 else 0.0

    return {
        "median_ms": median_lat,
        "p95_ms": p95_lat,
        "mean_ms": mean_lat,
        "std_ms": std_lat,
        "fps": fps,
    }


def main():
    print("=" * 80)
    print("QUY TRÌNH LƯỢNG TỬ HÓA INT8 & ĐO KIỂM THIẾT BỊ BIÊN (MASTER V9.5 §8)")
    print("=" * 80)

    map_file = Path("data/mappings/label_map.csv")
    dev_file = Path("data/splits/public_dev.csv")
    data_root = Path("data/raw/household_waste_30")
    out_dir = Path("deploy/onnx")
    out_dir.mkdir(parents=True, exist_ok=True)

    fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)
    dev_rows = read_split_csv(dev_file)
    num_classes = len(fine_classes)

    dev_transform = create_transforms(image_size=224, train=False)
    dev_ds = WasteDataset(data_root, dev_rows, fine_name_to_id, fine_to_coarse_id, dev_transform)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=2)

    # Danh sách các mô hình biên cần kiểm tra
    models_to_benchmark = [
        {
            "name": "tier_b_m0_mobilenet_v3_small",
            "arch": "mobilenet_v3_small",
            "ckpt": Path("checkpoints/tier_b_m0_mobilenet_v3_small/best.pt"),
            "desc": "Tier B Baseline (CE thuần)",
        },
        {
            "name": "tier_b_m1b_dkd_mobilenet_v3_small",
            "arch": "mobilenet_v3_small",
            "ckpt": Path("checkpoints/tier_b_m1b_dkd_mobilenet_v3_small/best.pt"),
            "desc": "Tier B Distillation (DKD)",
        },
    ]

    all_records = []

    for item in models_to_benchmark:
        print(f"\n" + "-" * 75)
        print(f"XỬ LÝ MÔ HÌNH: {item['name'].upper()} ({item['desc']})")
        print("-" * 75)

        # 1. Nạp PyTorch Model
        model = create_model(item["arch"], num_classes=num_classes, pretrained=False)
        state = torch.load(item["ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        # 2. Xuất ONNX FP32
        fp32_path = out_dir / f"{item['name']}_fp32.onnx"
        size_fp32_mb = export_to_onnx(model, fp32_path)

        # 3. Lượng tử hóa Dynamic INT8
        int8_path = out_dir / f"{item['name']}_int8.onnx"
        print(f"  Thực hiện Dynamic Post-Training Quantization (INT8)...")
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            weight_type=QuantType.QInt8,
        )
        size_int8_mb = int8_path.stat().st_size / (1024 * 1024)
        compression_ratio = size_fp32_mb / size_int8_mb
        print(f"  [PASS] Đã tạo INT8 ONNX: {int8_path} ({size_int8_mb:.2f} MB, nén {compression_ratio:.2f}x)")

        # 4. Đánh giá độ chính xác FP32 vs INT8 trên Dev set
        print(f"  Đánh giá độ chính xác trên 2.250 ảnh Dev...")
        metrics_fp32 = evaluate_onnx_model(fp32_path, dev_loader, fine_to_coarse_id)
        metrics_int8 = evaluate_onnx_model(int8_path, dev_loader, fine_to_coarse_id)

        drop_fine_f1 = metrics_fp32["fine_f1"] - metrics_int8["fine_f1"]
        drop_coarse_f1 = metrics_fp32["coarse_f1"] - metrics_int8["coarse_f1"]

        gate_pass = drop_fine_f1 <= 2.0
        gate_status = "PASS (<= 2.0%)" if gate_pass else "FAIL (> 2.0%)"

        print(f"    FP32: Dev Fine F1 = {metrics_fp32['fine_f1']:.2f}% | 3-Class F1 = {metrics_fp32['coarse_f1']:.2f}%")
        print(f"    INT8: Dev Fine F1 = {metrics_int8['fine_f1']:.2f}% | 3-Class F1 = {metrics_int8['coarse_f1']:.2f}%")
        print(f"    Mức sụt giảm sau INT8: Delta Fine F1 = {drop_fine_f1:.2f}% [{gate_status}]")

        # 5. Đo kiểm Latency CPU Edge Simulation
        print(f"  Đo độ trễ suy luận (Warmup 50, Test 200 lần)...")
        lat_fp32 = benchmark_latency(fp32_path)
        lat_int8 = benchmark_latency(int8_path)
        print(f"    FP32 Latency: Median = {lat_fp32['median_ms']:.2f} ms | P95 = {lat_fp32['p95_ms']:.2f} ms | {lat_fp32['fps']:.1f} FPS")
        print(f"    INT8 Latency: Median = {lat_int8['median_ms']:.2f} ms | P95 = {lat_int8['p95_ms']:.2f} ms | {lat_int8['fps']:.1f} FPS")

        all_records.append({
            "model_name": item["name"],
            "format": "FP32",
            "size_mb": f"{size_fp32_mb:.2f}",
            "fine_f1": f"{metrics_fp32['fine_f1']:.2f}",
            "coarse_f1": f"{metrics_fp32['coarse_f1']:.2f}",
            "median_lat_ms": f"{lat_fp32['median_ms']:.2f}",
            "p95_lat_ms": f"{lat_fp32['p95_ms']:.2f}",
            "fps": f"{lat_fp32['fps']:.1f}",
            "gate_drop": "0.00",
            "status": "N/A",
        })

        all_records.append({
            "model_name": item["name"],
            "format": "INT8",
            "size_mb": f"{size_int8_mb:.2f}",
            "fine_f1": f"{metrics_int8['fine_f1']:.2f}",
            "coarse_f1": f"{metrics_int8['coarse_f1']:.2f}",
            "median_lat_ms": f"{lat_int8['median_ms']:.2f}",
            "p95_lat_ms": f"{lat_int8['p95_ms']:.2f}",
            "fps": f"{lat_int8['fps']:.1f}",
            "gate_drop": f"{drop_fine_f1:.2f}",
            "status": gate_status,
        })

    # In Bảng Tổng Kết Chính Thức
    print("\n" + "=" * 95)
    print("BẢNG TỔNG HỢP KẾT QUẢ LƯỢNG TỬ HÓA & BENCHMARK THIẾT BỊ BIÊN (TIER B)")
    print("=" * 95)
    print(f"{'Mô hình':<28} | {'Định dạng':<8} | {'Dung lượng':<10} | {'Fine F1':<8} | {'3-Class F1':<10} | {'Latency':<9} | {'FPS':<6} | {'Cổng INT8'}")
    print("-" * 95)
    for r in all_records:
        print(f"{r['model_name']:<28} | {r['format']:<8} | {r['size_mb']:<6} MB | {r['fine_f1']:<8} | {r['coarse_f1']:<10} | {r['median_lat_ms']:<5} ms | {r['fps']:<6} | {r['status']}")
    print("=" * 95)

    # Ghi báo cáo ra CSV
    report_csv = Path("reports/edge_quantization_benchmark.csv")
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_records)
    print(f"\n[PASS] Báo cáo chi tiết đã lưu tại: {report_csv}")


if __name__ == "__main__":
    main()
