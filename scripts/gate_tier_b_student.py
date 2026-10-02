"""scripts/gate_tier_b_student.py

Cổng kiểm tra chọn Student Tier B (Tuần 1 theo Kế hoạch MASTER v9.5 §4.1).
Đánh giá 3 ứng viên:
  1. MobileNetV3-Small (torchvision / timm)
  2. MobileNetV3-Large (torchvision / timm)
  3. MobileNetV4-Conv-Small (timm)

Kiểm tra 2 tiêu chí tuần 1:
  - Cổng 1: Xuất ONNX và nạp chạy trên ONNX Runtime không lỗi toán tử.
  - Cổng 3: Đo độ trễ suy luận mô phỏng trên CPU (batch=1, 100 lần lặp).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
import torchvision.models as models


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_candidate(
    name: str,
    create_fn,
    input_size: tuple[int, int] = (224, 224),
    num_classes: int = 12,
    num_runs: int = 100,
    warmup: int = 20,
) -> dict:
    print(f"\n==================================================")
    print(f"Kiểm tra ứng viên: {name}")
    print(f"==================================================")

    # 1. Khởi tạo mô hình
    try:
        model = create_fn(num_classes=num_classes)
        model.eval()
        params = count_parameters(model)
        print(f"[OK] Khởi tạo PyTorch model thành công. Số tham số: {params:,}")
    except Exception as e:
        print(f"[FAIL] Khởi tạo mô hình thất bại: {e}")
        return {"name": name, "params": 0, "onnx_export": False, "ort_run": False}

    dummy_input = torch.randn(1, 3, *input_size)

    # 2. Cổng 1: Xuất sang ONNX
    onnx_file = Path(tempfile.gettempdir()) / f"{name}_temp.onnx"
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_file),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=None,  # Cố định batch=1 cho thiết bị biên
            opset_version=17,
        )
        # Kiểm tra tính toàn vẹn của file ONNX
        onnx_model = onnx.load(str(onnx_file))
        onnx.checker.check_model(onnx_model)
        file_size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"[PASS] Cổng 1a: Xuất ONNX thành công. Kích thước file: {file_size_mb:.2f} MB")
        onnx_export_ok = True
    except Exception as e:
        print(f"[FAIL] Cổng 1a: Lỗi xuất ONNX: {e}")
        return {"name": name, "params": params, "onnx_export": False, "ort_run": False}

    # 3. Cổng 1b: Nạp vào ONNX Runtime (CPU)
    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1  # Mô phỏng 1 nhân CPU biên
        sess = ort.InferenceSession(str(onnx_file), sess_options, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy_np = dummy_input.numpy()
        _ = sess.run(None, {input_name: dummy_np})
        print(f"[PASS] Cổng 1b: Nạp và chạy ONNX Runtime thành công (không có toán tử không tương thích).")
        ort_run_ok = True
    except Exception as e:
        print(f"[FAIL] Cổng 1b: ONNX Runtime thất bại: {e}")
        return {"name": name, "params": params, "onnx_export": onnx_export_ok, "ort_run": False}

    # 4. Cổng 3: Đo độ trễ suy luận mô phỏng (CPU, batch=1)
    print(f"Đang đo độ trễ ({warmup} warm-up, {num_runs} lần lặp)...")
    for _ in range(warmup):
        _ = sess.run(None, {input_name: dummy_np})

    latencies_ms = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = sess.run(None, {input_name: dummy_np})
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    median_lat = np.median(latencies_ms)
    p95_lat = np.percentile(latencies_ms, 95)
    std_lat = np.std(latencies_ms)

    print(f"Độ trễ CPU (batch=1):")
    print(f"  - Trung vị (Median): {median_lat:.2f} ms")
    print(f"  - P95: {p95_lat:.2f} ms")
    print(f"  - Độ lệch chuẩn: {std_lat:.2f} ms")

    # Dọn file tạm
    if onnx_file.exists():
        onnx_file.unlink()

    return {
        "name": name,
        "params": params,
        "file_size_mb": file_size_mb,
        "onnx_export": onnx_export_ok,
        "ort_run": ort_run_ok,
        "median_ms": median_lat,
        "p95_ms": p95_lat,
    }


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra Cổng chọn Student Tier B")
    parser.add_argument("--num-classes", type=int, default=12, help="Số lớp nhãn mịn (default: 12)")
    parser.add_argument("--runs", type=int, default=100, help="Số lần đo suy luận (default: 100)")
    args = parser.parse_args()

    candidates = [
        (
            "MobileNetV3-Small",
            lambda num_classes: models.mobilenet_v3_small(weights=None, num_classes=num_classes),
        ),
        (
            "MobileNetV3-Large",
            lambda num_classes: models.mobilenet_v3_large(weights=None, num_classes=num_classes),
        ),
        (
            "MobileNetV4-Conv-Small",
            lambda num_classes: timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=num_classes),
        ),
    ]

    results = []
    for name, create_fn in candidates:
        res = test_candidate(name, create_fn, num_classes=args.num_classes, num_runs=args.runs)
        results.append(res)

    print("\n" + "=" * 70)
    print("BẢNG TỔNG HỢP CỔNG CHỌN STUDENT TIER B (TUẦN 1)")
    print("=" * 70)
    print(f"{'Ứng viên':<24} | {'Params':<10} | {'Size(MB)':<9} | {'ONNX':<6} | {'ORT':<5} | {'Latency (ms)':<12}")
    print("-" * 70)
    for r in results:
        onnx_s = "PASS" if r.get("onnx_export") else "FAIL"
        ort_s = "PASS" if r.get("ort_run") else "FAIL"
        lat_s = f"{r.get('median_ms', 0):.2f} ms" if "median_ms" in r else "N/A"
        size_s = f"{r.get('file_size_mb', 0):.2f}" if "file_size_mb" in r else "N/A"
        params_s = f"{r.get('params', 0):,}"
        print(f"{r['name']:<24} | {params_s:<10} | {size_s:<9} | {onnx_s:<6} | {ort_s:<5} | {lat_s:<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()
