"""scripts/calc_kappa.py

Tính toán chỉ số độ đồng thuận Cohen's Kappa giữa 2 người gán nhãn độc lập
cho tập dữ liệu Canteen đợt 1 theo Kế hoạch MASTER v9.5 (§3.3 & docs/label_guide.md).

Tiêu chuẩn chấp thuận: Kappa >= 0.70.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np


def compute_kappa(csv_path: Path, out_report: Path | None = None):
    print("=" * 75)
    print(f"KIỂM ĐỊNH ĐỘ ĐỒNG THUẬN GÁN NHÃN CANTEEN (COHEN'S KAPPA)")
    print(f"File nguồn: {csv_path}")
    print("=" * 75)

    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("File CSV trống!")

    # Lọc các dòng hợp lệ có cả annotator_1 và annotator_2
    valid_rows = [r for r in rows if r.get("annotator_1") and r.get("annotator_2")]
    print(f"Tổng số ảnh đánh giá: {len(valid_rows)} ảnh")

    # Bản đồ nhãn chuẩn 3 nhóm
    classes = ["organic", "recyclable", "other_landfill"]
    class_to_id = {c: i for i, c in enumerate(classes)}

    y1_str = [r["annotator_1"].strip().lower() for r in valid_rows]
    y2_str = [r["annotator_2"].strip().lower() for r in valid_rows]

    # Kiểm tra nhãn hợp lệ
    for name, y_list in [("Người 1", y1_str), ("Người 2", y2_str)]:
        invalid = [lbl for lbl in y_list if lbl not in class_to_id]
        if invalid:
            raise ValueError(f"{name} có nhãn không hợp lệ: {set(invalid)}. Chỉ chấp nhận: {classes}")

    y1 = [class_to_id[s] for s in y1_str]
    y2 = [class_to_id[s] for s in y2_str]

    # Tính Cohen's Kappa
    kappa = cohen_kappa_score(y1, y2)
    acc = np.mean(np.array(y1) == np.array(y2)) * 100.0
    cm = confusion_matrix(y1, y2, labels=[0, 1, 2])

    # Danh sách các ca bất đồng
    disagreements = []
    for r, s1, s2 in zip(valid_rows, y1_str, y2_str):
        if s1 != s2:
            disagreements.append({
                "image_id": r.get("image_id", "N/A"),
                "annotator_1": s1,
                "annotator_2": s2,
                "note": r.get("note", ""),
            })

    gate_pass = kappa >= 0.70
    status_str = "ĐẠT CHUẨN (PASS >= 0.70)" if gate_pass else "CHƯA ĐẠT (FAIL < 0.70 - Cần họp đối chiếu)"

    print(f"\n1. CHỈ SỐ THỐNG KÊ:")
    print(f"  - Tỷ lệ đồng thuận quan sát (Observed Agreement): {acc:.2f}%")
    print(f"  - Hệ số Cohen's Kappa: {kappa:.4f}")
    print(f"  - Đánh giá cổng nghiệm thu (§3.3): [{status_str}]")

    print(f"\n2. MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
    print(f"   (Hàng: Người 1 | Cột: Người 2)")
    print(f"   {'':<16} | {'organic':<10} | {'recyclable':<12} | {'other_landfill':<14}")
    print("   " + "-" * 60)
    for i, cname in enumerate(classes):
        print(f"   {cname:<16} | {cm[i][0]:<10} | {cm[i][1]:<12} | {cm[i][2]:<14}")

    print(f"\n3. CÁC CA BẤT ĐỒNG CẦN HỌP ĐỐI CHIẾU ({len(disagreements)} ca):")
    if not disagreements:
        print("   Tuyệt vời! Hai người đồng thuận 100%.")
    else:
        for d in disagreements[:15]:
            print(f"   - Ảnh: {d['image_id']} | Người 1: {d['annotator_1']:<14} | Người 2: {d['annotator_2']:<14}")
        if len(disagreements) > 15:
            print(f"   ... và {len(disagreements) - 15} ca khác.")

    # Xuất báo cáo Markdown nếu yêu cầu
    if out_report:
        out_report.parent.mkdir(parents=True, exist_ok=True)
        with out_report.open("w", encoding="utf-8") as f:
            f.write("# Báo cáo Kiểm định Độ đồng thuận Gán nhãn Canteen đợt 1\n\n")
            f.write(f"- **Ngày kiểm định:** {time.strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"- **Tổng số mẫu:** {len(valid_rows)} ảnh\n")
            f.write(f"- **Tỷ lệ đồng thuận thô:** {acc:.2f}%\n")
            f.write(f"- **Hệ số Cohen's Kappa:** **{kappa:.4f}**\n")
            f.write(f"- **Trạng thái cổng (§3.3):** **{status_str}**\n\n")
            f.write("## Ma trận nhầm lẫn giữa 2 người\n\n")
            f.write("| Người 1 \\ Người 2 | organic | recyclable | other_landfill |\n")
            f.write("|---|---|---|---|\n")
            for i, cname in enumerate(classes):
                f.write(f"| **{cname}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |\n")
            f.write("\n## Danh sách ca bất đồng\n\n")
            f.write("| Tên ảnh | Người 1 | Người 2 |\n")
            f.write("|---|---|---|\n")
            for d in disagreements:
                f.write(f"| `{d['image_id']}` | `{d['annotator_1']}` | `{d['annotator_2']}` |\n")
        print(f"\n[PASS] Báo cáo chi tiết đã lưu tại: {out_report}")


def main():
    parser = argparse.ArgumentParser(description="Tính Cohen's Kappa cho gán nhãn Canteen")
    parser.add_argument("--csv", type=Path, default=Path("data/canteen_metadata.csv"), help="File CSV gán nhãn")
    parser.add_argument("--out", type=Path, default=Path("reports/canteen_kappa_report.md"), help="File báo cáo xuất ra")
    args = parser.parse_args()
    compute_kappa(args.csv, args.out)


if __name__ == "__main__":
    main()
