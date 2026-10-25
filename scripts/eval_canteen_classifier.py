"""scripts/eval_canteen_classifier.py

D3-C4: Đánh giá mô hình phân loại (Classifier 30 lớp hoặc 31 lớp) trên tập crop khay ăn canteen.

Chỉ số toán học chuẩn hóa:
1. Macro-F1 3 nhóm: gom 30 lớp mịn về 3 nhóm lớn (0: Hữu cơ, 1: Tái chế khô, 2: Còn lại).
   Với mô hình 31 lớp (pha 2), chuẩn hóa lại softmax trên 30 lớp rác thật:
       P'(c) = exp(z_c) / sum_{j=0..29} exp(z_j)
   Báo cáo riêng tỷ lệ mẫu bị gán nhãn background.
2. Expected Calibration Error (ECE - 10 bins):
   P(G) = sum_{c in G} P'(c), confidence p_i = max_G P(G).
   ECE = sum_{m=1..10} (|B_m| / N) * |acc(B_m) - conf(B_m)|.
3. Critical Error Rate:
   Mẫu số chỉ gồm các mẫu thật là Hữu cơ hoặc Tái chế:
       Critical Error = (N(Hữu cơ -> Tái chế) + N(Tái chế -> Hữu cơ)) / (N(Hữu cơ) + N(Tái chế)).
4. Xuất file CSV dự đoán theo từng crop (kèm cluster_id) phục vụ bootstrap ghép cặp B=2000 và McNemar Ngày 7.
5. Bảo vệ Test: Chặn --split test khi DSR_ALLOW_TEST=0. Hỗ trợ --split val và --split pilot.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

sys.path.insert(0, "src")
from dsr.data import create_transforms, load_label_map
from dsr.models import create_model
from dsr.test_guard import assert_not_test, is_test_allowed


if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def compute_ece(confidences: np.ndarray, predictions: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Tính toán Expected Calibration Error (ECE) chia 10 bins đều nhau."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(confidences)

    if n_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Lọc mẫu thuộc bin (i=0 nhận cả 0.0, các bin sau nhận (lower, upper])
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(predictions[in_bin] == labels[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

    return float(ece)


def parse_args():
    parser = argparse.ArgumentParser(description="Đánh giá Canteen Classifier đa chỉ số (Spec D3-C4)")
    parser.add_argument("--model", type=str, default="resnet18", help="Kiến trúc mô hình: resnet18 hoặc mobilenet_v3_small")
    parser.add_argument("--weights", type=Path, default=None, help="File checkpoint mô hình (.pt)")
    parser.add_argument("--split", type=str, required=True, choices=["val", "pilot", "test"],
                        help="Tập dữ liệu crop cần đánh giá: val, pilot, hoặc test")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/processed/canteen_crops"), help="Thư mục gốc chứa ảnh crop")
    parser.add_argument("--manifest", type=Path, default=None, help="File manifest crops CSV")
    parser.add_argument("--label-map", type=Path, default=Path("data/mappings/label_map.csv"), help="File label_map.csv")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"), help="Thư mục lưu báo cáo")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị suy luận")
    parser.add_argument("--pretrained", action="store_true", help="Dùng ImageNet pretrained weights nếu không có checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 68)
    print(f"ĐÁNH GIÁ CANTEEN CLASSIFIER | Split: {args.split.upper()} | Model: {args.model}")
    print(f"Weights: {args.weights}")
    print("=" * 68)

    # 1. Bảo vệ Test bằng Test Guard
    if args.split == "test" and not is_test_allowed():
        raise RuntimeError(
            "[PROTOCOL VIOLATION - TEST GUARD] Tập Test canteen chưa được mở! "
            "Đánh giá trên Test chỉ được phép thực hiện duy nhất 1 lần ở Ngày 6 qua eval_test_once.py."
        )

    # 2. Xác định file manifest crops
    manifest_p = args.manifest
    if manifest_p is None:
        manifest_p = args.crops_dir / f"crops_manifest_{args.split}_gt.csv"
        if not manifest_p.exists():
            manifest_p = args.crops_dir / f"crops_manifest_{args.split}_det.csv"

    if not manifest_p.exists():
        raise FileNotFoundError(f"Không tìm thấy file manifest crops: {manifest_p}. Hãy chạy extract_crops.py trước.")

    with open(manifest_p, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"File manifest {manifest_p} trống!")

    # Guard kiểm tra đường dẫn ảnh trong manifest
    check_paths = [r["crop_file"] for r in rows if "crop_file" in r]
    if args.split != "test":
        assert_not_test(check_paths)

    # 3. Nạp ánh xạ nhãn
    fine_to_id, fine_to_coarse, fine_classes, coarse_classes = load_label_map(args.label_map, include_background=False)
    # Gom nhóm theo coarse_id: 0: organic, 1: recyclable, 2: other
    group_to_fine_ids: dict[int, list[int]] = {0: [], 1: [], 2: []}
    for f_id, c_id in fine_to_coarse.items():
        if c_id in group_to_fine_ids:
            group_to_fine_ids[c_id].append(f_id)

    # 4. Nạp checkpoint hoặc khởi tạo mô hình
    device = torch.device(args.device)
    num_classes = 30
    has_background = False

    if args.weights and args.weights.exists():
        state = torch.load(args.weights, map_location="cpu", weights_only=False)
        ckpt_classes = state.get("num_classes", 30)
        has_background = state.get("has_background", ckpt_classes == 31)
        num_classes = ckpt_classes

        model = create_model(args.model, num_classes=num_classes, pretrained=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"[*] Đã nạp checkpoint thành công (num_classes={num_classes}, has_background={has_background})")
    else:
        print(f"[*] Không có weights, khởi tạo mô hình (pretrained={args.pretrained}) để đánh giá thử (Mock/Zero-shot).")
        model = create_model(args.model, num_classes=num_classes, pretrained=args.pretrained)

    model = model.to(device)
    model.eval()

    # 5. Suy luận trên từng crop
    transform = create_transforms(image_size=224, train=False)

    valid_samples = []
    filtered_ambiguous = 0

    for r in rows:
        grp_str = r.get("gt_group", "")
        # Bỏ qua các mẫu ambiguous_loc
        if grp_str == "ambiguous_loc":
            filtered_ambiguous += 1
            continue

        crop_rel_p = r["crop_file"]
        crop_full_p = args.crops_dir / crop_rel_p
        if not crop_full_p.exists():
            continue

        try:
            true_grp = int(grp_str)
            if true_grp not in (0, 1, 2):
                continue
        except ValueError:
            continue

        valid_samples.append({
            "crop_id": r.get("crop_id", Path(crop_rel_p).stem),
            "image_id": r.get("image_id", ""),
            "cluster_id": r.get("cluster_id", "canteen_cluster"),
            "path": crop_full_p,
            "true_group": true_grp,
        })

    print(f"[*] Tổng số crop hợp lệ: {len(valid_samples)} (Đã loại {filtered_ambiguous} crop ambiguous_loc)")

    if not valid_samples:
        print("[LỖI] Không có mẫu crop nào hợp lệ để đánh giá!")
        return

    # Suy luận theo batch
    all_preds = []
    all_confs = []
    all_labels = []
    all_group_probs = []
    bg_assigned_count = 0

    batch_imgs = []
    batch_meta = []

    for idx, samp in enumerate(valid_samples):
        try:
            with Image.open(samp["path"]) as pimg:
                img_t = transform(pimg.convert("RGB"))
            batch_imgs.append(img_t)
            batch_meta.append(samp)
        except Exception:
            continue

        if len(batch_imgs) == args.batch_size or idx == len(valid_samples) - 1:
            if not batch_imgs:
                continue
            tensor_batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                logits = model(tensor_batch)

                # Nếu là model 31 lớp có background
                if num_classes == 31:
                    # Chuẩn hóa lại softmax trên 30 lớp rác thật
                    waste_logits = logits[:, :30]
                    waste_probs = F.softmax(waste_logits, dim=1).cpu().numpy()

                    # Kiểm tra tỷ lệ gán nhãn background (class 30)
                    full_probs = F.softmax(logits, dim=1).cpu().numpy()
                    bg_assigned_count += int(np.sum(np.argmax(full_probs, axis=1) == 30))
                else:
                    waste_probs = F.softmax(logits, dim=1).cpu().numpy()

            for b_idx in range(len(batch_meta)):
                p_30 = waste_probs[b_idx]
                # Tính xác suất 3 nhóm P(G) = sum_{c in G} P'(c)
                p_org = float(np.sum(p_30[group_to_fine_ids[0]])) if group_to_fine_ids[0] else 0.0
                p_rec = float(np.sum(p_30[group_to_fine_ids[1]])) if group_to_fine_ids[1] else 0.0
                p_oth = float(np.sum(p_30[group_to_fine_ids[2]])) if group_to_fine_ids[2] else 0.0

                grp_probs = np.array([p_org, p_rec, p_oth])
                pred_grp = int(np.argmax(grp_probs))
                conf = float(grp_probs[pred_grp])

                all_preds.append(pred_grp)
                all_confs.append(conf)
                all_labels.append(batch_meta[b_idx]["true_group"])
                all_group_probs.append((p_org, p_rec, p_oth))

            batch_imgs = []
            batch_meta = []

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    confs = np.array(all_confs)

    # 6. Tính toán các chỉ số toán học
    # Precision, Recall, F1 theo 3 nhóm
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    macro_f1 = float(np.mean(f1))
    acc = float(np.mean(y_true == y_pred))

    # ECE (10 bins)
    ece = compute_ece(confs, y_pred, y_true, n_bins=10)

    # Critical Error Rate: Hữu cơ (0) <-> Tái chế khô (1)
    # Mẫu số: tổng số mẫu thực tế là Hữu cơ hoặc Tái chế
    mask_org = (y_true == 0)
    mask_rec = (y_true == 1)
    denom_crit = np.sum(mask_org) + np.sum(mask_rec)
    err_org_to_rec = np.sum((y_true == 0) & (y_pred == 1))
    err_rec_to_org = np.sum((y_true == 1) & (y_pred == 0))
    crit_error_rate = (err_org_to_rec + err_rec_to_org) / max(1, denom_crit)

    # Confusion Matrix (3x3)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print("\n" + "=" * 68)
    print("KẾT QUẢ ĐÁNH GIÁ CLASSIFIER CANTEEN:")
    print("-" * 68)
    print(f"• Độ chính xác (Accuracy):         {acc * 100:.2f}%")
    print(f"• Macro-F1 (3 nhóm):              {macro_f1 * 100:.2f}%")
    print(f"   - Nhóm 0 (Hữu cơ):             P={prec[0]*100:.2f}%, R={rec[0]*100:.2f}%, F1={f1[0]*100:.2f}%")
    print(f"   - Nhóm 1 (Tái chế khô):        P={prec[1]*100:.2f}%, R={rec[1]*100:.2f}%, F1={f1[1]*100:.2f}%")
    print(f"   - Nhóm 2 (Còn lại/Rác khác):   P={prec[2]*100:.2f}%, R={rec[2]*100:.2f}%, F1={f1[2]*100:.2f}%")
    print(f"• Expected Calibration Error (ECE):{ece * 100:.2f}% (10 bins)")
    print(f"• Critical Error (Hữu cơ <-> Tái chế): {crit_error_rate * 100:.2f}% ({err_org_to_rec + err_rec_to_org}/{denom_crit} mẫu)")
    if num_classes == 31:
        print(f"• Mẫu bị gán nhãn Background:      {bg_assigned_count}/{len(valid_samples)} ({bg_assigned_count/max(1, len(valid_samples))*100:.2f}%)")
    print("\nMa trận nhầm lẫn (Confusion Matrix 3x3):")
    print("      Dự đoán: Org   Rec   Oth")
    for r_idx, r_name in enumerate(["Hữu cơ", "Tái chế", "Còn lại"]):
        print(f"  {r_name:8s}:  {cm[r_idx, 0]:4d}  {cm[r_idx, 1]:4d}  {cm[r_idx, 2]:4d}")
    print("=" * 68)

    # 7. Xuất file CSV dự đoán theo từng crop (phục vụ Bootstrap & McNemar ở Ngày 7)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = args.out_dir / f"canteen_classifier_predictions_{args.split}.csv"
    pred_rows = []
    for s_idx, samp in enumerate(valid_samples):
        p_org, p_rec, p_oth = all_group_probs[s_idx]
        t_grp = samp["true_group"]
        p_grp = all_preds[s_idx]
        is_crit = int((t_grp == 0 and p_grp == 1) or (t_grp == 1 and p_grp == 0))
        pred_rows.append({
            "crop_id": samp["crop_id"],
            "image_id": samp["image_id"],
            "cluster_id": samp["cluster_id"],
            "true_group": t_grp,
            "pred_group": p_grp,
            "confidence": f"{all_confs[s_idx]:.4f}",
            "p_organic": f"{p_org:.4f}",
            "p_recyclable": f"{p_rec:.4f}",
            "p_other": f"{p_oth:.4f}",
            "is_correct": int(t_grp == p_grp),
            "is_critical_error": is_crit,
        })

    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "crop_id", "image_id", "cluster_id", "true_group", "pred_group",
            "confidence", "p_organic", "p_recyclable", "p_other", "is_correct", "is_critical_error"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pred_rows)
    print(f"[XUẤT FILE] Đã lưu dự đoán chi tiết từng crop tại: {pred_csv}")

    # Xuất tóm tắt metrics
    summary_csv = args.out_dir / f"canteen_classifier_metrics_{args.split}.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", f"{acc:.4f}"])
        writer.writerow(["macro_f1", f"{macro_f1:.4f}"])
        writer.writerow(["f1_organic", f"{f1[0]:.4f}"])
        writer.writerow(["f1_recyclable", f"{f1[1]:.4f}"])
        writer.writerow(["f1_other", f"{f1[2]:.4f}"])
        writer.writerow(["ece_10bin", f"{ece:.4f}"])
        writer.writerow(["critical_error_rate", f"{crit_error_rate:.4f}"])
        writer.writerow(["critical_error_count", f"{err_org_to_rec + err_rec_to_org}"])
        writer.writerow(["critical_error_denom", f"{denom_crit}"])
        writer.writerow(["total_evaluated_crops", f"{len(valid_samples)}"])
    print(f"[XUẤT FILE] Đã lưu tóm tắt metrics tại: {summary_csv}")


if __name__ == "__main__":
    main()
