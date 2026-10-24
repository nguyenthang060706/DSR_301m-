"""scripts/tune_kd.py

Quy trình tinh chỉnh siêu tham số tổng quát cho các phương pháp Knowledge Distillation
Tuân thủ Kế hoạch MASTER v10.0 (§8.3 & D1-C4):
  - Phương pháp: vanilla_kd (M1), attention_transfer (M3), hybrid (M4), fitnet (M2), dkd (M1b)
  - Đọc lưới 5 cấu hình từ configs/tune_grid/<method>.json (D11)
  - 1 seed: 20261001, 20 epoch, CosineAnnealingLR(T_max=20)
  - Successive halving / Median pruning tại epoch 10
  - Chọn cấu hình thắng theo best Dev Fine Macro-F1 30 lớp
  - Ghi: reports/tune_<method>_summary.csv và configs/best_kd/<method>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from dsr.data import (
    WasteDataset,
    create_transforms,
    load_label_map,
    read_split_csv,
)
from dsr.losses import (
    FitNetRegressor,
    attention_transfer_loss,
    dkd_loss,
    fitnet_loss,
    kd_loss,
)
from dsr.models import create_model
from dsr.train_kd import FeatureHook, load_teacher, register_stage_hooks


def parse_args():
    parser = argparse.ArgumentParser(description="Tune KD Hyperparameters (v10.0 §8.3)")
    parser.add_argument("--method", type=str, required=True,
                        choices=["vanilla_kd", "dkd", "attention_transfer", "hybrid", "fitnet"],
                        help="Phương pháp KD cần tune")
    parser.add_argument("--grid-file", type=Path, default=None,
                        help="Đường dẫn file lưới cấu hình JSON (mặc định: configs/tune_grid/<method>.json)")
    parser.add_argument("--student-model", type=str, default="resnet18")
    parser.add_argument("--teacher-model", type=str, default="resnet50")
    parser.add_argument("--teacher-ckpt", type=Path, default=Path("checkpoints/teacher_resnet50_clean/best.pt"))
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/household_waste_30"))
    parser.add_argument("--map-file", type=Path, default=Path("data/mappings/label_map.csv"))
    parser.add_argument("--train-file", type=Path, default=Path("data/splits/public_train.csv"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/splits/public_dev.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--best-kd-dir", type=Path, default=Path("configs/best_kd"))
    return parser.parse_args()


def train_config_segment(
    method: str,
    cfg: dict,
    student: nn.Module,
    teacher: nn.Module,
    regressor: nn.Module | None,
    student_hook: FeatureHook | None,
    teacher_hook: FeatureHook | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    criterion_ce: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    fine_to_coarse_id: dict[int, int],
    start_ep: int,
    end_ep: int,
    device: torch.device,
) -> list[dict]:
    history = []
    need_hooks = method in ["attention_transfer", "hybrid", "fitnet"]

    for ep in range(start_ep, end_ep + 1):
        t0 = time.perf_counter()
        student.train()
        if regressor is not None:
            regressor.train()

        running_loss = 0.0
        running_ce = 0.0
        running_kd = 0.0
        total_samples = 0

        for images, fine_labels, _ in train_loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            fine_labels = fine_labels.to(device, non_blocking=True)

            if need_hooks:
                student_hook.clear()
                teacher_hook.clear()

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits_S = student(images)
                with torch.no_grad():
                    logits_T = teacher(images)

                loss_ce = criterion_ce(logits_S, fine_labels)

                # Tính loss theo phương pháp
                if method == "vanilla_kd":
                    tau = float(cfg["temperature"])
                    alpha = float(cfg["alpha"])
                    loss_kd_val = kd_loss(logits_S, logits_T, tau=tau)
                    loss = loss_ce + alpha * loss_kd_val
                elif method == "dkd":
                    temp = float(cfg.get("temperature", 3.0))
                    a = float(cfg.get("alpha_tckd", cfg.get("alpha", 1.0)))
                    b = float(cfg.get("beta_nckd", cfg.get("beta", 1.0)))
                    loss_kd_val = dkd_loss(logits_S, logits_T, fine_labels, alpha=a, beta=b, temperature=temp)
                    loss = loss_ce + loss_kd_val
                elif method == "attention_transfer":
                    b_at = float(cfg["beta_at"])
                    loss_kd_val = attention_transfer_loss(student_hook.features, teacher_hook.features)
                    loss = loss_ce + b_at * loss_kd_val
                elif method == "hybrid":
                    tau = float(cfg["temperature"])
                    alpha = float(cfg["alpha"])
                    b_at = float(cfg["beta_at"])
                    loss_hinton = kd_loss(logits_S, logits_T, tau=tau)
                    loss_at = attention_transfer_loss(student_hook.features, teacher_hook.features)
                    loss_kd_val = alpha * loss_hinton + b_at * loss_at
                    loss = loss_ce + loss_kd_val
                elif method == "fitnet":
                    lam = float(cfg["lambda_fitnet"])
                    s_feat = student_hook.features[2]
                    t_feat = teacher_hook.features[2]
                    loss_kd_val = fitnet_loss(s_feat, t_feat, regressor)
                    loss = loss_ce + lam * loss_kd_val
                else:
                    raise ValueError(f"Unknown method: {method}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_ce += loss_ce.item() * bs
            running_kd += loss_kd_val.item() * bs
            total_samples += bs

        scheduler.step()
        train_time = time.perf_counter() - t0

        # Eval loop trên Dev
        student.eval()
        if regressor is not None:
            regressor.eval()
        dev_loss = 0.0
        dev_samples = 0
        y_true_fine = []
        y_pred_fine = []
        y_true_coarse = []
        y_pred_coarse = []

        with torch.no_grad():
            for images, fine_labels, coarse_labels in dev_loader:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                fine_labels = fine_labels.to(device, non_blocking=True)

                with torch.amp.autocast('cuda'):
                    logits = student(images)
                    loss = criterion_ce(logits, fine_labels)

                bs = images.size(0)
                dev_loss += loss.item() * bs
                dev_samples += bs

                preds_fine = logits.argmax(dim=1).cpu().numpy()
                preds_coarse = np.array([fine_to_coarse_id[p] for p in preds_fine])

                y_true_fine.extend(fine_labels.cpu().numpy())
                y_pred_fine.extend(preds_fine)
                y_true_coarse.extend(coarse_labels.numpy())
                y_pred_coarse.extend(preds_coarse)

        val_loss = dev_loss / dev_samples
        f1_fine = f1_score(y_true_fine, y_pred_fine, average="macro") * 100.0
        acc_fine = accuracy_score(y_true_fine, y_pred_fine) * 100.0
        f1_coarse = f1_score(y_true_coarse, y_pred_coarse, average="macro") * 100.0
        acc_coarse = accuracy_score(y_true_coarse, y_pred_coarse) * 100.0

        print(
            f"  [{cfg['name']}] Ep {ep:02d}/{end_ep:02d} [{train_time:.1f}s] | "
            f"Loss: {running_loss/total_samples:.4f} (CE: {running_ce/total_samples:.4f}, KD: {running_kd/total_samples:.4f}) | "
            f"Dev Loss: {val_loss:.4f} | "
            f"Fine F1: {f1_fine:.2f}% (Acc: {acc_fine:.2f}%) | "
            f"3-Class F1: {f1_coarse:.2f}%"
        )
        sys.stdout.flush()

        history.append({
            "epoch": ep,
            "train_time_sec": train_time,
            "train_loss": running_loss / total_samples,
            "dev_loss": val_loss,
            "dev_fine_f1": f1_fine,
            "dev_fine_acc": acc_fine,
            "dev_coarse_f1": f1_coarse,
            "dev_coarse_acc": acc_coarse,
        })

    return history


def main():
    args = parse_args()
    grid_path = args.grid_file or Path(f"configs/tune_grid/{args.method}.json")
    if not grid_path.exists():
        raise FileNotFoundError(f"Lưới cấu hình không tồn tại: {grid_path}")

    with grid_path.open("r", encoding="utf-8") as f:
        configs = json.load(f)

    print("=" * 80)
    print(f"BẮT ĐẦU QUY TRÌNH TUNE SIÊU THAM SỐ: {args.method.upper()}")
    print(f"Số cấu hình: N = {len(configs)} | Seed: {args.seed} | Lịch: 20 epoch (prune median ep 10)")
    print(f"Grid file: {grid_path}")
    print("=" * 80)

    # Khóa seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load label map và dữ liệu
    fine_name_to_id, fine_to_coarse_id, fine_classes, _ = load_label_map(args.map_file)
    num_fine = len(fine_classes)

    train_rows = read_split_csv(args.train_file)
    dev_rows = read_split_csv(args.dev_file)

    train_tf = create_transforms(image_size=224, train=True, use_trivial_augment=True)
    dev_tf = create_transforms(image_size=224, train=False)

    train_ds = WasteDataset(args.data_root, train_rows, fine_name_to_id, fine_to_coarse_id, train_tf)
    dev_ds = WasteDataset(args.data_root, dev_rows, fine_name_to_id, fine_to_coarse_id, dev_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    teacher = load_teacher(args.teacher_model, num_fine, args.teacher_ckpt, device)

    # ----------------------------------------------------
    # GIAI ĐOẠN 1: Chạy tất cả cấu hình đến epoch 10
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("GIAI ĐOẠN 1: HUẤN LUYỆN 5 CẤU HÌNH TRONG 10 EPOCH BAN ĐẦU")
    print("=" * 80)

    active_runs = {}
    need_hooks = args.method in ["attention_transfer", "hybrid", "fitnet"]

    for cfg in configs:
        name = cfg["name"]
        print(f"\n--- Bắt đầu cấu hình: {name} ({cfg}) ---")

        # Khởi tạo Student mới cho mỗi run
        student = create_model(args.student_model, num_classes=num_fine, pretrained=True)
        student = student.to(device, memory_format=torch.channels_last)

        student_hook = None
        teacher_hook = None
        regressor = None

        if need_hooks:
            student_hook = register_stage_hooks(student)
            teacher_hook = register_stage_hooks(teacher)
            if args.method == "fitnet":
                student.eval()
                teacher.eval()
                with torch.no_grad():
                    dummy = torch.zeros(1, 3, 224, 224, device=device)
                    student(dummy)
                    teacher(dummy)
                    s_ch = student_hook.features[2].shape[1]
                    t_ch = teacher_hook.features[2].shape[1]
                student_hook.clear()
                teacher_hook.clear()
                student.train()
                regressor = FitNetRegressor(s_ch, t_ch).to(device)

        opt_params = list(student.parameters())
        if regressor is not None:
            opt_params += list(regressor.parameters())

        optimizer = torch.optim.SGD(opt_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        hist1 = train_config_segment(
            method=args.method,
            cfg=cfg,
            student=student,
            teacher=teacher,
            regressor=regressor,
            student_hook=student_hook,
            teacher_hook=teacher_hook,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion_ce=criterion_ce,
            train_loader=train_loader,
            dev_loader=dev_loader,
            fine_to_coarse_id=fine_to_coarse_id,
            start_ep=1,
            end_ep=10,
            device=device,
        )

        ep10_fine_f1 = hist1[-1]["dev_fine_f1"]
        active_runs[name] = {
            "cfg": cfg,
            "student": student,
            "regressor": regressor,
            "student_hook": student_hook,
            "teacher_hook": teacher_hook,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "criterion_ce": criterion_ce,
            "hist": hist1,
            "ep10_fine_f1": ep10_fine_f1,
        }

    # ----------------------------------------------------
    # MỐC KIỂM ĐỊNH EPOCH 10: MEDIAN PRUNING
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("MỐC KIỂM ĐỊNH EPOCH 10: TÍNH TOÁN MEDIAN PRUNING (§5.1)")
    print("=" * 80)
    scores_ep10 = {name: d["ep10_fine_f1"] for name, d in active_runs.items()}
    f1_values = list(scores_ep10.values())
    median_f1 = float(np.median(f1_values))

    print(f"Phân phối Fine F1 tại Epoch 10: {[f'{v:.2f}%' for v in sorted(f1_values)]}")
    print(f"👉 Ngưỡng Median cắt tỉa: {median_f1:.2f}%\n")

    surviving_runs = []
    pruned_runs = []

    for name, s in scores_ep10.items():
        if s >= median_f1:
            surviving_runs.append(name)
            print(f"  [GIỮ LẠI] {name:<25} | Ep10 Fine F1 = {s:.2f}% >= Median ({median_f1:.2f}%)")
        else:
            pruned_runs.append((name, s))
            print(f"  [CẮT TỈA] {name:<25} | Ep10 Fine F1 = {s:.2f}% < Median ({median_f1:.2f}%)")

    # ----------------------------------------------------
    # GIAI ĐOẠN 2: Huấn luyện tiếp từ epoch 11 đến 20
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print(f"GIAI ĐOẠN 2: TIẾP TỤC HUẤN LUYỆN {len(surviving_runs)} CẤU HÌNH VÀO EPOCH 11–20")
    print("=" * 80)

    final_results = []

    for name in surviving_runs:
        run_data = active_runs[name]
        print(f"\n--- Chạy tiếp epoch 11..20 cho: {name} ---")

        hist2 = train_config_segment(
            method=args.method,
            cfg=run_data["cfg"],
            student=run_data["student"],
            teacher=teacher,
            regressor=run_data["regressor"],
            student_hook=run_data["student_hook"],
            teacher_hook=run_data["teacher_hook"],
            optimizer=run_data["optimizer"],
            scheduler=run_data["scheduler"],
            scaler=run_data["scaler"],
            criterion_ce=run_data["criterion_ce"],
            train_loader=train_loader,
            dev_loader=dev_loader,
            fine_to_coarse_id=fine_to_coarse_id,
            start_ep=11,
            end_ep=20,
            device=device,
        )

        full_hist = run_data["hist"] + hist2
        best_fine_f1 = max(h["dev_fine_f1"] for h in full_hist)
        best_coarse_f1 = max(h["dev_coarse_f1"] for h in full_hist)
        final_ep_f1 = full_hist[-1]["dev_fine_f1"]

        final_results.append({
            "name": name,
            "cfg": run_data["cfg"],
            "best_fine_f1": best_fine_f1,
            "best_coarse_f1": best_coarse_f1,
            "final_fine_f1": final_ep_f1,
            "ep10_fine_f1": run_data["ep10_fine_f1"],
            "status": "completed",
        })

    for name, s in pruned_runs:
        final_results.append({
            "name": name,
            "cfg": next(c for c in configs if c["name"] == name),
            "best_fine_f1": s,
            "best_coarse_f1": 0.0,
            "final_fine_f1": s,
            "ep10_fine_f1": s,
            "status": "pruned_at_ep10",
        })

    # Xếp hạng kết quả
    final_results.sort(key=lambda x: x["best_fine_f1"], reverse=True)
    winner = final_results[0]

    print("\n" + "=" * 80)
    print(f"BẢNG TỔNG HỢP KẾT QUẢ TUNE SIÊU THAM SỐ: {args.method.upper()}")
    print("=" * 80)
    print(f"{'Hạng':<5} | {'Cấu hình':<25} | {'Best Fine F1':<14} | {'Status'}")
    print("-" * 80)
    for rank, res in enumerate(final_results, 1):
        print(f"{rank:<5} | {res['name']:<25} | {res['best_fine_f1']:<13.2f}% | {res['status']}")
    print("=" * 80)
    print(f"\n🏆 CẤU HÌNH CHIẾN THẮNG: {winner['name']}")
    print(f"   Tham số tối ưu: {winner['cfg']}")
    print(f"   Best Dev Fine F1 (20 ep): {winner['best_fine_f1']:.2f}%")

    # 1. Lưu summary CSV
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / f"tune_{args.method}_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "name", "best_fine_f1", "ep10_fine_f1", "status", "params_json"])
        for rank, res in enumerate(final_results, 1):
            writer.writerow([
                rank,
                res["name"],
                f"{res['best_fine_f1']:.2f}",
                f"{res['ep10_fine_f1']:.2f}",
                res["status"],
                json.dumps(res["cfg"]),
            ])
    print(f"[PASS] Báo cáo tóm tắt tune đã lưu tại: {summary_csv}")

    # 2. Lưu config thắng
    args.best_kd_dir.mkdir(parents=True, exist_ok=True)
    best_json = args.best_kd_dir / f"{args.method}.json"
    with best_json.open("w", encoding="utf-8") as f:
        json.dump(winner["cfg"], f, indent=2)
    print(f"[PASS] Cấu hình tối ưu đã lưu tại: {best_json}")


if __name__ == "__main__":
    main()
