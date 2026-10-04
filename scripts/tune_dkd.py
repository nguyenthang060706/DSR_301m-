"""scripts/tune_dkd.py

Quy trình Tinh chỉnh Siêu tham số cho Decoupled Knowledge Distillation (M1b - DKD)
Tuân thủ Kế hoạch MASTER v9.5 (§5.1):
  - Ngân sách: N = 5 cấu hình.
  - Đánh giá trên Dev nhãn mịn 30 lớp (seed 20261001).
  - Lịch tune rút gọn (Successive Halving / Median Pruning):
      * Giai đoạn 1: Chạy 5 cấu hình trong 10 epoch.
      * Mốc kiểm định Epoch 10: Tính median Dev Fine F1, cắt tỉa (prune) các cấu hình < median.
      * Giai đoạn 2: Tiếp tục chạy các cấu hình còn lại đến epoch 20 để chọn ra Best Config.
"""

from __future__ import annotations

import csv
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
from dsr.losses import dkd_loss
from dsr.models import create_model
from dsr.train_kd import load_teacher

CONFIGS = [
    {"name": "dkd_cfg1_t3_a1_b1", "tau": 3.0, "alpha": 1.0, "beta": 1.0},
    {"name": "dkd_cfg2_t3_a0.5_b2", "tau": 3.0, "alpha": 0.5, "beta": 2.0},
    {"name": "dkd_cfg3_t3_a1_b2", "tau": 3.0, "alpha": 1.0, "beta": 2.0},
    {"name": "dkd_cfg4_t4_a1_b4", "tau": 4.0, "alpha": 1.0, "beta": 4.0},
    {"name": "dkd_cfg5_t3_a0.5_b4", "tau": 3.0, "alpha": 0.5, "beta": 4.0},
]


def train_config_segment(
    cfg: dict,
    student: nn.Module,
    teacher: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
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

    for ep in range(start_ep, end_ep + 1):
        t0 = time.perf_counter()

        # Train loop
        student.train()
        running_loss = 0.0
        running_ce = 0.0
        running_dkd = 0.0
        total_samples = 0

        for images, fine_labels, _ in train_loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            fine_labels = fine_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits_S = student(images)
                with torch.no_grad():
                    logits_T = teacher(images)

                loss_ce = criterion_ce(logits_S, fine_labels)
                loss_dkd_val = dkd_loss(
                    logits_S, logits_T, fine_labels,
                    alpha=cfg["alpha"], beta=cfg["beta"],
                    temperature=cfg["tau"],
                )
                loss = loss_ce + loss_dkd_val

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            running_ce += loss_ce.item() * bs
            running_dkd += loss_dkd_val.item() * bs
            total_samples += bs

        scheduler.step()
        train_time = time.perf_counter() - t0
        train_loss = running_loss / total_samples
        ce_loss = running_ce / total_samples
        dkd_loss_ep = running_dkd / total_samples

        # Dev evaluation
        student.eval()
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
        f1_coarse = f1_score(y_true_coarse, y_pred_coarse, average="macro") * 100.0
        acc_fine = accuracy_score(y_true_fine, y_pred_fine) * 100.0

        print(
            f"[{cfg['name']}] Ep {ep:02d}/{end_ep:02d} [{train_time:.1f}s] | "
            f"Loss: {train_loss:.4f} (CE: {ce_loss:.4f}, DKD: {dkd_loss_ep:.4f}) | "
            f"Dev Fine F1: {f1_fine:.2f}% | 3-Class F1: {f1_coarse:.2f}%"
        )
        sys.stdout.flush()

        history.append({
            "config": cfg["name"],
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dev_fine_f1": f1_fine,
            "dev_coarse_f1": f1_coarse,
            "dev_fine_acc": acc_fine,
        })

    return history


def main():
    seed = 20261001
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("BẮT ĐẦU QUY TRÌNH TUNE SIÊU THAM SỐ DKD (M1b - §5.1)")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Tổng số cấu hình thử nghiệm: N = {len(CONFIGS)}")
    print("=" * 80)

    # Dữ liệu
    map_file = Path("data/mappings/label_map.csv")
    train_file = Path("data/splits/public_train.csv")
    dev_file = Path("data/splits/public_dev.csv")
    data_root = Path("data/raw/household_waste_30")

    fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)
    train_rows = read_split_csv(train_file)
    dev_rows = read_split_csv(dev_file)
    num_classes = len(fine_classes)

    train_transform = create_transforms(image_size=224, train=True, use_trivial_augment=True)
    dev_transform = create_transforms(image_size=224, train=False)

    train_ds = WasteDataset(data_root, train_rows, fine_name_to_id, fine_to_coarse_id, train_transform)
    dev_ds = WasteDataset(data_root, dev_rows, fine_name_to_id, fine_to_coarse_id, dev_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # Nạp Teacher (đóng băng)
    teacher_ckpt = Path("checkpoints/teacher_resnet50_clean/best.pt")
    teacher = load_teacher("resnet50", num_classes=num_classes, ckpt_path=teacher_ckpt, device=device)

    # GIAI ĐOẠN 1: Chạy 10 epoch đầu cho cả 5 cấu hình
    print("\n" + "=" * 80)
    print("GIAI ĐOẠN 1: HUẤN LUYỆN 10 EPOCH CHO TẤT CẢ 5 CẤU HÌNH")
    print("=" * 80)

    active_runs = {}

    for cfg in CONFIGS:
        print(f"\n>>> Khởi động cấu hình: {cfg['name']} (Tau={cfg['tau']}, Alpha={cfg['alpha']}, Beta={cfg['beta']})")
        torch.manual_seed(seed)
        student = create_model("resnet18", num_classes=num_classes, pretrained=True)
        student = student.to(device, memory_format=torch.channels_last)

        optimizer = torch.optim.SGD(student.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        # Cosine cho 20 epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
        scaler = torch.amp.GradScaler('cuda')
        criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

        hist = train_config_segment(
            cfg=cfg,
            student=student,
            teacher=teacher,
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

        active_runs[cfg["name"]] = {
            "cfg": cfg,
            "student": student,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "criterion_ce": criterion_ce,
            "hist": hist,
            "ep10_fine_f1": hist[-1]["dev_fine_f1"],
        }

    # KIỂM ĐỊNH MỐC EPOCH 10 (MEDIAN PRUNING)
    print("\n" + "=" * 80)
    print("MỐC KIỂM ĐỊNH EPOCH 10: TÍNH TOÁN TRUNG VỊ & CẮT TỈA (MEDIAN PRUNING §5.1)")
    print("=" * 80)
    ep10_scores = [v["ep10_fine_f1"] for v in active_runs.values()]
    median_score = np.median(ep10_scores)
    print(f"Điểm Dev Fine F1 tại Epoch 10: {[f'{s:.2f}%' for s in ep10_scores]}")
    print(f"Ngưỡng trung vị (Median): {median_score:.2f}%")

    surviving_runs = {}
    pruned_runs = []

    for name, run_data in active_runs.items():
        s = run_data["ep10_fine_f1"]
        if s >= median_score:
            surviving_runs[name] = run_data
            print(f"  [GIỮ LẠI] {name:<22}: Epoch 10 F1 = {s:.2f}% (>= {median_score:.2f}%)")
        else:
            pruned_runs.append((name, s))
            print(f"  [CẮT TỈA] {name:<22}: Epoch 10 F1 = {s:.2f}% (< {median_score:.2f}%) -> DỪNG SỚM")

    # GIAI ĐOẠN 2: Chạy tiếp từ epoch 11 đến 20 cho các cấu hình vượt qua pruning
    print("\n" + "=" * 80)
    print(f"GIAI ĐOẠN 2: HUẤN LUYỆN EPOCH 11-20 CHO {len(surviving_runs)} CẤU HÌNH VƯỢT QUA VÒNG 1")
    print("=" * 80)

    final_results = []

    for name, run_data in surviving_runs.items():
        print(f"\n>>> Tiếp tục huấn luyện cấu hình: {name} (Epochs 11-20)")
        hist_part2 = train_config_segment(
            cfg=run_data["cfg"],
            student=run_data["student"],
            teacher=teacher,
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
        full_hist = run_data["hist"] + hist_part2
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
            "cfg": next(c for c in CONFIGS if c["name"] == name),
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
    print("BẢNG TỔNG HỢP KẾT QUẢ TUNE SIÊU THAM SỐ DKD (M1b - §5.1)")
    print("=" * 80)
    print(f"{'Hạng':<5} | {'Cấu hình':<22} | {'Tham số (Tau, Alpha, Beta)':<28} | {'Best Fine F1':<14} | {'Trạng thái'}")
    print("-" * 80)
    for rank, res in enumerate(final_results, 1):
        c = res["cfg"]
        params_str = f"tau={c['tau']}, a={c['alpha']}, b={c['beta']}"
        print(f"{rank:<5} | {res['name']:<22} | {params_str:<28} | {res['best_fine_f1']:<13.2f}% | {res['status']}")
    print("=" * 80)
    print(f"\n🏆 CẤU HÌNH CHIẾN THẮNG: {winner['name']}")
    print(f"   Tham số tối ưu: Tau = {winner['cfg']['tau']}, Alpha = {winner['cfg']['alpha']}, Beta = {winner['cfg']['beta']}")
    print(f"   Best Dev Fine F1 (20 ep): {winner['best_fine_f1']:.2f}%")

    # Lưu kết quả
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "tune_dkd_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "name", "tau", "alpha", "beta", "best_fine_f1", "ep10_fine_f1", "status"])
        for rank, res in enumerate(final_results, 1):
            c = res["cfg"]
            writer.writerow([rank, res["name"], c["tau"], c["alpha"], c["beta"], f"{res['best_fine_f1']:.2f}", f"{res['ep10_fine_f1']:.2f}", res["status"]])

    print(f"[PASS] Báo cáo đã lưu tại: {out_csv}")


if __name__ == "__main__":
    main()
