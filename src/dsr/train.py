"""src/dsr/train.py

Script huấn luyện mô hình chuẩn (M0 Student, Teacher, hoặc Baselines)
Tuân thủ Kế hoạch MASTER v9.5 (§4.1, §5.0, §5.1):
  - Optimizer: SGD momentum 0.9, weight decay 1e-4
  - Scheduler: CosineAnnealingLR (eta_min=1e-5, không warmup)
  - Augmentation: RandomResizedCrop(224) + TrivialAugmentWide
  - AMP FP16 + channels_last memory format
  - Đánh giá trên tập Dev sau mỗi epoch: cả Nhãn mịn 30 lớp và 3 nhóm quyết định Canteen
"""

from __future__ import annotations

import argparse
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
from dsr.models import create_model


def parse_args():
    parser = argparse.ArgumentParser(description="DSR Modern Training Script (V9.5)")
    parser.add_argument("--model", type=str, required=True, help="Tên kiến trúc (resnet50, resnet18, mobilenet_v3_small, ...)")
    parser.add_argument("--run-name", type=str, required=True, help="Tên lượt chạy (ví dụ: teacher_resnet50_clean, student_m0_clean)")
    parser.add_argument("--epochs", type=int, default=40, help="Số epoch huấn luyện (default: 40)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate ban đầu (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing (0.0 cho Teacher, 0.1 cho Student M0)")
    parser.add_argument("--num-workers", type=int, default=4, help="Số worker nạp dữ liệu (default: 4)")
    parser.add_argument("--seed", type=int, default=20261001, help="Random seed cố định (default: 20261001)")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/household_waste_30"))
    parser.add_argument("--map-file", type=Path, default=Path("data/mappings/label_map.csv"))
    parser.add_argument("--train-file", type=Path, default=Path("data/splits/public_train.csv"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/splits/public_dev.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def main():
    args = parse_args()

    # Khóa seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"KHỞI ĐỘNG HUẤN LUYỆN: {args.run_name.upper()}")
    print(f"=======================================================")
    print(f"Model: {args.model} | Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print(f"LR: {args.lr} | Label Smoothing: {args.label_smoothing} | Seed: {args.seed}")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Tải ánh xạ nhãn và dữ liệu
    fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(args.map_file)
    train_rows = read_split_csv(args.train_file)
    dev_rows = read_split_csv(args.dev_file)
    num_fine_classes = len(fine_classes)

    print(f"Dữ liệu: {len(train_rows):,} Train | {len(dev_rows):,} Dev | {num_fine_classes} Fine Classes")

    train_transform = create_transforms(image_size=224, train=True, use_trivial_augment=True)
    dev_transform = create_transforms(image_size=224, train=False)

    train_ds = WasteDataset(args.data_root, train_rows, fine_name_to_id, fine_to_coarse_id, train_transform)
    dev_ds = WasteDataset(args.data_root, dev_rows, fine_name_to_id, fine_to_coarse_id, dev_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # 2. Khởi tạo mô hình
    model = create_model(args.model, num_classes=num_fine_classes, pretrained=True)
    model = model.to(device, memory_format=torch.channels_last)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda')

    # Thư mục lưu checkpoint & log
    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_csv = args.out_dir / f"{args.run_name}_history.csv"

    history = []
    best_fine_f1 = -1.0
    total_start_time = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # --- TRAIN ---
        model.train()
        running_loss = 0.0
        train_samples = 0

        for images, fine_labels, _ in train_loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            fine_labels = fine_labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss = criterion(logits, fine_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_loss += loss.item() * bs
            train_samples += bs

        scheduler.step()
        train_loss = running_loss / train_samples
        train_time = time.perf_counter() - t0

        # --- EVALUATION (DEV) ---
        model.eval()
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
                    logits = model(images)
                    loss = criterion(logits, fine_labels)

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

        is_best = f1_fine > best_fine_f1
        if is_best:
            best_fine_f1 = f1_fine
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "f1_fine": f1_fine,
                "f1_coarse": f1_coarse,
                "model_name": args.model,
                "num_classes": num_fine_classes,
            }, ckpt_dir / "best.pt")

        # Cầu chì phân kỳ (§5.1: dừng nếu loss NaN hoặc F1 Dev < 30% sau epoch 20)
        if ep >= 20 and f1_fine < 30.0:
            print(f"\n[CẦU CHÌ KÍCH HOẠT] F1 Dev ({f1_fine:.2f}%) quá thấp sau 20 epoch. Dừng tiến trình để tránh lãng phí GPU.")
            break

        print(
            f"Epoch {ep:02d}/{args.epochs:02d} [{train_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {val_loss:.4f} | "
            f"Fine F1: {f1_fine:.2f}% (Acc: {acc_fine:.2f}%) | "
            f"3-Class F1: {f1_coarse:.2f}% {'*BEST*' if is_best else ''}"
        )
        sys.stdout.flush()

        history.append({
            "epoch": ep,
            "train_time_sec": f"{train_time:.2f}",
            "train_loss": f"{train_loss:.4f}",
            "dev_loss": f"{val_loss:.4f}",
            "dev_fine_f1": f"{f1_fine:.2f}",
            "dev_fine_acc": f"{acc_fine:.2f}",
            "dev_coarse_f1": f"{f1_coarse:.2f}",
            "dev_coarse_acc": f"{acc_coarse:.2f}",
        })

    # Lưu checkpoint last
    torch.save({
        "epoch": int(history[-1]["epoch"]),
        "model_state_dict": model.state_dict(),
        "final_f1_fine": history[-1]["dev_fine_f1"],
        "model_name": args.model,
    }, ckpt_dir / "last.pt")

    # Ghi toàn bộ lịch sử ra CSV
    fieldnames = ["epoch", "train_time_sec", "train_loss", "dev_loss", "dev_fine_f1", "dev_fine_acc", "dev_coarse_f1", "dev_coarse_acc"]
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    total_time = (time.perf_counter() - total_start_time) / 60.0
    print(f"\n[HOÀN TẤT] Tổng thời gian: {total_time:.2f} phút.")
    print(f"Best Fine F1: {best_fine_f1:.2f}%")
    print(f"Checkpoint lưu tại: {ckpt_dir / 'best.pt'}")
    print(f"Lịch sử lưu tại: {history_csv}")


if __name__ == "__main__":
    main()
