"""scripts/pilot_recipe.py

Pilot Recipe nền (Tuần 1 theo Kế hoạch MASTER v9.5 §5.0).
So sánh SGD vs AdamW trên Student M0 (ResNet18) đánh giá trên tập Dev (2.250 ảnh).

Mục tiêu:
  1. Đo tốc độ thực tế (giây/epoch) khi có đầy đủ DataLoader, TrivialAugment, AMP, channels_last.
  2. So sánh hội tụ (Loss, Macro-F1 nhãn mịn 30 lớp, Macro-F1 3 nhóm) giữa SGD và AdamW.
  3. Chốt Optimizer chính thức cho protocol-lock.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from dsr.data import (
    WasteDataset,
    create_transforms,
    load_label_map,
    read_split_csv,
)
from dsr.models import create_model


def run_pilot(
    opt_name: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    data_root: Path,
    train_rows: list[dict],
    dev_rows: list[dict],
    fine_name_to_id: dict[str, int],
    fine_to_coarse_id: dict[int, int],
    seed: int = 20261001,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n=======================================================")
    print(f"BẮT ĐẦU PILOT OPTIMIZER: {opt_name.upper()} ({epochs} epochs)")
    print(f"=======================================================")

    # 1. Datasets & Loaders
    train_transform = create_transforms(image_size=224, train=True, use_trivial_augment=True)
    dev_transform = create_transforms(image_size=224, train=False)

    train_ds = WasteDataset(data_root, train_rows, fine_name_to_id, fine_to_coarse_id, train_transform)
    dev_ds = WasteDataset(data_root, dev_rows, fine_name_to_id, fine_to_coarse_id, dev_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    # 2. Model & Optimization
    model = create_model("resnet18", num_classes=30, pretrained=True)
    model = model.to(device, memory_format=torch.channels_last)

    if opt_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    elif opt_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda')

    epoch_times = []
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()

        # Training loop
        model.train()
        running_train_loss = 0.0
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
            running_train_loss += loss.item() * bs
            train_samples += bs

        scheduler.step()
        t1 = time.perf_counter()
        ep_time = t1 - t0
        epoch_times.append(ep_time)
        train_loss = running_train_loss / train_samples

        # Dev evaluation
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
        f1_coarse = f1_score(y_true_coarse, y_pred_coarse, average="macro") * 100.0

        print(
            f"Epoch {ep:02d}/{epochs:02d} | "
            f"Time: {ep_time:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {val_loss:.4f} | "
            f"Dev Fine F1: {f1_fine:.2f}% | "
            f"Dev 3-Class F1: {f1_coarse:.2f}%"
        )

        history.append({
            "epoch": ep,
            "time_sec": ep_time,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "f1_fine": f1_fine,
            "f1_coarse": f1_coarse,
        })

    avg_time = np.mean(epoch_times)
    final_f1_fine = history[-1]["f1_fine"]
    final_f1_coarse = history[-1]["f1_coarse"]

    return {
        "optimizer": opt_name,
        "avg_time_sec": avg_time,
        "final_f1_fine": final_f1_fine,
        "final_f1_coarse": final_f1_coarse,
        "best_f1_fine": max(h["f1_fine"] for h in history),
        "best_f1_coarse": max(h["f1_coarse"] for h in history),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Pilot Recipe Optimizer Comparison (SGD vs AdamW)")
    parser.add_argument("--epochs", type=int, default=5, help="Số epoch pilot (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers (default: 4)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị tính toán: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_root = Path("data/raw/household_waste_30")
    map_file = Path("data/mappings/label_map.csv")
    train_file = Path("data/splits/public_train.csv")
    dev_file = Path("data/splits/public_dev.csv")

    fine_name_to_id, fine_to_coarse_id, fine_classes, coarse_classes = load_label_map(map_file)
    train_rows = read_split_csv(train_file)
    dev_rows = read_split_csv(dev_file)

    print(f"Tập Train: {len(train_rows):,} ảnh | Tập Dev: {len(dev_rows):,} ảnh")

    # Run SGD
    res_sgd = run_pilot(
        opt_name="sgd",
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        data_root=data_root,
        train_rows=train_rows,
        dev_rows=dev_rows,
        fine_name_to_id=fine_name_to_id,
        fine_to_coarse_id=fine_to_coarse_id,
    )

    # Run AdamW
    res_adamw = run_pilot(
        opt_name="adamw",
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        data_root=data_root,
        train_rows=train_rows,
        dev_rows=dev_rows,
        fine_name_to_id=fine_name_to_id,
        fine_to_coarse_id=fine_to_coarse_id,
    )

    print("\n" + "=" * 75)
    print("BẢNG TỔNG HỢP KẾT QUẢ PILOT RECIPE (TUẦN 1 §5.0)")
    print("=" * 75)
    print(f"{'Optimizer':<10} | {'Giây/Epoch':<12} | {'Dev Fine F1 (Best)':<20} | {'Dev 3-Class F1 (Best)':<22}")
    print("-" * 75)
    for r in [res_sgd, res_adamw]:
        print(
            f"{r['optimizer'].upper():<10} | "
            f"{r['avg_time_sec']:<10.1f}s | "
            f"{r['best_f1_fine']:<18.2f}% | "
            f"{r['best_f1_coarse']:<20.2f}%"
        )
    print("=" * 75)

    # Ghi kết quả ra CSV
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "pilot_recipe_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["optimizer", "avg_time_sec", "best_fine_f1", "best_coarse_f1"])
        for r in [res_sgd, res_adamw]:
            writer.writerow([r["optimizer"], f"{r['avg_time_sec']:.2f}", f"{r['best_f1_fine']:.2f}", f"{r['best_f1_coarse']:.2f}"])

    print(f"\n[PASS] Báo cáo pilot đã lưu tại: {out_csv}")


if __name__ == "__main__":
    main()
