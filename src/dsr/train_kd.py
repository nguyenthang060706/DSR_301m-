"""src/dsr/train_kd.py

Script huấn luyện Knowledge Distillation đa cơ chế chuẩn Kế hoạch MASTER v9.5 (§4.2, §4.3, §5.0).
Hỗ trợ:
  1. vanilla_kd        (Hinton et al., 2015)
  2. dkd               (Decoupled KD, Zhao et al., CVPR 2022)
  3. attention_transfer(AT, Zagoruyko & Komodakis, 2017)
  4. hybrid            (Hinton KD + Attention Transfer)

Tối ưu hóa:
  - Nạp Teacher ResNet50 đã đóng băng hoàn toàn (eval mode, no grad).
  - Online forward: Teacher nhận cùng ảnh augment trong batch với Student.
  - Tích hợp PyTorch AMP FP16, channels_last, SGD momentum 0.9, CosineAnnealing.
  - Label smoothing 0.1 trên thành phần CE của Student.
  - Đánh giá trên tập Dev sau mỗi epoch: cả Nhãn mịn 30 lớp và 3 nhóm quyết định Canteen.
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
from dsr.losses import attention_transfer_loss, dkd_loss, kd_loss
from dsr.models import create_model


class FeatureHook:
    """Hook để trích xuất feature maps từ các stage của ResNet phục vụ Attention Transfer."""
    def __init__(self):
        self.features: list[torch.Tensor] = []

    def hook_fn(self, module, input, output):
        self.features.append(output)

    def clear(self):
        self.features.clear()


def register_stage_hooks(model: nn.Module) -> FeatureHook:
    hook = FeatureHook()
    # Với kiến trúc họ ResNet: layer1, layer2, layer3, layer4 là 4 stage chuẩn
    if hasattr(model, "layer1"):
        model.layer1.register_forward_hook(hook.hook_fn)
        model.layer2.register_forward_hook(hook.hook_fn)
        model.layer3.register_forward_hook(hook.hook_fn)
        model.layer4.register_forward_hook(hook.hook_fn)
    return hook


def parse_args():
    parser = argparse.ArgumentParser(description="DSR Modern Knowledge Distillation Training (V9.5)")
    parser.add_argument("--kd-method", type=str, required=True, 
                        choices=["vanilla_kd", "dkd", "attention_transfer", "hybrid"],
                        help="Phương pháp KD: vanilla_kd, dkd, attention_transfer, hybrid")
    parser.add_argument("--student-model", type=str, default="resnet18", help="Kiến trúc Student (default: resnet18)")
    parser.add_argument("--teacher-model", type=str, default="resnet50", help="Kiến trúc Teacher (default: resnet50)")
    parser.add_argument("--teacher-ckpt", type=Path, default=Path("checkpoints/teacher_resnet50_clean/best.pt"),
                        help="Đường dẫn file checkpoint Teacher")
    parser.add_argument("--run-name", type=str, required=True, help="Tên lượt chạy (ví dụ: m1_hintonk_tau3_a1)")
    
    # Siêu tham số KD
    parser.add_argument("--temperature", type=float, default=3.0, help="Nhiệt độ làm mềm softmax tau (default: 3.0)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Trọng số KD loss trong vanilla_kd / hybrid (default: 1.0)")
    parser.add_argument("--alpha-tckd", type=float, default=1.0, help="Trọng số TCKD trong DKD (default: 1.0)")
    parser.add_argument("--beta-nckd", type=float, default=1.0, help="Trọng số NCKD trong DKD (default: 1.0)")
    parser.add_argument("--beta-at", type=float, default=1000.0, help="Trọng số Attention Transfer loss (default: 1000.0)")

    # Siêu tham số tối ưu hóa
    parser.add_argument("--epochs", type=int, default=40, help="Số epoch huấn luyện (default: 40)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate ban đầu (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (default: 1e-4)")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing cho Student CE (default: 0.1)")
    parser.add_argument("--num-workers", type=int, default=4, help="Số worker nạp dữ liệu (default: 4)")
    parser.add_argument("--seed", type=int, default=20261001, help="Random seed cố định (default: 20261001)")

    # Đường dẫn dữ liệu
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/household_waste_30"))
    parser.add_argument("--map-file", type=Path, default=Path("data/mappings/label_map.csv"))
    parser.add_argument("--train-file", type=Path, default=Path("data/splits/public_train.csv"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/splits/public_dev.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def load_teacher(model_name: str, num_classes: int, ckpt_path: Path, device: torch.device) -> nn.Module:
    print(f"Nạp trọng số Teacher từ: {ckpt_path}")
    teacher = create_model(model_name, num_classes=num_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        teacher.load_state_dict(state["model_state_dict"])
    else:
        teacher.load_state_dict(state)
    teacher = teacher.to(device, memory_format=torch.channels_last)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher {model_name} đã sẵn sàng và được đóng băng hoàn toàn.")
    return teacher


def main():
    args = parse_args()

    # 1. Khóa seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"KHỞI ĐỘNG KNOWLEDGE DISTILLATION: {args.run_name.upper()}")
    print(f"Phương pháp KD: {args.kd_method} | Student: {args.student_model} | Teacher: {args.teacher_model}")
    print(f"=======================================================")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | LR: {args.lr} | Seed: {args.seed}")
    print(f"Tham số KD: Tau={args.temperature} | Alpha={args.alpha} | Alpha_TCKD={args.alpha_tckd} | Beta_NCKD={args.beta_nckd} | Beta_AT={args.beta_at}")
    print(f"Thiết bị: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 2. Tải nhãn và dữ liệu
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

    # 3. Khởi tạo Teacher (đóng băng) và Student
    teacher = load_teacher(args.teacher_model, num_fine_classes, args.teacher_ckpt, device)
    student = create_model(args.student_model, num_classes=num_fine_classes, pretrained=True)
    student = student.to(device, memory_format=torch.channels_last)

    # Hooks cho Attention Transfer nếu phương pháp yêu cầu
    need_hooks = args.kd_method in ["attention_transfer", "hybrid"]
    if need_hooks:
        student_hook = register_stage_hooks(student)
        teacher_hook = register_stage_hooks(teacher)
        print("Đã đăng ký stage hooks cho Student và Teacher phục vụ Attention Transfer.")

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler('cuda')

    # Thư mục lưu kết quả
    ckpt_dir = args.checkpoint_dir / args.run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history_csv = args.out_dir / f"{args.run_name}_history.csv"

    history = []
    best_fine_f1 = -1.0
    total_start_time = time.perf_counter()

    for ep in range(1, args.epochs + 1):
        t0 = time.perf_counter()

        # --- HUẤN LUYỆN ---
        student.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_kd_loss = 0.0
        train_samples = 0

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

                # Tính toán loss phụ thuộc phương pháp KD
                if args.kd_method == "vanilla_kd":
                    loss_kd_val = kd_loss(logits_S, logits_T, args.temperature)
                    loss = loss_ce + args.alpha * loss_kd_val
                elif args.kd_method == "dkd":
                    loss_kd_val = dkd_loss(
                        logits_S, logits_T, fine_labels,
                        alpha=args.alpha_tckd, beta=args.beta_nckd,
                        temperature=args.temperature,
                    )
                    loss = loss_ce + loss_kd_val
                elif args.kd_method == "attention_transfer":
                    loss_kd_val = attention_transfer_loss(student_hook.features, teacher_hook.features)
                    loss = loss_ce + args.beta_at * loss_kd_val
                elif args.kd_method == "hybrid":
                    loss_hinton = kd_loss(logits_S, logits_T, args.temperature)
                    loss_at = attention_transfer_loss(student_hook.features, teacher_hook.features)
                    loss_kd_val = args.alpha * loss_hinton + args.beta_at * loss_at
                    loss = loss_ce + loss_kd_val
                else:
                    raise ValueError(f"Phương pháp KD không hợp lệ: {args.kd_method}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            running_total_loss += loss.item() * bs
            running_ce_loss += loss_ce.item() * bs
            running_kd_loss += loss_kd_val.item() * bs
            train_samples += bs

        scheduler.step()
        train_loss = running_total_loss / train_samples
        ce_loss_epoch = running_ce_loss / train_samples
        kd_loss_epoch = running_kd_loss / train_samples
        train_time = time.perf_counter() - t0

        # --- ĐÁNH GIÁ TRÊN TẬP DEV ---
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
        acc_fine = accuracy_score(y_true_fine, y_pred_fine) * 100.0
        f1_coarse = f1_score(y_true_coarse, y_pred_coarse, average="macro") * 100.0
        acc_coarse = accuracy_score(y_true_coarse, y_pred_coarse) * 100.0

        is_best = f1_fine > best_fine_f1
        if is_best:
            best_fine_f1 = f1_fine
            torch.save({
                "epoch": ep,
                "model_state_dict": student.state_dict(),
                "f1_fine": f1_fine,
                "f1_coarse": f1_coarse,
                "model_name": args.student_model,
                "num_classes": num_fine_classes,
                "kd_method": args.kd_method,
            }, ckpt_dir / "best.pt")

        # Cầu chì phân kỳ (§5.1: dừng nếu sau 20 epoch mà Dev Fine F1 < 30%)
        if ep >= 20 and f1_fine < 30.0:
            print(f"\n[CẦU CHÌ KÍCH HOẠT] F1 Dev ({f1_fine:.2f}%) quá thấp sau 20 epoch. Dừng tiến trình KD.")
            break

        print(
            f"Epoch {ep:02d}/{args.epochs:02d} [{train_time:.1f}s] | "
            f"Loss: {train_loss:.4f} (CE: {ce_loss_epoch:.4f}, KD: {kd_loss_epoch:.4f}) | "
            f"Dev Loss: {val_loss:.4f} | "
            f"Fine F1: {f1_fine:.2f}% (Acc: {acc_fine:.2f}%) | "
            f"3-Class F1: {f1_coarse:.2f}% {'*BEST*' if is_best else ''}"
        )
        sys.stdout.flush()

        history.append({
            "epoch": ep,
            "train_time_sec": f"{train_time:.2f}",
            "train_loss": f"{train_loss:.4f}",
            "train_ce_loss": f"{ce_loss_epoch:.4f}",
            "train_kd_loss": f"{kd_loss_epoch:.4f}",
            "dev_loss": f"{val_loss:.4f}",
            "dev_fine_f1": f"{f1_fine:.2f}",
            "dev_fine_acc": f"{acc_fine:.2f}",
            "dev_coarse_f1": f"{f1_coarse:.2f}",
            "dev_coarse_acc": f"{acc_coarse:.2f}",
        })

    # Lưu checkpoint last
    torch.save({
        "epoch": int(history[-1]["epoch"]),
        "model_state_dict": student.state_dict(),
        "final_f1_fine": history[-1]["dev_fine_f1"],
        "model_name": args.student_model,
        "kd_method": args.kd_method,
    }, ckpt_dir / "last.pt")

    # Ghi toàn bộ lịch sử ra CSV
    fieldnames = [
        "epoch", "train_time_sec", "train_loss", "train_ce_loss", "train_kd_loss",
        "dev_loss", "dev_fine_f1", "dev_fine_acc", "dev_coarse_f1", "dev_coarse_acc",
    ]
    with history_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    total_time = (time.perf_counter() - total_start_time) / 60.0
    print(f"\n[HOÀN TẤT KD] Tổng thời gian: {total_time:.2f} phút.")
    print(f"Best Fine F1: {best_fine_f1:.2f}%")
    print(f"Checkpoint lưu tại: {ckpt_dir / 'best.pt'}")
    print(f"Lịch sử lưu tại: {history_csv}")


if __name__ == "__main__":
    main()
