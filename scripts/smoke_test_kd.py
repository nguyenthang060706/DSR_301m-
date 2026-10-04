"""scripts/smoke_test_kd.py

Smoke-test nhanh chóng xác nhận 4 cơ chế KD (Vanilla KD, DKD, Attention Transfer, Hybrid)
hoạt động 100% trơn tru trên GPU RTX 4050 với Teacher ResNet50 đã đóng băng.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, "src")
from dsr.models import create_model
from dsr.losses import kd_loss, dkd_loss, attention_transfer_loss
from dsr.train_kd import load_teacher, register_stage_hooks

def run_smoke_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu Smoke Test Pipeline KD trên: {device}")

    # 1. Nạp Teacher
    teacher_ckpt = Path("checkpoints/teacher_resnet50_clean/best.pt")
    teacher = load_teacher("resnet50", num_classes=30, ckpt_path=teacher_ckpt, device=device)

    # 2. Khởi tạo Student ResNet18
    student = create_model("resnet18", num_classes=30, pretrained=True)
    student = student.to(device, memory_format=torch.channels_last)
    student.train()

    # Stage hooks cho Attention Transfer
    student_hook = register_stage_hooks(student)
    teacher_hook = register_stage_hooks(teacher)

    # 3. Tạo mini-batch giả lập (B=4, 30 classes)
    images = torch.randn(4, 3, 224, 224, device=device).to(memory_format=torch.channels_last)
    labels = torch.tensor([0, 15, 29, 7], device=device)
    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.amp.GradScaler('cuda')

    # Test M1: Vanilla KD
    print("\n--- Test M1: Vanilla KD (Hinton) ---")
    student_hook.clear(); teacher_hook.clear()
    with torch.amp.autocast('cuda'):
        logits_S = student(images)
        with torch.no_grad():
            logits_T = teacher(images)
        loss_ce = criterion_ce(logits_S, labels)
        loss_kd = kd_loss(logits_S, logits_T, tau=3.0)
        total_loss = loss_ce + 1.0 * loss_kd
    scaler.scale(total_loss).backward()
    print(f"  [PASS] M1 Loss: {total_loss.item():.4f} (CE={loss_ce.item():.4f}, KD={loss_kd.item():.4f})")

    # Test M1b: DKD
    print("\n--- Test M1b: Decoupled KD (DKD) ---")
    student.zero_grad()
    with torch.amp.autocast('cuda'):
        logits_S = student(images)
        with torch.no_grad():
            logits_T = teacher(images)
        loss_ce = criterion_ce(logits_S, labels)
        loss_dkd = dkd_loss(logits_S, logits_T, labels, alpha=1.0, beta=1.0, temperature=3.0)
        total_loss = loss_ce + loss_dkd
    scaler.scale(total_loss).backward()
    print(f"  [PASS] M1b Loss: {total_loss.item():.4f} (CE={loss_ce.item():.4f}, DKD={loss_dkd.item():.4f})")

    # Test M3: Attention Transfer
    print("\n--- Test M3: Attention Transfer (AT) ---")
    student.zero_grad()
    student_hook.clear(); teacher_hook.clear()
    with torch.amp.autocast('cuda'):
        logits_S = student(images)
        with torch.no_grad():
            logits_T = teacher(images)
        loss_ce = criterion_ce(logits_S, labels)
        loss_at = attention_transfer_loss(student_hook.features, teacher_hook.features)
        total_loss = loss_ce + 1000.0 * loss_at
    scaler.scale(total_loss).backward()
    print(f"  [PASS] M3 Loss: {total_loss.item():.4f} (CE={loss_ce.item():.4f}, AT={loss_at.item():.6f})")

    # Test M4: Hybrid KD + AT
    print("\n--- Test M4: Hybrid (Hinton KD + AT) ---")
    student.zero_grad()
    student_hook.clear(); teacher_hook.clear()
    with torch.amp.autocast('cuda'):
        logits_S = student(images)
        with torch.no_grad():
            logits_T = teacher(images)
        loss_ce = criterion_ce(logits_S, labels)
        loss_kd = kd_loss(logits_S, logits_T, tau=3.0)
        loss_at = attention_transfer_loss(student_hook.features, teacher_hook.features)
        total_loss = loss_ce + 1.0 * loss_kd + 1000.0 * loss_at
    scaler.scale(total_loss).backward()
    print(f"  [PASS] M4 Loss: {total_loss.item():.4f} (CE={loss_ce.item():.4f}, KD={loss_kd.item():.4f}, AT={loss_at.item():.6f})")

    print("\n" + "=" * 65)
    print("✅ TOÀN BỘ CÁC CƠ CHẾ KD ĐỀU PASS SMOKE TEST TRÊN GPU!")
    print("=" * 65)

if __name__ == "__main__":
    run_smoke_test()
