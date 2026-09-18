import torch
import torch.nn.functional as F
import pytest

def test_kd_loss_properties():
    torch.manual_seed(42)
    batch_size, num_classes = 8, 6
    logits_S = torch.randn(batch_size, num_classes)
    logits_T = torch.randn(batch_size, num_classes)
    temperature = 3.0

    # Tính toán theo đúng công thức trong train_kd.py
    student_log_prob = F.log_softmax(logits_S / temperature, dim=1)
    teacher_prob = F.softmax(logits_T / temperature, dim=1)

    loss_kd = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)

    # 1. Non-negativity: KL Divergence luôn >= 0
    assert loss_kd.item() >= 0.0, "KD loss must be non-negative"

    # 2. Identical distributions: KL(T || T) = 0
    teacher_log_prob = F.log_softmax(logits_T / temperature, dim=1)
    loss_self = F.kl_div(teacher_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)
    assert abs(loss_self.item()) < 1e-5, f"KL(T||T) must be close to 0, got {loss_self.item()}"

def test_kd_direction_error():
    # Kiểm tra bắt lỗi nếu truyền ngược input và target
    # PyTorch F.kl_div yêu cầu input = log_prob, target = prob.
    torch.manual_seed(42)
    logits_S = torch.randn(2, 6)
    logits_T = torch.randn(2, 6)
    
    student_prob = F.softmax(logits_S, dim=1)
    teacher_prob = F.softmax(logits_T, dim=1)
    
    # Truyền ngược (cả 2 đều là prob, thiếu log)
    wrong_loss = F.kl_div(student_prob, teacher_prob, reduction="batchmean")
    # Thông thường khi làm sai thế này sẽ ra giá trị âm
    assert wrong_loss.item() < 0, "Swapped/Wrong inputs should mathematically yield invalid (negative) KL"
