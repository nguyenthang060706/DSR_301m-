from pathlib import Path
import torch
import torch.nn.functional as F

from dsr.losses import kd_loss, dkd_loss, attention_transfer_loss


def test_kd_loss_formula_and_properties():
    torch.manual_seed(42)
    batch_size, num_classes = 8, 12
    logits_S = torch.randn(batch_size, num_classes)
    logits_T = torch.randn(batch_size, num_classes)
    temperature = 3.0

    # Inline reference formula
    student_log_prob = F.log_softmax(logits_S / temperature, dim=1)
    teacher_prob = F.softmax(logits_T / temperature, dim=1)
    expected_loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature ** 2)

    actual_loss = kd_loss(logits_S, logits_T, temperature)

    assert torch.allclose(actual_loss, expected_loss, atol=1e-7), (
        f"kd_loss {actual_loss.item()} != expected {expected_loss.item()}"
    )
    assert actual_loss.item() >= 0.0, "KD loss must be non-negative"

    # KL=0 when distributions are identical
    self_loss = kd_loss(logits_T, logits_T, temperature)
    assert abs(self_loss.item()) < 1e-5, f"KL(T||T) must be close to 0, got {self_loss.item()}"


def test_dkd_loss_properties():
    torch.manual_seed(42)
    batch_size, num_classes = 8, 12
    logits_S = torch.randn(batch_size, num_classes)
    logits_T = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    temperature = 3.0

    loss = dkd_loss(logits_S, logits_T, target, alpha=1.0, beta=1.0, temperature=temperature)
    assert torch.isfinite(loss), "DKD loss must be finite"
    assert loss.item() >= 0.0, f"DKD loss must be non-negative, got {loss.item()}"

    # Self-distillation should yield near-zero loss
    self_loss = dkd_loss(logits_T, logits_T, target, alpha=1.0, beta=1.0, temperature=temperature)
    assert abs(self_loss.item()) < 1e-4, f"DKD self-distillation must be near zero, got {self_loss.item()}"


def test_attention_transfer_properties():
    torch.manual_seed(42)
    batch_size, channels, h, w = 4, 64, 14, 14
    feat_S = [torch.randn(batch_size, channels, h, w)]
    feat_T = [torch.randn(batch_size, channels * 2, h, w)]

    at_loss = attention_transfer_loss(feat_S, feat_T)
    assert torch.isfinite(at_loss), "AT loss must be finite"
    assert at_loss.item() >= 0.0, "AT loss must be non-negative"

    # Identical features yield 0 loss
    zero_loss = attention_transfer_loss(feat_S, feat_S)
    assert abs(zero_loss.item()) < 1e-5, f"AT loss for identical features must be 0, got {zero_loss.item()}"


if __name__ == "__main__":
    print("Testing kd_loss...")
    test_kd_loss_formula_and_properties()
    print("PASS")

    print("Testing dkd_loss...")
    test_dkd_loss_properties()
    print("PASS")

    print("Testing attention_transfer_loss...")
    test_attention_transfer_properties()
    print("PASS")

    print("\nALL KD LOSS TESTS PASSED!")
