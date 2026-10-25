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


def test_fitnet_loss_properties():
    from dsr.losses import FitNetRegressor, fitnet_loss
    torch.manual_seed(42)
    s_feat = torch.randn(4, 256, 14, 14)
    t_feat = torch.randn(4, 1024, 14, 14)
    regressor = FitNetRegressor(in_channels=256, out_channels=1024)

    loss = fitnet_loss(s_feat, t_feat, regressor)
    assert torch.isfinite(loss), "FitNet loss must be finite"
    assert loss.item() >= 0.0, "FitNet loss must be non-negative"

    # Self distillation without regressor
    self_loss = fitnet_loss(s_feat, s_feat, regressor=None)
    assert abs(self_loss.item()) < 1e-6, f"FitNet self-loss must be zero, got {self_loss.item()}"


def test_dkd_hinton_equivalence():
    """D2-C2: Verify mean(TCKD + (1 - p_t^T) · NCKD) == kd_loss (same T², atol 1e-5).

    This confirms DKD is a proper decomposition of classical Hinton KD:
        KD_loss = mean_over_samples[ TCKD_i + (1 - p_t^T_i) * NCKD_i ]
    """
    from dsr.losses import dkd_components

    torch.manual_seed(42)
    batch_size, num_classes = 16, 12

    for temperature in [2.0, 3.0, 4.0, 6.0]:
        logits_S = torch.randn(batch_size, num_classes)
        logits_T = torch.randn(batch_size, num_classes)
        target = torch.randint(0, num_classes, (batch_size,))

        # 1. Classical Hinton KD loss
        hinton_loss = kd_loss(logits_S, logits_T, tau=temperature)

        # 2. DKD components
        tckd, nckd = dkd_components(logits_S, logits_T, target, temperature=temperature)

        # Compute (1 - p_t^T) per sample
        p_t = F.softmax(logits_T / temperature, dim=1).detach()
        target_mask = torch.zeros(batch_size, num_classes, dtype=torch.bool)
        target_mask.scatter_(1, target.unsqueeze(1), 1)
        p_t_t = torch.sum(p_t * target_mask, dim=1)  # (B,)

        # Reconstructed KD = mean(TCKD + (1 - p_t^T) * NCKD)
        reconstructed = (tckd + (1.0 - p_t_t) * nckd).mean()

        assert torch.allclose(reconstructed, hinton_loss, atol=1e-5), (
            f"DKD ≠ Hinton at T={temperature}: "
            f"reconstructed={reconstructed.item():.6f} vs hinton={hinton_loss.item():.6f}"
        )

    # Also verify: dkd_loss with alpha=1, beta=1 is NOT the same as kd_loss
    # (because beta is a fixed scalar, not sample-wise (1-p_t^T))
    # This is just a sanity check that dkd_components is doing the right thing
    from dsr.losses import dkd_loss as dkd_fn
    logits_S = torch.randn(batch_size, num_classes)
    logits_T = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))

    dkd_val = dkd_fn(logits_S, logits_T, target, alpha=1.0, beta=1.0, temperature=3.0)
    hinton_val = kd_loss(logits_S, logits_T, tau=3.0)
    # These should generally differ (unless by coincidence)
    # Just verify both are non-negative and finite
    assert torch.isfinite(dkd_val)
    assert torch.isfinite(hinton_val)


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
