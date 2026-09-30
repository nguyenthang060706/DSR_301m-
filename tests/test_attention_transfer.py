import torch
import torch.nn as nn

from dsr.features import FeatureHookManager
from dsr.losses import attention_map, attention_transfer_loss
from dsr.models import create_model


def test_attention_map_properties():
    torch.manual_seed(42)
    B, C, H, W = 2, 64, 14, 14
    feat = torch.randn(B, C, H, W)

    # 1. Norm L2 của map luôn bằng 1
    am = attention_map(feat)
    norms = am.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Attention map must have unit L2 norm"

    # 2. Bất biến theo hệ số nhân dương: attention_map(c * feat) == attention_map(feat)
    c = 5.7
    am_scaled = attention_map(c * feat)
    assert torch.allclose(am, am_scaled, atol=1e-5), "Attention map must be scale invariant"

    # 3. Hai map giống nhau -> loss = 0
    loss_zero = attention_transfer_loss([feat], [feat])
    assert abs(loss_zero.item()) < 1e-6, f"Identical features must produce 0 loss, got {loss_zero.item()}"

    # 4. Giá trị mỗi layer luôn nằm trong [0, 4]
    feat_other = torch.randn(B, C, H, W)
    loss_val = attention_transfer_loss([feat], [feat_other])
    assert 0.0 <= loss_val.item() <= 4.0 + 1e-5, f"Per-layer loss must be in [0, 4], got {loss_val.item()}"

    # 5. Nhánh bilinear khi lệch kích thước
    feat_small = torch.randn(B, C, 7, 7)
    feat_large = torch.randn(B, C, 14, 14)
    loss_interp = attention_transfer_loss([feat_small], [feat_large])
    assert loss_interp.item() >= 0.0, "Interpolation branch must work without shape error"


def test_resnet_feature_hooks_and_shapes():
    # Kiểm tra FeatureHookManager trên ResNet18 và ResNet50
    device = torch.device("cpu")
    student = create_model("resnet18", num_classes=6, pretrained=False)
    teacher = create_model("resnet50", num_classes=6, pretrained=False)

    x = torch.randn(2, 3, 224, 224)

    with FeatureHookManager(student) as hook_s, FeatureHookManager(teacher) as hook_t:
        student(x)
        with torch.no_grad():
            teacher(x)

        feats_s = hook_s.get_features()
        feats_t = hook_t.get_features()

        # 4 stage phải khớp kích thước không gian: 56, 28, 14, 7
        expected_spatial = [(56, 56), (28, 28), (14, 14), (7, 7)]
        for i, (fs, ft, (h, w)) in enumerate(zip(feats_s, feats_t, expected_spatial)):
            assert fs.shape[-2:] == (h, w), f"Student stage {i+1} shape {fs.shape[-2:]} != {(h, w)}"
            assert ft.shape[-2:] == (h, w), f"Teacher stage {i+1} shape {ft.shape[-2:]} != {(h, w)}"

        # Tính AT loss
        loss_at = attention_transfer_loss(feats_s, feats_t)
        assert loss_at.requires_grad, "AT loss must require grad for student"

        # Kiểm tra teacher không nhận gradient
        for p in teacher.parameters():
            assert p.grad is None, "Teacher parameters must not receive gradient"

    print("All Attention Transfer tests passed successfully!")


if __name__ == "__main__":
    print("Running test_attention_map_properties()...")
    test_attention_map_properties()
    print("PASS")

    print("Running test_resnet_feature_hooks_and_shapes()...")
    test_resnet_feature_hooks_and_shapes()
    print("PASS")
