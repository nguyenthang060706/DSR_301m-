import torch
import torch.nn as nn
import pytest

from dsr.losses import FitNetRegressor, fitnet_loss


def test_fitnet_regressor_shape():
    regressor = FitNetRegressor(in_channels=256, out_channels=1024)
    x = torch.randn(2, 256, 14, 14)
    out = regressor(x)
    assert out.shape == (2, 1024, 14, 14), f"Expected shape (2, 1024, 14, 14), got {out.shape}"


def test_fitnet_loss_non_negative_and_zero_identity():
    # Rule 9: Loss mới phải có unit test (không âm, =0 khi self-distill)
    feat = torch.randn(4, 256, 7, 7)
    
    # Self-distillation: loss must be 0.0
    loss_self = fitnet_loss(feat, feat, regressor=None)
    assert torch.isclose(loss_self, torch.tensor(0.0), atol=1e-7), f"Expected 0.0 on self-distillation, got {loss_self.item()}"

    # Non-negativity test with arbitrary tensors
    s_feat = torch.randn(4, 256, 7, 7)
    t_feat = torch.randn(4, 256, 7, 7)
    loss = fitnet_loss(s_feat, t_feat, regressor=None)
    assert loss.item() >= 0.0, "FitNet loss must be non-negative"


def test_fitnet_loss_gradient_flow():
    # Gradient flow to both student feature and regressor parameters
    regressor = FitNetRegressor(in_channels=64, out_channels=128)
    s_feat = torch.randn(2, 64, 8, 8, requires_grad=True)
    t_feat = torch.randn(2, 128, 8, 8)

    loss = fitnet_loss(s_feat, t_feat, regressor)
    loss.backward()

    assert s_feat.grad is not None, "Gradient must flow back to student features"
    assert regressor.conv.weight.grad is not None, "Gradient must flow back to regressor conv"


def test_fitnet_spatial_interpolation():
    # Different spatial sizes (e.g. 7x7 vs 14x14)
    s_feat = torch.randn(2, 128, 7, 7)
    t_feat = torch.randn(2, 128, 14, 14)

    loss = fitnet_loss(s_feat, t_feat, regressor=None)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
