from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Knowledge Distillation loss (Hinton et al., 2015).

    Formula:
        KL(softmax(z_T / tau) || softmax(z_S / tau)) * tau^2
    where input to F.kl_div is student log-prob and target is teacher prob.
    Mathematically, F.kl_div(log(S), T) = sum(T * (log(T) - log(S))) = KL(T || S).
    Reduction is 'batchmean' to properly divide by batch size.
    """
    s = F.log_softmax(student_logits / tau, dim=1)
    t = F.softmax(teacher_logits / tau, dim=1).detach()
    return F.kl_div(s, t, reduction="batchmean") * (tau ** 2)


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 1.0,
    temperature: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Decoupled Knowledge Distillation (DKD) loss (Zhao et al., CVPR 2022).

    Decouples classical KD loss into:
      1. Target Class Knowledge Distillation (TCKD)
      2. Non-target Class Knowledge Distillation (NCKD)

    Formula:
        L_DKD = alpha * TCKD + beta * NCKD

    Property:
        When alpha = 1.0 and beta = (1 - p_t^T) sample-wise,
        L_DKD is mathematically identical to Hinton's KD loss.
    """
    # Softmax probabilities scaled by temperature
    p_s = F.softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1).detach()

    batch_size, num_classes = student_logits.shape
    device = student_logits.device

    # Target class mask (One-Hot)
    target_mask = torch.zeros(batch_size, num_classes, dtype=torch.bool, device=device)
    target_mask.scatter_(1, target.unsqueeze(1), 1)

    # Probabilities of target class: p_t^S and p_t^T
    p_s_t = torch.sum(p_s * target_mask, dim=1, keepdim=True)  # (B, 1)
    p_t_t = torch.sum(p_t * target_mask, dim=1, keepdim=True)  # (B, 1)

    # 1. TCKD: Binary Cross-Entropy / KL between target and non-target probability mass
    # b_S = [p_s_t, 1 - p_s_t], b_T = [p_t_t, 1 - p_t_t]
    # KL(b_T || b_S) = p_t_t * log(p_t_t / p_s_t) + (1 - p_t_t) * log((1 - p_t_t) / (1 - p_s_t))
    p_s_t_clamp = p_s_t.clamp(eps, 1.0 - eps)
    p_t_t_clamp = p_t_t.clamp(eps, 1.0 - eps)
    
    tckd = (
        p_t_t_clamp * (torch.log(p_t_t_clamp) - torch.log(p_s_t_clamp))
        + (1.0 - p_t_t_clamp) * (torch.log(1.0 - p_t_t_clamp) - torch.log(1.0 - p_s_t_clamp))
    ).sum(dim=1).mean() * (temperature ** 2)

    # 2. NCKD: Normalized distribution over non-target classes
    # mask out the target class
    non_target_mask = ~target_mask
    
    # Normalized non-target distribution
    # \tilde{p}_i = p_i / (1 - p_t)
    denom_s = (1.0 - p_s_t).clamp(min=eps)
    denom_t = (1.0 - p_t_t).clamp(min=eps)

    p_s_non_target = (p_s / denom_s).clamp(min=eps)
    p_t_non_target = (p_t / denom_t).clamp(min=eps)

    # Compute KL divergence over non-target elements: \tilde{p}_T * (log(\tilde{p}_T) - log(\tilde{p}_S))
    kl_non_target = p_t_non_target * (torch.log(p_t_non_target) - torch.log(p_s_non_target))
    # Sum only over non-target entries
    nckd = (kl_non_target * non_target_mask).sum(dim=1).mean() * (temperature ** 2)

    return alpha * tckd + beta * nckd


def dkd_components(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 3.0,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract TCKD and NCKD components per sample (D2-C2).

    Returns:
        (tckd_per_sample, nckd_per_sample): each of shape (B,), scaled by T².

    Note: mean(tckd + (1 - p_t^T) * nckd) should equal kd_loss(student, teacher, temperature)
    when both are computed with the same T².
    """
    p_s = F.softmax(student_logits / temperature, dim=1)
    p_t = F.softmax(teacher_logits / temperature, dim=1).detach()

    batch_size, num_classes = student_logits.shape
    device = student_logits.device

    target_mask = torch.zeros(batch_size, num_classes, dtype=torch.bool, device=device)
    target_mask.scatter_(1, target.unsqueeze(1), 1)

    p_s_t = torch.sum(p_s * target_mask, dim=1, keepdim=True)  # (B, 1)
    p_t_t = torch.sum(p_t * target_mask, dim=1, keepdim=True)  # (B, 1)

    # TCKD per sample
    p_s_t_c = p_s_t.clamp(eps, 1.0 - eps)
    p_t_t_c = p_t_t.clamp(eps, 1.0 - eps)
    tckd = (
        p_t_t_c * (torch.log(p_t_t_c) - torch.log(p_s_t_c))
        + (1.0 - p_t_t_c) * (torch.log(1.0 - p_t_t_c) - torch.log(1.0 - p_s_t_c))
    ).sum(dim=1) * (temperature ** 2)  # (B,)

    # NCKD per sample
    non_target_mask = ~target_mask
    denom_s = (1.0 - p_s_t).clamp(min=eps)
    denom_t = (1.0 - p_t_t).clamp(min=eps)
    p_s_nt = (p_s / denom_s).clamp(min=eps)
    p_t_nt = (p_t / denom_t).clamp(min=eps)
    kl_nt = p_t_nt * (torch.log(p_t_nt) - torch.log(p_s_nt))
    nckd = (kl_nt * non_target_mask).sum(dim=1) * (temperature ** 2)  # (B,)

    return tckd, nckd


def attention_map(
    feat: torch.Tensor,
    size: tuple[int, int] | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute spatial attention map from activation tensor (Zagoruyko & Komodakis, 2017).

    Args:
        feat: Feature map of shape (B, C, H, W)
        size: Target spatial size (H, W) for interpolation if dimensions mismatch.
        eps: Small epsilon for L2 normalization.

    Returns:
        Flattened L2-normalized spatial attention map of shape (B, H * W).
    """
    # Spatial energy: sum of squared activations along channel dimension
    a = feat.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    if size is not None and a.shape[-2:] != size:
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    # L2 normalize over spatial dimensions
    return F.normalize(a.flatten(1), p=2, dim=1, eps=eps)


def attention_transfer_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
) -> torch.Tensor:
    """Attention Transfer loss across multiple stages (Zagoruyko & Komodakis, 2017).

    L_AT = (1 / 2) * sum_{layer} ||Q_S^l - Q_T^l||_2^2
    where Q_S and Q_T are L2-normalized spatial attention maps.
    """
    loss = torch.tensor(0.0, device=student_feats[0].device)
    for fs, ft in zip(student_feats, teacher_feats):
        qs = attention_map(fs)
        qt = attention_map(ft, size=fs.shape[-2:]).detach()
        loss = loss + 0.5 * (qs - qt).pow(2).sum(dim=1).mean()
    return loss


class FitNetRegressor(nn.Module):
    """1x1 Convolution regressor for FitNet intermediate feature alignment (Romero et al., 2015).

    Single-stage variant per MASTER v10.0 §D6 & D1-C4.
    Maps student intermediate feature channels to teacher intermediate feature channels at layer3.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


def fitnet_loss(
    student_feat: torch.Tensor,
    teacher_feat: torch.Tensor,
    regressor: nn.Module | None = None,
) -> torch.Tensor:
    """FitNet hint-based feature distillation loss (Romero et al., 2015).

    L_FitNet = 0.5 * MSE(regressor(student_feat), teacher_feat.detach())
    """
    if regressor is not None:
        student_feat = regressor(student_feat)

    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.interpolate(
            student_feat, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False
        )

    return 0.5 * F.mse_loss(student_feat, teacher_feat.detach(), reduction="mean")

