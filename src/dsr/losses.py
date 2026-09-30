from __future__ import annotations

import torch
import torch.nn.functional as F


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Knowledge Distillation loss (Hinton et al., 2015).

    Formula:
        KL(softmax(z_S / tau) || softmax(z_T / tau)) * tau^2
    where input is student log-prob and target is teacher prob.
    Reduction is 'batchmean' to properly divide by batch size.
    """
    s = F.log_softmax(student_logits / tau, dim=1)
    t = F.softmax(teacher_logits / tau, dim=1).detach()
    return F.kl_div(s, t, reduction="batchmean") * (tau ** 2)


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
    a = feat.pow(2).sum(dim=1, keepdim=True)  # (B, 1, H, W)
    if size is not None and a.shape[-2:] != size:
        a = F.interpolate(a, size=size, mode="bilinear", align_corners=False)
    return F.normalize(a.flatten(1), p=2, dim=1, eps=eps)


def attention_transfer_loss(
    student_feats: list[torch.Tensor],
    teacher_feats: list[torch.Tensor],
) -> torch.Tensor:
    """Attention Transfer loss across multiple stages.

    L_AT = sum_{layer} ||Q_S^l - Q_T^l||_2^2
    averaged over batch and summed across layers.
    """
    loss = torch.tensor(0.0, device=student_feats[0].device)
    for fs, ft in zip(student_feats, teacher_feats):
        qs = attention_map(fs)
        qt = attention_map(ft, size=fs.shape[-2:]).detach()
        loss = loss + (qs - qt).pow(2).sum(dim=1).mean()
    return loss
