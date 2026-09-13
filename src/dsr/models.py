from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    role: str
    has_eca: bool = False
    notes: str = ""


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "resnet50": ModelSpec("resnet50", "teacher", notes="Week-1 teacher baseline."),
    "resnet18": ModelSpec("resnet18", "student", notes="Week-1 student-only baseline."),
    "resnet18_eca": ModelSpec("resnet18_eca", "student", has_eca=True),
}


def create_model(name: str, num_classes: int, pretrained: bool = True):
    try:
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch/torchvision are required for model creation. Install requirements.txt first."
        ) from exc

    if name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
    elif name == "resnet18_eca":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model = add_stage_eca(model)
    else:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class ECALayer:
    """Factory wrapper to keep imports lazy until torch is installed."""

    def __new__(cls, channels: int, kernel_size: int = 3):
        import torch.nn as nn

        class _ECALayer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.avg_pool = nn.AdaptiveAvgPool2d(1)
                self.conv = nn.Conv1d(
                    1,
                    1,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                )
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                y = self.avg_pool(x)
                y = self.conv(y.squeeze(-1).transpose(-1, -2))
                y = self.sigmoid(y.transpose(-1, -2).unsqueeze(-1))
                return x * y.expand_as(x)

        return _ECALayer()


def add_stage_eca(resnet18):
    import torch.nn as nn

    resnet18.layer1 = nn.Sequential(resnet18.layer1, ECALayer(64))
    resnet18.layer2 = nn.Sequential(resnet18.layer2, ECALayer(128))
    resnet18.layer3 = nn.Sequential(resnet18.layer3, ECALayer(256))
    resnet18.layer4 = nn.Sequential(resnet18.layer4, ECALayer(512))
    return resnet18
