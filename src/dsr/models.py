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
    "mobilenet_v3_large": ModelSpec("mobilenet_v3_large", "baseline", notes="Week-2 baseline CNN."),
    "mobilenet_v3_small": ModelSpec("mobilenet_v3_small", "baseline", notes="Week-2 baseline CNN."),
    "efficientnet_b0": ModelSpec("efficientnet_b0", "baseline", notes="Week-2 baseline CNN."),
    "efficientformer_l1": ModelSpec("efficientformer_l1", "baseline", notes="Week-2 baseline CNN with Attention."),
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
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
    elif name == "efficientformer_l1":
        import timm
        model = timm.create_model("efficientformer_l1", pretrained=pretrained, num_classes=num_classes)
        return model # timm handles num_classes internally
    else:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")

    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, 'classifier'):
        if name.startswith("mobilenet_v3"):
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        elif name.startswith("efficientnet"):
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        
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
