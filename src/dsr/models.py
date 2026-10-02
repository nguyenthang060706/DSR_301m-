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
    "resnet18": ModelSpec("resnet18", "student", notes="Tier A student-only baseline."),
    "mobilenet_v3_small": ModelSpec("mobilenet_v3_small", "tier_b", notes="Tier B candidate 1."),
    "mobilenet_v3_large": ModelSpec("mobilenet_v3_large", "tier_b", notes="Tier B candidate 2."),
    "mobilenetv4_conv_small": ModelSpec("mobilenetv4_conv_small", "tier_b", notes="Tier B candidate 3 (timm)."),
    "efficientnet_b0": ModelSpec("efficientnet_b0", "baseline", notes="External baseline."),
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
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    elif name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model
    elif name == "mobilenetv4_conv_small":
        import timm
        model = timm.create_model("mobilenetv4_conv_small", pretrained=pretrained, num_classes=num_classes)
        return model
    elif name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
    else:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Known models: {known}")
