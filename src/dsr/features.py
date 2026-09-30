from __future__ import annotations

import torch
import torch.nn as nn


class FeatureHookManager:
    """Manages forward hooks to extract intermediate feature maps from specified layers.

    Designed for ResNet architectures (layer1, layer2, layer3, layer4).
    Ensures memory safety by allowing explicit clearing of stored tensors after each batch.
    """

    def __init__(self, model: nn.Module, layer_names: list[str] | None = None) -> None:
        self.model = model
        self.layer_names = layer_names or ["layer1", "layer2", "layer3", "layer4"]
        self.features: dict[str, torch.Tensor] = {}
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        for name in self.layer_names:
            layer = getattr(self.model, name, None)
            if layer is None:
                raise ValueError(f"Layer '{name}' not found on model {type(self.model).__name__}.")

            def _create_hook(layer_name: str):
                def hook(module: nn.Module, input: tuple, output: torch.Tensor):
                    self.features[layer_name] = output

                return hook

            self.hooks.append(layer.register_forward_hook(_create_hook(name)))

    def get_features(self) -> list[torch.Tensor]:
        """Return the collected feature maps in the order of layer_names."""
        return [self.features[name] for name in self.layer_names]

    def clear(self) -> None:
        """Clear cached feature tensors to free memory."""
        self.features.clear()

    def close(self) -> None:
        """Remove all hooks from the model."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.features.clear()

    def __enter__(self) -> FeatureHookManager:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
