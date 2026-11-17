"""MobileNetV3-Small preentrenada en ImageNet con cabeza de 29 clases.

Es el modelo que ganó la selección del Lab 3 (robust-v3). Se mantiene igual
para que la comparación CPU vs GPU sea sobre el mismo algoritmo.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .constants import CLASSES


def build_model(architecture: str = "mobilenet_v3_small", pretrained: bool = True) -> nn.Module:
    if architecture != "mobilenet_v3_small":
        raise ValueError(f"Arquitectura no soportada: {architecture}")
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    last = cast(nn.Linear, model.classifier[-1])
    model.classifier[-1] = nn.Linear(last.in_features, len(CLASSES))
    return model


def head(model: nn.Module) -> nn.Module:
    return cast(nn.Module, cast(Any, model).classifier)


def set_trainable(model: nn.Module, unfrozen_blocks: int) -> None:
    """Congela todo y libera la cabeza más los últimos `unfrozen_blocks` bloques de features."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head(model).parameters():
        parameter.requires_grad = True
    if unfrozen_blocks > 0:
        for block in list(cast(Any, model).features)[-unfrozen_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict[str, Any]]:
    head_ids = {id(p) for p in head(model).parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups: list[dict[str, Any]] = [{"params": [p for p in head(model).parameters()], "lr": head_lr, "name": "head"}]
    if backbone and backbone_lr > 0:
        groups.insert(0, {"params": backbone, "lr": backbone_lr, "name": "backbone"})
    return groups


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model: nn.Module, path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu", torch.float32) if v.is_floating_point() else v.detach().cpu()
             for k, v in model.state_dict().items()}
    torch.save({"state_dict": state, "classes": list(CLASSES), **metadata}, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(payload.get("architecture", "mobilenet_v3_small"), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.eval().to(device)
    return model, payload
