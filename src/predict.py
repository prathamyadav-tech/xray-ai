"""Inference: image tensor -> per-finding scores."""
from __future__ import annotations

import torch

from . import config
from .model import get_device


def predict_all(model: torch.nn.Module, tensor: torch.Tensor) -> dict[str, float]:
    """Return a score in [0, 1] for every pathology the model outputs.

    TorchXRayVision DenseNet outputs are already normalised to [0, 1], so no
    extra sigmoid is applied.
    """
    x = tensor.unsqueeze(0).to(get_device())
    with torch.no_grad():
        out = model(x)[0].cpu()
    return {name: float(v) for name, v in zip(model.pathologies, out)}


def predict_targets(model: torch.nn.Module, tensor: torch.Tensor) -> dict[str, float]:
    """Scores for the findings configured in `config.TARGETS` only."""
    allp = predict_all(model, tensor)
    return {t: allp[t] for t in config.TARGETS}


def band(score: float) -> str:
    """Human-friendly label. Display only, not a clinical threshold."""
    if score < config.BAND_LOW:
        return "Low"
    if score < config.BAND_HIGH:
        return "Moderate"
    return "High"
