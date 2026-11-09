"""Model loading."""
from __future__ import annotations

from functools import lru_cache

import torch
import torchxrayvision as xrv

from . import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def load_model(weights: str = config.WEIGHTS) -> torch.nn.Module:
    """Load the pretrained DenseNet121 once and keep it in eval mode."""
    model = xrv.models.DenseNet(weights=weights)
    model.eval().to(get_device())
    missing = [t for t in config.TARGETS if t not in model.pathologies]
    if missing:
        raise RuntimeError(f"Model has no outputs for: {missing}")
    return model
