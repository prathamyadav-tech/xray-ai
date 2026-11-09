"""Grad-CAM explanations for the DenseNet121 model."""
from __future__ import annotations

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .model import get_device


def compute_heatmap(model: torch.nn.Module, tensor: torch.Tensor, finding: str) -> np.ndarray:
    """Grad-CAM map for one finding.

    Returns a float32 array [224, 224] scaled to [0, 1]. The target layer is the
    final BatchNorm of the DenseNet feature extractor (`model.features[-1]`).
    """
    if finding not in model.pathologies:
        raise ValueError(f"Unknown finding: {finding}")
    idx = model.pathologies.index(finding)
    x = tensor.unsqueeze(0).to(get_device()).clone().requires_grad_(True)
    with GradCAM(model=model, target_layers=[model.features[-1]]) as cam:
        heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(idx)])[0]
    return heat.astype(np.float32)


def has_signal(heat: np.ndarray, eps: float = 1e-3) -> bool:
    """False when Grad-CAM found no region that pushed the score up.

    This is normal for low-scoring findings: Grad-CAM applies a ReLU, so a
    finding with no positively contributing regions yields an all-zero map.
    """
    return bool(heat.max() > eps)


def overlay(gray_u8: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a JET heatmap over a grayscale image. Returns RGB uint8."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    base = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2RGB)
    colour = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(base, 1.0 - alpha, colour, alpha, 0)
