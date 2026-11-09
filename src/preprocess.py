"""Upload validation and preprocessing.

TorchXRayVision expects a single-channel image in the [-1024, 1024] range,
centre-cropped to a square and resized to 224x224. Skipping this scaling gives
wrong predictions without any error, so it lives in one place here.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
from PIL import Image, UnidentifiedImageError

from . import config


class InvalidImageError(ValueError):
    """Raised with a user-friendly message when an upload is rejected."""


@dataclass
class Preprocessed:
    tensor: torch.Tensor      # [1, 224, 224], range [-1024, 1024]
    display: np.ndarray       # [224, 224] uint8, what the model actually sees


def _ext(name: str) -> str:
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def validate_upload(data: bytes, filename: str) -> Image.Image:
    """Validate raw upload bytes and return a decoded PIL image.

    Checks: extension, size, decodability, minimum resolution and that the
    image is (nearly) grayscale. The grayscale check is a cheap sanity filter,
    not a guarantee that the picture is a chest X-ray.
    """
    if _ext(filename) not in config.ALLOWED_EXT:
        raise InvalidImageError(
            f"Unsupported file type. Please upload one of: "
            f"{', '.join(sorted(config.ALLOWED_EXT))}."
        )
    if len(data) == 0:
        raise InvalidImageError("The file is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise InvalidImageError(f"File is larger than {config.MAX_UPLOAD_MB} MB.")

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise InvalidImageError("The file could not be read as an image.") from None

    if min(img.size) < config.MIN_SIDE_PX:
        raise InvalidImageError(
            f"Image is too small ({img.size[0]}x{img.size[1]}). "
            f"Minimum side is {config.MIN_SIDE_PX}px."
        )

    if img.mode in ("RGB", "RGBA", "P"):
        hsv = np.asarray(img.convert("RGB").convert("HSV"), dtype=np.float32)
        if hsv[..., 1].mean() / 255.0 > config.MAX_SATURATION:
            raise InvalidImageError(
                "This looks like a colour photo. Chest X-rays are grayscale; "
                "please upload an X-ray image."
            )
    return img


def _to_uint8_gray(img: Image.Image) -> np.ndarray:
    """Convert any PIL image (incl. 16-bit) to a uint8 grayscale array."""
    arr = np.asarray(img)
    if arr.dtype != np.uint8:  # e.g. 16-bit PNG
        arr = arr.astype(np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        arr = ((arr - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
        img = Image.fromarray(arr)
    return np.asarray(img.convert("L"))


_TRANSFORM = torchvision.transforms.Compose(
    [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(config.IMG_SIZE)]
)


def preprocess(img: Image.Image) -> Preprocessed:
    """PIL image -> model-ready tensor plus a displayable version."""
    gray = _to_uint8_gray(img).astype(np.float32)
    norm = xrv.datasets.normalize(gray, 255)         # -> [-1024, 1024]
    norm = _TRANSFORM(norm[None, ...])               # [1, 224, 224]
    tensor = torch.from_numpy(np.ascontiguousarray(norm)).float()
    display = ((tensor[0].numpy() + 1024.0) / 2048.0 * 255.0).clip(0, 255)
    return Preprocessed(tensor=tensor, display=display.astype(np.uint8))
