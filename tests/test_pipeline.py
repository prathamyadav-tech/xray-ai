import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config
from src.gradcam import compute_heatmap, has_signal, overlay
from src.model import load_model
from src.predict import band, predict_all, predict_targets
from src.preprocess import InvalidImageError, preprocess, validate_upload


def _png(arr: np.ndarray, mode: str = "L") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def gray_bytes():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:300, 0:300]
    blob = 180 * np.exp(-(((xx - 150) ** 2 + (yy - 150) ** 2) / 8000))
    return _png((blob + rng.normal(0, 8, blob.shape)).clip(0, 255).astype(np.uint8))


@pytest.fixture(scope="module")
def model():
    return load_model()


# ---- validation ----------------------------------------------------------
def test_rejects_bad_extension(gray_bytes):
    with pytest.raises(InvalidImageError, match="Unsupported"):
        validate_upload(gray_bytes, "scan.gif")


def test_rejects_empty():
    with pytest.raises(InvalidImageError, match="empty"):
        validate_upload(b"", "a.png")


def test_rejects_corrupt():
    with pytest.raises(InvalidImageError, match="could not be read"):
        validate_upload(b"not an image", "a.png")


def test_rejects_tiny():
    with pytest.raises(InvalidImageError, match="too small"):
        validate_upload(_png(np.zeros((40, 40), np.uint8)), "a.png")


def test_rejects_colour_photo():
    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[..., 0] = 255  # pure red
    with pytest.raises(InvalidImageError, match="colour"):
        validate_upload(_png(rgb, "RGB"), "a.png")


def test_rejects_too_large(monkeypatch, gray_bytes):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    with pytest.raises(InvalidImageError, match="larger"):
        validate_upload(gray_bytes, "a.png")


def test_accepts_grayscale(gray_bytes):
    assert validate_upload(gray_bytes, "a.png").size == (300, 300)


# ---- preprocessing -------------------------------------------------------
def test_preprocess_shape_and_range(gray_bytes):
    pre = preprocess(validate_upload(gray_bytes, "a.png"))
    assert tuple(pre.tensor.shape) == (1, config.IMG_SIZE, config.IMG_SIZE)
    assert pre.tensor.min() >= -1024 and pre.tensor.max() <= 1024
    assert pre.display.dtype == np.uint8 and pre.display.shape == (224, 224)


def test_preprocess_handles_16bit():
    arr = (np.random.default_rng(1).random((200, 200)) * 65535).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    pre = preprocess(validate_upload(buf.getvalue(), "a.png"))
    assert tuple(pre.tensor.shape) == (1, 224, 224)


# ---- model / prediction --------------------------------------------------
def test_predictions_in_unit_range(model, gray_bytes):
    pre = preprocess(validate_upload(gray_bytes, "a.png"))
    scores = predict_targets(model, pre.tensor)
    assert set(scores) == set(config.TARGETS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert len(predict_all(model, pre.tensor)) == len(model.pathologies)


def test_band_boundaries():
    assert band(0.1) == "Low" and band(0.45) == "Moderate" and band(0.9) == "High"


# ---- grad-cam ------------------------------------------------------------
@pytest.mark.parametrize("finding", config.TARGETS)
def test_gradcam_shape_and_range(model, gray_bytes, finding):
    pre = preprocess(validate_upload(gray_bytes, "a.png"))
    heat = compute_heatmap(model, pre.tensor, finding)
    assert heat.shape == (224, 224)
    assert 0.0 <= heat.min() and heat.max() <= 1.0 + 1e-6
    ov = overlay(pre.display, heat, 0.5)
    assert ov.shape == (224, 224, 3) and ov.dtype == np.uint8


def test_gradcam_unknown_finding(model, gray_bytes):
    pre = preprocess(validate_upload(gray_bytes, "a.png"))
    with pytest.raises(ValueError):
        compute_heatmap(model, pre.tensor, "NotAFinding")


def test_real_sample_if_present(model):
    samples = sorted(config.SAMPLE_DIR.glob("*.png"))
    if not samples:
        pytest.skip("no sample images")
    p = samples[0]
    pre = preprocess(validate_upload(p.read_bytes(), p.name))
    assert all(0 <= v <= 1 for v in predict_targets(model, pre.tensor).values())


def test_has_signal():
    assert not has_signal(np.zeros((224, 224), np.float32))
    h = np.zeros((224, 224), np.float32); h[10, 10] = 1.0
    assert has_signal(h)
