"""Central configuration for the Chest X-Ray AI project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "sample_images"

# --- Model -----------------------------------------------------------------
# TorchXRayVision DenseNet121 trained on several public chest X-ray datasets.
# Only the "all" weights train every output; other weight sets leave some
# outputs untrained (they would predict randomly).
WEIGHTS = "densenet121-res224-all"
IMG_SIZE = 224

# Findings shown in the app. Names must exist in model.pathologies.
TARGETS = ["Pneumonia", "Cardiomegaly", "Effusion", "Pneumothorax"]
DISPLAY_NAMES = {
    "Pneumonia": "Pneumonia",
    "Cardiomegaly": "Cardiomegaly (enlarged heart)",
    "Effusion": "Pleural Effusion",
    "Pneumothorax": "Pneumothorax (collapsed lung)",
}

# --- Upload validation -----------------------------------------------------
ALLOWED_EXT = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_MB = 10
MIN_SIDE_PX = 128
# Mean colour saturation above this => probably not an X-ray (they are grayscale).
MAX_SATURATION = 0.15

# --- Presentation ----------------------------------------------------------
# Score bands for display only. These are NOT clinical thresholds.
BAND_LOW = 0.30
BAND_HIGH = 0.60

DISCLAIMER = (
    "Research / educational prototype. NOT a medical diagnosis and NOT a "
    "medical device. Scores are model outputs, not probabilities of disease. "
    "Always consult a qualified medical professional."
)
