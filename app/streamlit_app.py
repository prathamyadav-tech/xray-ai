"""Streamlit UI for AI-Powered Chest X-Ray Analysis.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import config
from src.gradcam import compute_heatmap, has_signal, overlay
from src.model import load_model
from src.predict import band, predict_all
from src.preprocess import InvalidImageError, preprocess, validate_upload

st.set_page_config(page_title="AI Chest X-Ray Analysis", page_icon="🫁", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#172554; --blue:#08618a; --teal:#06b6a4; --pale:#eaf4f8; --muted:#64748b; }
.stApp { background:linear-gradient(180deg,#f7fbfd 0%,#ffffff 48%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.025em; }
.block-container { max-width:1280px; padding-top:2rem; padding-bottom:3rem; }
.hero { border-radius:24px; padding:30px 34px; color:#fff; background:linear-gradient(120deg,#10213b 0%,#075a7c 67%,#087f83 100%); box-shadow:0 16px 38px #12314a22; position:relative; overflow:hidden; }
.hero:after { content:'✣'; position:absolute; right:5%; top:-60px; font-size:290px; color:#ffffff0d; transform:rotate(18deg); }
.eyebrow { text-transform:uppercase; letter-spacing:.18em; font-size:12px; font-weight:700; color:#8de7d4; margin-bottom:10px; }
.hero h1 { color:#fff !important; font-size:clamp(30px,4vw,46px); margin:0 0 10px; }
.hero p { color:#dceaf3; font-size:16px; max-width:760px; margin:0; line-height:1.65; }
.pill { display:inline-block; margin-top:18px; padding:7px 12px; border:1px solid #ffffff36; border-radius:999px; color:#d8f9f1; font-size:12px; }
.section-label { color:#087da0; font-size:12px; text-transform:uppercase; letter-spacing:.16em; font-weight:800; margin:22px 0 12px; }
.steps { display:grid; grid-template-columns:repeat(6,1fr); gap:9px; margin:12px 0 24px; }
.step { border:1px solid #c8dce6; border-radius:12px; background:#fff; padding:12px 10px; color:#526579; font-size:12px; font-weight:600; min-height:70px; box-shadow:0 3px 12px #102a4310; }
.step b { display:inline-grid; place-items:center; border-radius:50%; width:23px; height:23px; background:#e5f3f7; color:#08759b; margin-right:6px; }
.step.active { border-color:#0785a7; background:#eaf6fa; color:#17345a; }
.step.done b { background:#087da0; color:white; }
.panel { background:white; border:1px solid #dce9ef; border-radius:18px; padding:18px 20px; box-shadow:0 8px 24px #17324a0c; }
.mini { color:var(--muted); font-size:13px; }
@media(max-width:850px){ .steps{grid-template-columns:repeat(2,1fr)} .hero{padding:24px} }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def get_model():
    return load_model()


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def analyse(data: bytes, name: str) -> dict:
    """Validate -> preprocess -> predict. Cached in session by file hash."""
    key = hashlib.sha256(data).hexdigest()
    cache = st.session_state.setdefault("analyses", {})
    if key in cache:
        return cache[key]
    img = validate_upload(data, name)
    pre = preprocess(img)
    model = get_model()
    result = {
        "original": np.asarray(img.convert("L")),
        "pre": pre,
        "scores": predict_all(model, pre.tensor),
        "heatmaps": {},
    }
    cache.clear()  # keep memory small: one image at a time
    cache[key] = result
    return result


# ------------------------------- Header ------------------------------------
st.markdown("""
<section class="hero">
  <div class="eyebrow">Computer vision · Explainable AI</div>
  <h1>AI-Powered Chest X-Ray Analysis</h1>
  <p>Explore how an image moves through validation, image preparation, AI scoring and visual explanation. Review the model’s output alongside the X-ray at every step.</p>
  <span class="pill">Research &amp; educational prototype · Human review required</span>
</section>
""", unsafe_allow_html=True)
st.markdown('<div class="section-label">Image → insight pipeline</div>', unsafe_allow_html=True)
st.markdown("""
<div class="steps">
  <div class="step"><b>1</b> Upload X-ray</div><div class="step"><b>2</b> Validate &amp; preprocess</div>
  <div class="step"><b>3</b> Resize &amp; normalize</div><div class="step"><b>4</b> DenseNet121 features</div>
  <div class="step"><b>5</b> Prediction + Grad-CAM</div><div class="step"><b>6</b> Review results</div>
</div>
""", unsafe_allow_html=True)
st.warning(config.DISCLAIMER, icon="⚠️")

# ------------------------------- Sidebar -----------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "DenseNet121 (TorchXRayVision, transfer learning) predicts findings; "
        "Grad-CAM highlights the regions that influenced a prediction."
    )
    st.markdown("**Privacy:** images are processed in memory and not stored. "
                "Please upload only public or anonymised images.")
    show_all = st.checkbox("Show all 18 model outputs", value=False)
    st.divider()
    st.caption("Research / educational use only.")

# ------------------------------- Input -------------------------------------
left, right = st.columns([2, 1])
with left:
    upload = st.file_uploader("Upload a chest X-ray (PNG / JPG)", type=["png", "jpg", "jpeg"])
with right:
    samples = sorted(config.SAMPLE_DIR.glob("*.png")) + sorted(config.SAMPLE_DIR.glob("*.jp*g"))
    sample_choice = None
    if samples:
        st.write("Or try a public sample:")
        sample_choice = st.selectbox("Sample image", [None] + samples,
                                     format_func=lambda p: "—" if p is None else p.name,
                                     label_visibility="collapsed")

data, name = None, None
if upload is not None:
    data, name = upload.getvalue(), upload.name
elif sample_choice is not None:
    data, name = sample_choice.read_bytes(), sample_choice.name

if data is None:
    st.markdown('<div class="panel"><b>Start an analysis</b><p class="mini">Upload a PNG or JPG chest X-ray, or choose the included public sample. The app validates the image before processing.</p></div>', unsafe_allow_html=True)
    st.stop()

# ------------------------------- Analysis ----------------------------------
try:
    with st.spinner("Validating, preprocessing and analysing..."):
        res = analyse(data, name)
except InvalidImageError as e:
    st.error(str(e))
    st.stop()

st.markdown('<div class="section-label">01 · Image validation &amp; preprocessing</div>', unsafe_allow_html=True)
st.markdown("The image is centre-cropped, converted to grayscale, normalized for the model, and resized to **224 × 224**.")
c1, c2 = st.columns(2)
c1.image(res["original"], caption="01 · Original X-ray", use_container_width=True, clamp=True)
c2.image(res["pre"].display, caption="02 · Preprocessed model input · 224 × 224", use_container_width=True)

st.markdown('<div class="section-label">02 · AI prediction</div>', unsafe_allow_html=True)
st.markdown("DenseNet121 extracts image features and returns a separate model score for each finding.")
scores = res["scores"]
rows = [{"Finding": config.DISPLAY_NAMES.get(t, t), "Score": scores[t], "Level": band(scores[t])}
        for t in config.TARGETS]
df = pd.DataFrame(rows)
s1, s2 = st.columns([3, 2])
s1.bar_chart(pd.Series({t: scores[t] for t in config.TARGETS}, name="Score"), horizontal=True, height=280)
s2.dataframe(df.style.format({"Score": "{:.2f}"}), hide_index=True, use_container_width=True)
st.caption("Level bands (Low <0.30, Moderate <0.60, High) are for display only, not clinical thresholds.")

if show_all:
    with st.expander("All model outputs", expanded=True):
        all_df = (pd.DataFrame({"Pathology": list(scores), "Score": list(scores.values())})
                  .sort_values("Score", ascending=False))
        st.dataframe(all_df.style.format({"Score": "{:.2f}"}), hide_index=True, use_container_width=True)

st.markdown('<div class="section-label">03 · Visual explanation · Grad-CAM</div>', unsafe_allow_html=True)
st.markdown("Choose a finding to view the image regions that most influenced its model score.")
g1, g2 = st.columns([1, 2])
finding = g1.selectbox("Finding to explain", config.TARGETS,
                       format_func=lambda t: config.DISPLAY_NAMES.get(t, t))
alpha = g1.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)
g1.caption("Warm colours = regions that pushed the score up. This shows where the "
           "model looked, not proof that it is medically correct.")

if finding not in res["heatmaps"]:
    with st.spinner("Generating heatmap..."):
        res["heatmaps"][finding] = compute_heatmap(get_model(), res["pre"].tensor, finding)
heat = res["heatmaps"][finding]
ov = overlay(res["pre"].display, heat, alpha)
if not has_signal(heat):
    g1.info("No region pushed this finding's score up, so the heatmap is empty. "
            "This is normal for low-scoring findings.")
g2.image(ov, caption=f"Grad-CAM: {config.DISPLAY_NAMES.get(finding, finding)} "
                     f"(score {scores[finding]:.2f})", width=460)
g1.download_button("Download heatmap PNG", png_bytes(ov),
                   file_name=f"gradcam_{finding.lower()}.png", mime="image/png")

st.divider()
st.caption(config.DISCLAIMER)
