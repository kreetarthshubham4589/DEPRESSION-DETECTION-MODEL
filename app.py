# =============================================
# PRODUCTION-LEVEL STREAMLIT APP
# =============================================

# Run using:
# streamlit run app.py

from pathlib import Path
import re
import sys
import time
import types
from enum import Enum
from importlib.machinery import ModuleSpec

import joblib
import streamlit as st
import torch

# Minimal torchvision stub so text-only transformers models do not import
# a broken global torchvision installation.
if "torchvision" not in sys.modules:
    torchvision_stub = types.ModuleType("torchvision")
    torchvision_stub.__spec__ = ModuleSpec("torchvision", loader=None)

    torchvision_transforms_stub = types.ModuleType("torchvision.transforms")
    torchvision_transforms_stub.__spec__ = ModuleSpec("torchvision.transforms", loader=None)

    torchvision_io_stub = types.ModuleType("torchvision.io")
    torchvision_io_stub.__spec__ = ModuleSpec("torchvision.io", loader=None)

    class InterpolationMode(Enum):
        NEAREST = 0
        NEAREST_EXACT = 1
        BILINEAR = 2
        BICUBIC = 3
        BOX = 4
        HAMMING = 5
        LANCZOS = 6

    torchvision_transforms_stub.InterpolationMode = InterpolationMode
    torchvision_stub.transforms = torchvision_transforms_stub
    torchvision_stub.io = torchvision_io_stub

    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = torchvision_transforms_stub
    sys.modules["torchvision.io"] = torchvision_io_stub

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# -----------------------------
# PATHS / CONSTANTS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ML_MODEL_PATH = BASE_DIR / "depression_model.pkl"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
TRANSFORMER_MODEL_PATH = BASE_DIR / "distilbert_depression_model"
MAX_LENGTH = 128

PRECAUTIONS = [
    "Talk to a trusted friend, family member, teacher, or colleague today.",
    "Consider booking an appointment with a licensed mental health professional.",
    "Try not to stay isolated for long periods and ask someone to check in on you.",
    "Avoid alcohol or recreational drugs, which can make low mood or distress worse.",
    "If your thoughts start to feel unsafe or overwhelming, seek urgent help immediately.",
]

INDIA_SUPPORT_CONTACTS = [
    (
        "Tele-MANAS (Government of India, 24/7)",
        "14416",
        "National tele-mental health support line.",
    ),
    (
        "Vandrevala Foundation (24/7 call or WhatsApp)",
        "+91 9999 666 555",
        "Free mental health counselling support across India.",
    ),
    (
        "iCALL by TISS (Mon-Sat, 10 AM-8 PM)",
        "9152987821",
        "Telephone counselling and emotional support.",
    ),
    (
        "AASRA",
        "022-27546667",
        "Emotional support and crisis help contact.",
    ),
]

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Depression Detection AI",
    layout="centered",
)


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# -----------------------------
# CACHED MODEL LOADING
# -----------------------------
@st.cache_resource
def load_models():
    ml_model = joblib.load(ML_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    device = torch.device("cpu")
    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        str(TRANSFORMER_MODEL_PATH),
        local_files_only=True,
    )
    transformer_tokenizer = AutoTokenizer.from_pretrained(
        str(TRANSFORMER_MODEL_PATH),
        local_files_only=True,
    )
    transformer_model.to(device)
    transformer_model.eval()

    return ml_model, vectorizer, transformer_model, transformer_tokenizer, device


try:
    ml_model, vectorizer, transformer_model, transformer_tokenizer, device = load_models()
except Exception as exc:
    st.error(f"Failed to load model artifacts: {exc}")
    st.stop()


# -----------------------------
# PREDICTION FUNCTIONS
# -----------------------------
def predict_ml(text: str):
    cleaned_text = preprocess(text)
    vec = vectorizer.transform([cleaned_text])
    pred = int(ml_model.predict(vec)[0])
    prob = ml_model.predict_proba(vec)[0]
    return pred, prob


def predict_transformer(text: str):
    inputs = transformer_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = transformer_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    return pred, probs


def render_precautions():
    st.subheader("Suggested Precautions")
    st.caption("This score is not a diagnosis, but it may be a good time to take supportive next steps.")
    for precaution in PRECAUTIONS:
        st.markdown(f"- {precaution}")


def render_india_support_contacts():
    st.subheader("India-Based Support Contacts")
    st.caption("These are public mental health support contacts in India that may help you reach qualified support faster.")
    for name, number, details in INDIA_SUPPORT_CONTACTS:
        st.markdown(f"- **{name}:** {number}  \n  {details}")
    st.error(
        "If you feel at immediate risk, have thoughts of self-harm, or feel unsafe, call 112 in India right now or go to the nearest emergency department."
    )

# -----------------------------
# UI HEADER
# -----------------------------
st.title("Depression Detection AI")
st.markdown("### Analyze mental health signals using Machine Learning and DistilBERT")
st.caption("This tool is for educational use only and is not a medical diagnosis.")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["ML (Fast)", "Transformer (Accurate)"],
)
show_details = st.sidebar.checkbox("Show Detailed Analysis")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.markdown("### Enter Text")
user_input = st.text_area("Type here...", height=150)

# -----------------------------
# ANALYZE BUTTON
# -----------------------------
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.6)

            if model_choice == "ML (Fast)":
                pred, prob = predict_ml(user_input)
            else:
                pred, prob = predict_transformer(user_input)

        st.markdown("## Result")
        depression_confidence = float(prob[1])

        if pred == 1:
            st.error("Signs of depression detected.")
        else:
            st.success("No depression detected.")

        st.markdown("### Confidence")
        st.progress(depression_confidence)

        col1, col2 = st.columns(2)
        col1.metric("Not Depressed", f"{prob[0] * 100:.2f}%")
        col2.metric("Depressed", f"{depression_confidence * 100:.2f}%")

        if depression_confidence >= 0.80:
            st.markdown("---")
            st.warning(
                "The model's depression-confidence score is above 80%. Please consider reaching out for support soon."
            )
            render_precautions()
            render_india_support_contacts()
        elif depression_confidence >= 0.50:
            st.markdown("---")
            st.info(
                "The model's depression-confidence score is above 50%. A few supportive precautions may be helpful."
            )
            render_precautions()

        if show_details:
            st.markdown("---")
            st.subheader("Model Insights")
            st.write("Model Used:", model_choice)
            st.write("Text Length:", len(user_input.split()))
            st.write("Preprocessed Length:", len(preprocess(user_input).split()))

            if pred == 1:
                st.info(
                    "The model detected language patterns often associated with distress, sadness, or hopelessness."
                )
            else:
                st.info("The model did not detect strong depressive signals in this text.")

# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built for production with Streamlit, classic ML, and a transformer classifier.")

# =============================================
# END
# =============================================
