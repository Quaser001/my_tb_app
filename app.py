# =============================
# Multi-Modal TB Detection App
# =============================

import os
import torch
import torchaudio
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import streamlit as st

# ----------------------------
# Device Setup
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Paths
# ----------------------------
MODEL_DIR = os.path.join(os.getcwd(), "models")
XRAY_MODEL_PATH = os.path.join(MODEL_DIR, "xray_model.pth")
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, "audio_model.pth")

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # X-ray Model
    if not os.path.exists(XRAY_MODEL_PATH):
        st.error(f"X-ray model not found at {XRAY_MODEL_PATH}")
        st.stop()
    xray_model = models.resnet18(weights=None)
    xray_model.fc = torch.nn.Linear(xray_model.fc.in_features, 2)
    xray_model.load_state_dict(torch.load(XRAY_MODEL_PATH, map_location=DEVICE))
    xray_model.to(DEVICE).eval()

    # Audio Model
    if not os.path.exists(AUDIO_MODEL_PATH):
        st.error(f"Audio model not found at {AUDIO_MODEL_PATH}")
        st.stop()
    audio_model = torch.load(AUDIO_MODEL_PATH, map_location=DEVICE)
    audio_model.to(DEVICE).eval()

    return xray_model, audio_model

xray_model, audio_model = load_models()

# ----------------------------
# X-ray Preprocessing
# ----------------------------
xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_xray(image):
    image = xray_transform(image).unsqueeze(0).to(DEVICE)
    with st.spinner("Running X-ray model..."):
        with torch.no_grad():
            probs = torch.softmax(xray_model(image), dim=1)
    return probs.cpu().numpy().flatten()

# ----------------------------
# Audio Preprocessing
# ----------------------------
SAMPLE_RATE = 16000  # adjust according to your training

def predict_audio(audio_file):
    waveform, sr = torchaudio.load(audio_file)
    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Compute Mel Spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_mels=64
    )(waveform)
    mel_spec = mel_spec.unsqueeze(0).to(DEVICE)  # Add batch dim

    with st.spinner("Running audio model..."):
        with torch.no_grad():
            probs = torch.softmax(audio_model(mel_spec.float()), dim=1)
    return probs.cpu().numpy().flatten()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🩺 Multi-Modal TB Detection")
st.write("Upload an X-ray image and/or an audio file to detect Tuberculosis.")

xray_file = st.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3", "m4a"])

combined_probs = []

if st.button("Run Prediction"):
    if not xray_file and not audio_file:
        st.warning("Please upload at least one file!")
        st.stop()

    # X-ray Prediction
    if xray_file:
        image = Image.open(xray_file).convert("RGB")
        xray_probs = predict_xray(image)
        combined_probs.append(xray_probs)
        xray_prob_dict = {"Normal": f"{xray_probs[0]*100:.2f}%",
                          "TB": f"{xray_probs[1]*100:.2f}%"}
        st.write("**X-ray Model Probabilities:**", xray_prob_dict)

    # Audio Prediction
    if audio_file:
        audio_probs = predict_audio(audio_file)
        combined_probs.append(audio_probs)
        audio_prob_dict = {"Normal": f"{audio_probs[0]*100:.2f}%",
                           "TB": f"{audio_probs[1]*100:.2f}%"}
        st.write("**Audio Model Probabilities:**", audio_prob_dict)

    # Combined Prediction (average)
    if combined_probs:
        combined = np.mean(combined_probs, axis=0)
        combined_dict = {"Normal": f"{combined[0]*100:.2f}%",
                         "TB": f"{combined[1]*100:.2f}%"}
        st.write("**Combined Probabilities:**", combined_dict)
        st.success(f"Predicted: {'TB' if combined[1] > combined[0] else 'Normal'}")
