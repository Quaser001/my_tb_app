# ==============================
# Multi-Modal TB Detection App
# ==============================

import streamlit as st
import torch
import torchaudio
from torchvision import transforms, models
from PIL import Image
import numpy as np
import time
import os

# ----------------------------
# Device setup
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Trained Models
# ----------------------------
@st.cache_resource
def load_models():
    # Load X-ray model
    xray_model = models.resnet18(weights=None)
    xray_model.fc = torch.nn.Linear(xray_model.fc.in_features, 2)
    xray_model.load_state_dict(torch.load("models/xray_model.pth", map_location=DEVICE))
    xray_model.to(DEVICE).eval()

    # Load Audio model
    audio_model = torch.load("models/audio_model.pth", map_location=DEVICE)
    audio_model.to(DEVICE).eval()

    return xray_model, audio_model

xray_model, audio_model = load_models()

# ----------------------------
# X-ray Preprocessing
# ----------------------------
xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_xray(image: Image.Image):
    img = xray_transform(image).unsqueeze(0).to(DEVICE)
    with st.spinner("Running X-ray Model..."):
        time.sleep(0.5)
        with torch.no_grad():
            probs = xray_model(img)
            probs = torch.softmax(probs, dim=1)
    return probs.cpu().numpy().flatten()

# ----------------------------
# Audio Preprocessing
# ----------------------------
def preprocess_audio(file):
    # Load audio
    wav, sr = torchaudio.load(file)

    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        wav = resampler(wav)

    # Mel Spectrogram
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_mels=128
    )
    mel_spec = mel_spec_transform(wav)  # [1, 128, time]

    # Average along time for linear classifier
    mel_spec = mel_spec.mean(dim=2)  # [1, 128]
    return mel_spec.to(DEVICE)

def predict_audio(file):
    mel_spec = preprocess_audio(file)
    with st.spinner("Running Audio Model..."):
        time.sleep(0.5)
        with torch.no_grad():
            probs = audio_model(mel_spec.float())
            probs = torch.softmax(probs, dim=1)
    return probs.cpu().numpy().flatten()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Multi-Modal TB Detection", layout="centered")
st.title("🩺 Multi-Modal TB Detection")
st.write("Upload a chest X-ray image or cough/breath audio to detect TB.")

# File upload
xray_file = st.file_uploader("Upload Chest X-ray Image", type=["png", "jpg", "jpeg"])
audio_file = st.file_uploader("Upload Cough / Breath Audio", type=["wav", "mp3"])

combined_probs = []

# ----------------------------
# Run Audio Model
# ----------------------------
if audio_file:
    st.info("Processing audio...")
    audio_probs = predict_audio(audio_file)
    combined_probs.append(audio_probs)
    
    audio_prob_dict = {
        "Normal": f"{audio_probs[0]*100:.2f}%",
        "TB": f"{audio_probs[1]*100:.2f}%"
    }
    st.write("**Audio Model Probabilities:**", audio_prob_dict)

# ----------------------------
# Run X-ray Model
# ----------------------------
if xray_file:
    st.info("Processing X-ray...")
    image = Image.open(xray_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    xray_probs = predict_xray(image)
    combined_probs.append(xray_probs)
    
    xray_prob_dict = {
        "Normal": f"{xray_probs[0]*100:.2f}%",
        "TB": f"{xray_probs[1]*100:.2f}%"
    }
    st.write("**X-ray Model Probabilities:**", xray_prob_dict)

# ----------------------------
# Combined Feedback
# ----------------------------
if combined_probs:
    combined_probs = np.mean(np.array(combined_probs), axis=0)
    combined_dict = {
        "Normal": f"{combined_probs[0]*100:.2f}%",
        "TB": f"{combined_probs[1]*100:.2f}%"
    }
    st.success("**Combined Model Probabilities:**")
    st.write(combined_dict)
