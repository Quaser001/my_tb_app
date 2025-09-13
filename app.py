# =============================
# Tuberculosis Detection App
# =============================

import streamlit as st
import torch
import torchaudio
import torchvision
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from PIL import Image
from io import BytesIO

# ==============================
# Device
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# Load Audio Model
# ==============================
def build_resnet_audio(n_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)

audio_model = build_resnet_audio()
audio_model.load_state_dict(torch.load("audio_best.pt", map_location=device))
audio_model.eval()

# ==============================
# Load X-ray Model
# ==============================
def build_resnet_xray(n_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)

xray_model = build_resnet_xray()
xray_model.load_state_dict(torch.load("xray_best.pt", map_location=device))
xray_model.eval()

# ==============================
# Audio Preprocessing
# ==============================
def preprocess_audio(file_bytes, sr=16000, n_mels=64, max_len=256):
    wav, original_sr = torchaudio.load(BytesIO(file_bytes))
    if original_sr != sr:
        wav = torchaudio.functional.resample(wav, original_sr, sr)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # mono

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=400, hop_length=160, n_mels=n_mels
    )(wav)
    db = torchaudio.transforms.AmplitudeToDB()
    spec = db(mel)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)

    if spec.shape[-1] < max_len:
        pad = max_len - spec.shape[-1]
        spec = torch.nn.functional.pad(spec, (0, pad))
    else:
        spec = spec[:, :, :max_len]

    spec = spec.unsqueeze(0).to(device)  # add batch dim
    return spec

# ==============================
# X-ray Preprocessing
# ==============================
xray_transform = transforms.Compose([transforms.Resize((224, 224))])

def preprocess_xray(img_file):
    img = Image.open(img_file).convert("RGB")
    img = xray_transform(img)
    img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
    return img

# ==============================
# Prediction Functions
# ==============================
def predict_audio(file_bytes):
    spec = preprocess_audio(file_bytes)
    with st.spinner("Running Audio Model..."):
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = audio_model(spec)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

def predict_xray(img_file):
    img = preprocess_xray(img_file)
    with st.spinner("Running X-ray Model..."):
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = xray_model(img)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="TB Detection App", layout="wide", page_icon="🩺")
st.title("🩺 Tuberculosis Detection App")
st.write("Upload **Cough Audio** or **Chest X-ray Image** or both. The model predicts TB probability.")

# File uploaders
audio_file = st.file_uploader(
    "Upload Cough Audio", type=["wav", "mp3", "ogg", "flac"]
)
xray_file = st.file_uploader(
    "Upload Chest X-ray Image", type=["png", "jpg", "jpeg"]
)

if st.button("Predict"):
    if not audio_file and not xray_file:
        st.warning("Please upload at least one file!")
    else:
        combined_probs = []

        col1, col2 = st.columns(2)

        # Audio Prediction
        if audio_file:
            with col1:
                st.info("Running Audio Model...")
                file_bytes = audio_file.read()
                audio_probs = predict_audio(file_bytes)
                combined_probs.append(audio_probs)
                st.markdown(
                    f"<h4 style='color:blue;'>Audio Probabilities:</h4>"
                    f"Normal: {audio_probs[0]*100:.2f}%  |  TB: {audio_probs[1]*100:.2f}%",
                    unsafe_allow_html=True
                )

        # X-ray Prediction
        if xray_file:
            with col2:
                st.info("Running X-ray Model...")
                xray_probs = predict_xray(xray_file)
                combined_probs.append(xray_probs)
                st.markdown(
                    f"<h4 style='color:green;'>X-ray Probabilities:</h4>"
                    f"Normal: {xray_probs[0]*100:.2f}%  |  TB: {xray_probs[1]*100:.2f}%",
                    unsafe_allow_html=True
                )

        # Combined Prediction
        if combined_probs:
            avg_probs = np.mean(combined_probs, axis=0)
            st.markdown("---")
            st.success(
                f"**Combined Prediction:** Normal: {avg_probs[0]*100:.2f}%, TB: {avg_probs[1]*100:.2f}%"
            )

        # Progress bar
        import time
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.005)
            progress_bar.progress(i + 1)
