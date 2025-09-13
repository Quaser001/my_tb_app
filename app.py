import streamlit as st
import torch
import torchaudio
import torchvision
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import time

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
# Audio Preprocessing (Multi-format)
# ==============================
def preprocess_audio_multi_format(file_like, sr=16000, n_mels=64, max_len=256):
    wav, original_sr = torchaudio.load(file_like)
    if original_sr != sr:
        wav = torchaudio.functional.resample(wav, original_sr, sr)
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=400, hop_length=160, n_mels=n_mels
    )
    db = torchaudio.transforms.AmplitudeToDB()
    spec = db(mel(wav))
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    if spec.shape[-1] < max_len:
        pad = max_len - spec.shape[-1]
        spec = torch.nn.functional.pad(spec, (0, pad))
    else:
        spec = spec[:, :, :max_len]
    spec = spec.unsqueeze(0).to(device)
    return spec

# ==============================
# X-ray Preprocessing
# ==============================
xray_transform = transforms.Compose([
    transforms.Resize((224, 224)),
])

def preprocess_xray(img_file):
    img = Image.open(img_file).convert("RGB")
    img = xray_transform(img)
    img = torchvision.transforms.functional.to_tensor(img).unsqueeze(0).to(device)
    return img

# ==============================
# Prediction Functions
# ==============================
def predict_audio(file_like):
    spec = preprocess_audio_multi_format(file_like)
    with torch.no_grad():
        out = audio_model(spec)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

def predict_xray(img_file):
    img = preprocess_xray(img_file)
    with torch.no_grad():
        out = xray_model(img)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

# ==============================
# Streamlit UI - Beautiful
# ==============================
st.set_page_config(
    page_title="Tuberculosis Detection App 🩺",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown("<h1 style='text-align: center; color: #2E3B55;'>Tuberculosis Detection App 🩺</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4F5D75;'>Upload <b>Cough Audio</b> or <b>X-ray</b> or both. The model predicts TB probability.</p>", unsafe_allow_html=True)
st.write("---")

audio_file = st.file_uploader("Upload Cough Audio (.wav, .mp3, .flac, .ogg)", 
                              type=["wav", "mp3", "flac", "ogg"])
xray_file = st.file_uploader("Upload Chest X-ray Image (.png, .jpg, .jpeg)", 
                             type=["png", "jpg", "jpeg"])

if st.button("Predict"):
    if not audio_file and not xray_file:
        st.warning("Please upload at least one file!")
    else:
        combined_probs = []

        # -----------------------------
        # Audio Prediction
        # -----------------------------
        if audio_file:
            st.info("Processing Audio Model...")
            audio_bytes = io.BytesIO(audio_file.read())
            with st.spinner("Running Audio Model..."):
                for i in range(100):
                    time.sleep(0.01)
                    st.progress(i + 1)
                audio_probs = predict_audio(audio_bytes)
            with st.container():
                st.markdown("### 🎤 Audio Model Results")
                st.success(f"Normal: {audio_probs[0]*100:.2f}%")
                st.error(f"TB: {audio_probs[1]*100:.2f}%")
            combined_probs.append(audio_probs)

        # -----------------------------
        # X-ray Prediction
        # -----------------------------
        if xray_file:
            st.info("Processing X-ray Model...")
            with st.spinner("Running X-ray Model..."):
                for i in range(100):
                    time.sleep(0.01)
                    st.progress(i + 1)
                xray_probs = predict_xray(xray_file)
            with st.container():
                st.markdown("### 🩻 X-ray Model Results")
                st.success(f"Normal: {xray_probs[0]*100:.2f}%")
                st.error(f"TB: {xray_probs[1]*100:.2f}%")
            combined_probs.append(xray_probs)

        # -----------------------------
        # Combined Prediction
        # -----------------------------
        if len(combined_probs) == 2:
            avg_probs = np.mean(combined_probs, axis=0)
            st.markdown("### 📊 Combined Prediction")
            st.success(f"Normal: {avg_probs[0]*100:.2f}%")
            st.error(f"TB: {avg_probs[1]*100:.2f}%")
        elif len(combined_probs) == 1:
            st.markdown("### 📊 Prediction")
            st.success(f"Normal: {combined_probs[0][0]*100:.2f}%")
            st.error(f"TB: {combined_probs[0][1]*100:.2f}%")
