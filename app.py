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
    spec = spec.unsqueeze(0).to(device)  # Add batch dimension
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
# UI Enhancements - Dark Cozy Professional
# ==============================
st.markdown("""
<style>
/* -------------------------- Background -------------------------- */
.stApp {
    background-color: #1A1A2E;
    color: #E0E0E0;
    font-family: 'Quicksand', sans-serif;
}

/* -------------------------- Cards (glassmorphism effect) -------------------------- */
.card {
    background: rgba(30, 30, 60, 0.85);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
    transition: all 0.3s ease-in-out;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.7);
}

/* -------------------------- Headers -------------------------- */
h1 {
    font-family: 'Poppins', sans-serif;
    color: #FFD369; 
    text-align: center;
    font-weight: 700;
    letter-spacing: 1px;
}
h2, h3 {
    font-family: 'Quicksand', sans-serif;
    color: #FF7E5F;
}

/* -------------------------- Prediction Text -------------------------- */
.normal { color: #38BDF8; font-weight: 700; font-size: 18px; }
.tb { color: #FF6B6B; font-weight: 700; font-size: 18px; }
.info-text { color: #FFD369; font-size:16px; font-weight:500; }

/* -------------------------- Buttons -------------------------- */
button {
    background: linear-gradient(135deg, #FF7E5F, #FFD369);
    color: #1A1A2E;
    font-weight: bold;
    border-radius: 15px;
    padding: 10px 30px;
    transition: 0.3s ease;
}
button:hover {
    background: linear-gradient(135deg, #FFD369, #FF7E5F);
    transform: scale(1.05);
}

/* -------------------------- Progress Bar -------------------------- */
.stProgress > div > div > div > div {
    background: linear-gradient(270deg, #38BDF8, #FF7E5F, #FFD369);
    border-radius: 15px;
    opacity: 0.9;
    transition: width 0.6s ease-in-out;
}

/* -------------------------- File uploader & layout tweaks -------------------------- */
.css-1aumxhk {
    font-size: 16px;
    font-weight: 500;
    color: #E0E0E0;
}
.css-1y0tads {
    border-radius: 15px;
    background: rgba(40, 40, 70, 0.85);
    padding: 10px;
    color: #E0E0E0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header & Subtext
# -------------------------------
st.markdown("<h1>🩺 TB Detection Magic</h1>", unsafe_allow_html=True)
st.markdown("<p class='info-text' style='text-align:center;'>Upload your cough audio or chest X-ray 🩻. The model will do its magic! ✨</p>", unsafe_allow_html=True)
st.write("---")

# ==============================
# Streamlit File Uploaders
# ==============================
audio_file = st.file_uploader("Upload Cough Audio (.wav, .mp3, .flac, .ogg)", 
                              type=["wav", "mp3", "flac", "ogg"])
xray_file = st.file_uploader("Upload Chest X-ray Image (.png, .jpg, .jpeg)", 
                             type=["png", "jpg", "jpeg"])

# ==============================
# Prediction and Animated Progress
# ==============================
if st.button("Predict"):
    if not audio_file and not xray_file:
        st.warning("Please upload at least one file!")
    else:
        combined_probs = []

        progress_text = "Processing..."
        my_bar = st.progress(0, text=progress_text)

        for i in range(101):
            time.sleep(0.01)
            my_bar.progress(i, text=progress_text)

        # Audio Prediction
        if audio_file:
            st.info("Running Audio Model...")
            audio_bytes = io.BytesIO(audio_file.read())
            audio_probs = predict_audio(audio_bytes)
            st.markdown(f"<div class='card'><p class='normal'>Audio Model Probabilities: Normal: {audio_probs[0]*100:.2f}%, TB: {audio_probs[1]*100:.2f}%</p></div>", unsafe_allow_html=True)
            combined_probs.append(audio_probs)

        # X-ray Prediction
        if xray_file:
            st.info("Running X-ray Model...")
            xray_probs = predict_xray(xray_file)
            st.markdown(f"<div class='card'><p class='tb'>X-ray Model Probabilities: Normal: {xray_probs[0]*100:.2f}%, TB: {xray_probs[1]*100:.2f}%</p></div>", unsafe_allow_html=True)
            combined_probs.append(xray_probs)

        # Combined Prediction
        if len(combined_probs) == 2:
            avg_probs = np.mean(combined_probs, axis=0)
            st.success(f"Combined Prediction: Normal: {avg_probs[0]*100:.2f}%, TB: {avg_probs[1]*100:.2f}%")
        elif len(combined_probs) == 1:
            st.success(f"Prediction: Normal: {combined_probs[0][0]*100:.2f}%, TB: {combined_probs[0][1]*100:.2f}%")
