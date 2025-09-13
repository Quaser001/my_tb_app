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
# Streamlit UI - Designer Upgrade
# ==============================
st.set_page_config(
    page_title="Tuberculosis Detection App 🩺",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Custom CSS for Modern UI
# -------------------------------
st.markdown("""
<style>
/* Import cute modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&family=Quicksand:wght@400;600&family=Montserrat:wght@400;700&display=swap');

/* App Background Gradient */
.stApp {
    background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
    background-attachment: fixed;
    font-family: 'Quicksand', sans-serif;
}

/* Card Style - Frosted Glass Effect */
.card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 25px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.2);
    backdrop-filter: blur(10px);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 60px rgba(0,0,0,0.3);
}

/* Header */
h1 {
    font-family: 'Poppins', sans-serif;
    color: #4B0082;
    text-align: center;
    font-weight: 700;
    font-size: 3rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #334155;
    font-size: 1.2rem;
    margin-bottom: 30px;
}

/* Prediction Colors */
.normal { color: #10B981; font-weight: 600; font-size:18px; }
.tb { color: #EF4444; font-weight: 600; font-size:18px; }

/* Buttons */
button {
    border-radius: 12px;
    font-family: 'Montserrat', sans-serif;
    font-weight: 600;
}
button:hover {
    background-color: #7C3AED !important;
    color: white !important;
}

/* Animated Gradient Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(270deg, #FF6A88, #FF99AC, #B5FFFC, #FFDEE9);
    background-size: 600% 600%;
    animation: gradientAnimation 3s ease infinite;
    border-radius: 12px;
}

/* Gradient Animation */
@keyframes gradientAnimation {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# App Header
# -------------------------------
st.markdown("<h1>🩺 Tuberculosis Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload Cough Audio or Chest X-ray or both. Get TB prediction instantly.</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------------
# File Uploaders
# -------------------------------
audio_file = st.file_uploader("🎤 Upload Cough Audio (.wav, .mp3, .flac, .ogg)", 
                              type=["wav", "mp3", "flac", "ogg"])
xray_file = st.file_uploader("🩻 Upload Chest X-ray Image (.png, .jpg, .jpeg)", 
                             type=["png", "jpg", "jpeg"])

# -------------------------------
# Smooth Animated Progress Bar
# -------------------------------
def animated_progress(duration=2.0):
    progress_bar = st.progress(0)
    steps = 100
    interval = duration / steps
    for i in range(steps + 1):
        progress_bar.progress(i)
        time.sleep(interval)
    progress_bar.empty()

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict"):
    if not audio_file and not xray_file:
        st.warning("Please upload at least one file!")
    else:
        combined_probs = []

        # Audio Prediction
        if audio_file:
            st.info("Processing Audio Model...")
            audio_bytes = io.BytesIO(audio_file.read())
            with st.spinner("Running Audio Model..."):
                animated_progress(duration=1.5)
                audio_probs = predict_audio(audio_bytes)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🎤 Audio Model Results")
            st.markdown(f"<p class='normal'>Normal: {audio_probs[0]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='tb'>TB: {audio_probs[1]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            combined_probs.append(audio_probs)

        # X-ray Prediction
        if xray_file:
            st.info("Processing X-ray Model...")
            with st.spinner("Running X-ray Model..."):
                animated_progress(duration=1.5)
                xray_probs = predict_xray(xray_file)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🩻 X-ray Model Results")
            st.markdown(f"<p class='normal'>Normal: {xray_probs[0]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='tb'>TB: {xray_probs[1]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            combined_probs.append(xray_probs)

        # Combined Prediction
        if len(combined_probs) == 2:
            avg_probs = np.mean(combined_probs, axis=0)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Combined Prediction")
            st.markdown(f"<p class='normal'>Normal: {avg_probs[0]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='tb'>TB: {avg_probs[1]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        elif len(combined_probs) == 1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 📊 Prediction")
            st.markdown(f"<p class='normal'>Normal: {combined_probs[0][0]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='tb'>TB: {combined_probs[0][1]*100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
