
import streamlit as st
import torch
import torchaudio
import torchvision
from torchvision import transforms, models
import torch.nn as nn
import numpy as np
from PIL import Image


# Fix for Python 3.13 where audioop is missing
import sys
try:
    import audioop  # noqa
except ImportError:
    sys.modules["audioop"] = None  # Prevent pydub import error


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
import torchaudio
from pydub import AudioSegment
import numpy as np
import torch

# Try to set soundfile backend for torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except Exception as e:
    print("Warning: could not set torchaudio backend to soundfile:", e)

def preprocess_audio(wav_path, target_sr=16000):
    """
    Load and preprocess audio.
    Tries torchaudio first, falls back to pydub if torchaudio fails.
    """
    try:
        # --- Try torchaudio ---
        wav, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav, target_sr
    except Exception as e:
        print("torchaudio.load failed, falling back to pydub. Error:", e)

        # --- Fallback to pydub ---
        audio = AudioSegment.from_file(wav_path)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
        wav = torch.tensor(samples).unsqueeze(0)  # shape (1, N)
        return wav, target_sr


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
def predict_audio(wav_path):
    spec = preprocess_audio(wav_path)
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
# Streamlit UI
# ==============================
st.title("Tuberculosis Detection App 🩺")
st.write("Upload **Audio (Cough)** or **X-ray** or both. The model predicts TB probability.")

audio_file = st.file_uploader("Upload Cough Audio (.wav)", type=["wav"])
xray_file = st.file_uploader("Upload Chest X-ray Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])

if st.button("Predict"):
    if not audio_file and not xray_file:
        st.warning("Please upload at least one file!")
    else:
        combined_probs = []
        if audio_file:
            st.info("Running Audio Model...")
            audio_probs = predict_audio(audio_file)
            st.write(f"Audio Model Probabilities: Normal: {audio_probs[0]*100:.2f}%, TB: {audio_probs[1]*100:.2f}%")
            combined_probs.append(audio_probs)
        if xray_file:
            st.info("Running X-ray Model...")
            xray_probs = predict_xray(xray_file)
            st.write(f"X-ray Model Probabilities: Normal: {xray_probs[0]*100:.2f}%, TB: {xray_probs[1]*100:.2f}%")
            combined_probs.append(xray_probs)
        if len(combined_probs) == 2:
            avg_probs = np.mean(combined_probs, axis=0)
            st.success(f"Combined Prediction: Normal: {avg_probs[0]*100:.2f}%, TB: {avg_probs[1]*100:.2f}%")
        elif len(combined_probs) == 1:
            st.success(f"Prediction: Normal: {combined_probs[0][0]*100:.2f}%, TB: {combined_probs[0][1]*100:.2f}%")
