import streamlit as st
import torch
import torchaudio
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np

# ----------------------------
# Device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load Models
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_audio_model():
    # Placeholder: replace with your actual model loading
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 2),  # Example
        torch.nn.Softmax(dim=1)
    )
    model.eval()
    return model.to(DEVICE)

@st.cache_resource(show_spinner=False)
def load_xray_model():
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)  # 2 classes: Normal/TB
    model.eval()
    return model.to(DEVICE)

audio_model = load_audio_model()
xray_model = load_xray_model()

# ----------------------------
# Class Names
# ----------------------------
audio_classes = ["Normal", "TB"]
xray_classes = ["Normal", "TB"]

# ----------------------------
# Audio Preprocessing
# ----------------------------
def preprocess_audio(file):
    waveform, sr = torchaudio.load(file)
    # Example: compute Mel Spectrogram (modify according to your trained model)
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
    mel_spec = mel_spec.mean(dim=0)  # simple average over channels
    mel_spec = mel_spec.unsqueeze(0).to(DEVICE)  # add batch dim
    return mel_spec

# ----------------------------
# Audio Prediction
# ----------------------------
def predict_audio(file):
    mel_spec = preprocess_audio(file)
    with torch.no_grad():
        probs = audio_model(mel_spec.float())
    return probs.cpu().numpy().flatten()

# ----------------------------
# X-ray Preprocessing
# ----------------------------
def preprocess_xray(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)
    return img

# ----------------------------
# X-ray Prediction
# ----------------------------
def predict_xray(image):
    img_tensor = preprocess_xray(image)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(xray_model(img_tensor), dim=1)
    return probs.cpu().numpy().flatten()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("Multi-Modal TB Detection")

st.sidebar.header("Upload Inputs")
audio_file = st.sidebar.file_uploader("Upload Audio (WAV)", type=["wav"])
xray_file = st.sidebar.file_uploader("Upload X-ray Image", type=["png", "jpg", "jpeg"])

combined_probs = []

if audio_file or xray_file:
    st.subheader("Multi-Modal Predictions")

    # Audio Prediction
    if audio_file:
        st.info("Running Audio Model...")
        audio_probs = predict_audio(audio_file)
        combined_probs.append(audio_probs)
        audio_prob_dict = {name: f"{prob*100:.2f}%" for name, prob in zip(audio_classes, audio_probs)}
        st.write("Audio Model Probabilities:", audio_prob_dict)

    # X-ray Prediction
    if xray_file:
        st.info("Running X-ray Model...")
        xray_probs = predict_xray(xray_file)
        combined_probs.append(xray_probs)
        xray_prob_dict = {name: f"{prob*100:.2f}%" for name, prob in zip(xray_classes, xray_probs)}
        st.write("X-ray Model Probabilities:", xray_prob_dict)

    # Combined (Average) Probabilities
    if combined_probs:
        combined_arr = np.mean(np.array(combined_probs), axis=0)
        combined_dict = {name: f"{prob*100:.2f}%" for name, prob in zip(audio_classes, combined_arr)}
        st.subheader("Combined Probabilities")
        st.write(combined_dict)

st.info("Upload audio or X-ray to get TB prediction.")

