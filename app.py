
import streamlit as st
import torch
import torchaudio
import torchvision
from model_utils import build_resnet_audio, build_resnet_xray, AudioSpectrogramDataset, XrayDataset, device
from torchvision import transforms

# Load models
audio_model = build_resnet_audio()
audio_model.load_state_dict(torch.load("audio_best.pt", map_location=device))
audio_model.eval()

xray_model = build_resnet_xray()
xray_model.load_state_dict(torch.load("xray_best.pt", map_location=device))
xray_model.eval()

st.title("Tuberculosis Detection: Audio & X-ray")

audio_file = st.file_uploader("Upload Cough Audio (.wav)", type=["wav"])
xray_file = st.file_uploader("Upload X-ray Image (.png, .jpg, .jpeg)", type=["png","jpg","jpeg"])

def predict_audio(file):
    wav, sr = torchaudio.load(file)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=400, hop_length=160, n_mels=64)(wav)
    db = torchaudio.transforms.AmplitudeToDB()(mel)
    db = (db - db.mean()) / (db.std() + 1e-6)
    max_len = 256
    if db.shape[-1] < max_len:
        pad = max_len - db.shape[-1]
        db = torch.nn.functional.pad(db, (0, pad))
    else:
        db = db[:, :, :max_len]
    db = db.unsqueeze(0).to(device)
    with torch.no_grad():
        out = audio_model(db)
        prob = torch.softmax(out, dim=1)[0,1].item()
    return prob

def predict_xray(file):
    img = torchvision.io.read_image(file).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3,1,1)
    transform = transforms.Compose([transforms.Resize((224,224))])
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = xray_model(img)
        prob = torch.softmax(out, dim=1)[0,1].item()
    return prob

if st.button("Predict"):
    results = []
    if audio_file:
        audio_prob = predict_audio(audio_file)
        st.write(f"Audio TB Probability: {audio_prob*100:.2f}%")
        results.append(audio_prob)
    if xray_file:
        xray_prob = predict_xray(xray_file)
        st.write(f"X-ray TB Probability: {xray_prob*100:.2f}%")
        results.append(xray_prob)
    if results:
        combined_prob = sum(results)/len(results)
        st.success(f"Combined TB Probability: {combined_prob*100:.2f}%")
