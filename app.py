# ==============================
# Streamlit UI (modified for multiple audio types)
# ==============================
st.title("Tuberculosis Detection App 🩺")
st.write("Upload **Audio (Cough)** or **X-ray** or both. The model predicts TB probability.")

# Allow commonly used audio types
audio_file = st.file_uploader(
    "Upload Cough Audio", 
    type=["wav", "mp3", "m4a", "flac", "ogg"]
)
xray_file = st.file_uploader(
    "Upload Chest X-ray Image (.png, .jpg, .jpeg)", 
    type=["png", "jpg", "jpeg"]
)

# Modified preprocess_audio to handle in-memory files
def preprocess_audio_file(uploaded_file, sr=16000, n_mels=64, max_len=256):
    # Load file as tensor (torchaudio supports many formats)
    wav, original_sr = torchaudio.load(uploaded_file)
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

# Update prediction function to use the in-memory version
def predict_audio(uploaded_file):
    spec = preprocess_audio_file(uploaded_file)
    with st.spinner("Running Audio Model..."):
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = audio_model(spec)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

# Prediction button remains unchanged
if st.button("Predict"):
    if not audio_file and not xray_file:
        st.warning("Please upload at least one file!")
    else:
        combined_probs = []

        if audio_file:
            audio_probs = predict_audio(audio_file)
            st.write(f"Audio Model Probabilities: Normal: {audio_probs[0]*100:.2f}%, TB: {audio_probs[1]*100:.2f}%")
            combined_probs.append(audio_probs)

        if xray_file:
            xray_probs = predict_xray(xray_file)
            st.write(f"X-ray Model Probabilities: Normal: {xray_probs[0]*100:.2f}%, TB: {xray_probs[1]*100:.2f}%")
            combined_probs.append(xray_probs)

        # Combined Prediction
        if combined_probs:
            avg_probs = np.mean(combined_probs, axis=0)
            st.success(f"Combined Prediction: Normal: {avg_probs[0]*100:.2f}%, TB: {avg_probs[1]*100:.2f}%")
