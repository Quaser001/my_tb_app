# ==============================
# Streamlit UI - Ultra-Cute, Modern, Dynamic
# ==============================
st.set_page_config(
    page_title="Tuberculosis Detection App 🩺",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Expert UI CSS - Fonts, Colors, Dynamic
# -------------------------------
st.markdown("""
<style>
/* ---------------- Background Gradient ---------------- */
.stApp {
    background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 50%, #F9FFB5 100%);
    background-attachment: fixed;
    font-family: 'Comic Neue', 'Baloo 2', 'Segoe UI', sans-serif;
}

/* ---------------- Cards ---------------- */
.card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 25px;
    padding: 30px;
    margin-bottom: 20px;
    box-shadow: 0 15px 40px rgba(0,0,0,0.25);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.3);
}

/* ---------------- Headers ---------------- */
h1 {
    font-family: 'Baloo 2', cursive;
    color: #ff4d6d;
    text-align: center;
    text-shadow: 1px 1px 5px rgba(0,0,0,0.2);
}
h2, h3 {
    font-family: 'Baloo 2', cursive;
    color: #1e3a8a;
}

/* ---------------- Prediction Text Colors ---------------- */
.normal { color: #10b981; font-weight: 700; font-size: 20px; }
.tb { color: #ef4444; font-weight: 700; font-size: 20px; }

/* ---------------- Progress Bar ---------------- */
.stProgress > div > div > div > div {
    border-radius: 12px;
    background: linear-gradient(270deg, #fbbf24, #3b82f6, #ec4899, #fbbf24);
    background-size: 600% 100%;
    animation: wave 2s linear infinite;
}
@keyframes wave {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* ---------------- Button Styling ---------------- */
button {
    background-color: #6366f1;
    color: white;
    font-weight: 600;
    border-radius: 15px;
    padding: 10px 25px;
    transition: transform 0.2s ease, background 0.2s ease;
}
button:hover {
    background-color: #3b82f6 !important;
    transform: scale(1.05);
}

/* ---------------- File Uploader ---------------- */
div.stFileUploader {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 20px;
    padding: 10px;
    margin-bottom: 20px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}
</style>
<link href="https://fonts.googleapis.com/css2?family=Baloo+2:wght@400;700&family=Comic+Neue:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# -------------------------------
# App Header
# -------------------------------
st.markdown("<h1>🩺 Tuberculosis Detection App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#334155;font-weight:600;'>Upload Cough Audio or X-ray or both. Get TB prediction instantly.</p>", unsafe_allow_html=True)
st.write("---")

# -------------------------------
# File Uploaders
# -------------------------------
audio_file = st.file_uploader("🎤 Upload Cough Audio (.wav, .mp3, .flac, .ogg)", 
                              type=["wav", "mp3", "flac", "ogg"])
xray_file = st.file_uploader("🩻 Upload Chest X-ray Image (.png, .jpg, .jpeg)", 
                             type=["png", "jpg", "jpeg"])

# -------------------------------
# Smooth Animated Progress Bar Function
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
