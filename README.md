# 🩺 AI-Powered Tuberculosis Detection App

## 📌 Overview
Tuberculosis (TB) remains one of the leading infectious causes of death worldwide, with India carrying nearly **27% of the global TB burden**.  
Traditional diagnostic methods (sputum microscopy, GeneXpert, cultures) are reliable but often **slow, costly, and inaccessible** in rural areas.  

This project introduces an **AI-powered TB detection system** that uses:
- **Cough Audio Analysis** (functional indicators of respiratory health)
- **Chest X-ray Analysis** (structural abnormalities in the lungs)

Users can upload **audio, X-ray, or both**, and the system predicts TB probability. If both are uploaded, results are combined for higher reliability.

---

## ✨ Features
- Upload **cough recordings (.wav)** or **chest X-rays (.png/.jpg)**  
- **AI models** trained using PyTorch for both modalities  
- **Multimodal fusion**: uses either one or both inputs for prediction  
- Clean **Streamlit Web App** interface  
- Can run **locally** or be **deployed on Streamlit Cloud**  

---

## 🏗️ Architectures Used
1. **Audio Model**  
   - Preprocessing: Mel-spectrograms using `torchaudio`  
   - Model: CNN trained on frequency-temporal cough features  

2. **X-ray Model**  
   - Preprocessing: Image resizing, normalization, augmentation  
   - Model: Fine-tuned **ResNet-18** / **ResNet-50**  

3. **Fusion**  
   - Independent outputs combined by **averaging probability scores**  

---

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Quaser001/my_tb_app.git
cd my_tb_app

2. Install Dependencies

Create a virtual environment and install requirements:

pip install -r requirements.txt

3. Run App Locally
streamlit run app.py


The app will open in your browser at http://localhost:8501.

🌐 Online Deployment (Streamlit Cloud)

Push repo to GitHub

Sign in to Streamlit Cloud

Connect your GitHub repository

Deploy the app (app.py as entry point)

📊 Example Usage

Upload only cough audio → TB probability shown

Upload only X-ray → TB probability shown

Upload both → Combined TB probability displayed

💡 Societal Impact

Early Screening: Enables rapid detection in seconds instead of days.

Accessibility: Works on basic devices, usable in rural and low-resource settings.

Scalability: Deployable via cloud to millions of users.

Support for Healthcare: Provides doctors and frontline workers with an assistive tool, not a replacement.

Global Relevance: Useful in TB-burdened regions (India, Africa, Southeast Asia).

📂 Repository Structure
my_tb_app/
│── app.py              # Streamlit app
│── model_utils.py      # Helper functions for preprocessing & prediction
│── audio_best.pt       # Trained audio model weights
│── xray_best.pt        # Trained X-ray model weights
│── requirements.txt    # Dependencies
│── README.md           # Project documentation

👥 Authors

Asef Ur Rahman

With support from AI/ML research & open-source community

📜 License

This project is licensed under the MIT License – free to use, modify, and distribute.
