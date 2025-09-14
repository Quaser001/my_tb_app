Project Description: AI-Powered Tuberculosis Detection System
Problem Statement

Tuberculosis (TB) remains one of the deadliest infectious diseases in the world.

India alone carries ~27% of the global TB burden, with over 2.8 million cases annually.

The challenge lies in early detection:

Traditional tests like sputum microscopy, GeneXpert, or culture tests are accurate but slow, expensive, and require lab infrastructure.

In rural areas, access to these facilities is limited.

Many patients are only diagnosed after severe progression, which increases mortality and transmission.

Thus, there is an urgent need for a fast, accessible, and cost-effective screening tool.

Proposed Solution

Our project introduces a Multimodal AI-powered TB Detection System, which uses two key indicators of pulmonary TB:

Cough sounds (audio modality) → functional changes in the respiratory system.

Chest X-rays (image modality) → structural damage in the lungs.

By combining both, the system provides a probability score for TB that is:

Faster (seconds instead of days).

More accessible (usable in rural clinics with only a smartphone or X-ray machine).

Low-cost compared to lab tests.

Data & Preprocessing

Cough Audio Data: Preprocessed using Mel-spectrograms via torchaudio, capturing frequency and temporal acoustic features unique to TB coughs.

X-ray Data: Applied transformations (resizing, normalization, augmentation) for consistency and robustness.

Training Split: Data divided into training, validation, and test sets to ensure model generalization.

Model Architectures

Audio Model (Cough Analysis)

Used CNNs (Convolutional Neural Networks) trained on spectrograms.

Extracts temporal–spectral cough features indicative of TB.

Image Model (X-ray Analysis)

Fine-tuned ResNet-18 / ResNet-50 from torchvision.

Detects TB-related lung abnormalities such as nodules, infiltrates, or cavities.

Fusion Approach (Multimodal Integration)

If only one modality is given (audio or X-ray), the corresponding model outputs the probability.

If both are given, the outputs are combined (averaged/weighted fusion) for more reliable screening.

Implementation

Built using PyTorch for training and Streamlit for deployment.

Deployed as a web application where users can:

Upload a cough audio file (.wav).

Upload a chest X-ray (.jpg/.png).

Or upload both.

The app runs the saved AI models (.pt weights), computes TB probability, and outputs results in a simple, user-friendly UI.

Impact on Society

For India:

Acts as a first-level screening tool in villages and primary healthcare centers.

Reduces dependency on labs and speeds up detection.

Helps in achieving India’s goal of TB elimination by 2025.

Global Impact:

Can be deployed in low-resource countries (Africa, Southeast Asia) facing similar TB burdens.

Provides a non-invasive, rapid, and scalable approach.

Potential integration with mobile health platforms, enabling remote screening.

Core Benefits

Accessibility: Runs on basic hardware, no need for high-end medical devices.

Scalability: Can be expanded to thousands of users through cloud deployment.

Cost-effectiveness: Reduces the need for expensive lab tests for initial screening.

Early Diagnosis: Identifies potential TB patients early, reducing spread and improving outcomes.

Support for Clinicians: Not a replacement, but an assistive tool that augments doctors’ decisions.

Broader Vision

This project demonstrates how AI + healthcare can work hand-in-hand:

Empowering frontline workers with decision-support tools.

Bridging the gap between rural patients and advanced diagnostics.

Reducing TB’s burden in India and worldwide by making early screening universal.

👉 In short:
"Our project leverages Artificial Intelligence to detect TB using cough sounds and chest X-rays. By fusing multimodal data, it delivers rapid, accurate, and accessible TB screening. It addresses India’s massive TB challenge and has the potential to transform global healthcare by enabling early detection in low-resource settings."

Would you like me to also draft a 1–2 minute script (like a spoken version) so you can present it smoothly during your presentation, instead of just having slide content?

You said:
add a README FILE
ChatGPT said:

Got it ✅ — here’s a polished README.md file for your GitHub repo (my_tb_app) that clearly explains the project, installation, usage, and societal impact.

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
