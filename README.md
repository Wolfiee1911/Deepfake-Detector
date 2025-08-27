# 🎭 Deepfake Detection System  

## 🔹 Overview  
This project implements a **Deepfake Detection pipeline** using **YOLOv7, PyTorch, and Streamlit**.  
It detects manipulated faces in videos and images, providing a simple **web app interface** for demonstrations.  

✨ Features:  
- 🔍 Real-time deepfake detection on images/videos  
- 🖥️ Streamlit web UI for interactive demo  
- 📊 Model trained on **FakeAVCeleb** and **FaceForensics++ (C23)** datasets  
- 📈 Achieved ~92% classification accuracy in evaluation  

---

## 📂 Repository Structure  

deepfake-detection/
│── README.md
│── requirements.txt
│── main.py
│── .gitignore
│
├── configs/ # JSON configs (splits, calibration)
├── scripts/ # Training, evaluation, preprocessing, Streamlit app
├── src/ # Core pipeline code
├── results/ # Sample predictions & UI snapshots
├── .streamlit/ # Streamlit configuration
└── data/ # (ignored) datasets/weights


---

## ⚙️ Installation & Setup  

### 1. Clone the repository  
```bash
git clone https://github.com/Wolfiee1911/Deepfake-Detector.git
cd Deepfake-Detector
