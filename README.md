# 🧠 Breast Tumor Classification and Segmentation Web App

This project is a Flask-based web application for **breast tumor analysis** from ultrasound images. It provides:

- 🩺 **Classification** using EfficientNetB3 (`Benign`, `Malignant`, `Normal`)
- 🧠 **Segmentation** of tumors using a custom-built **Attention U-Net**
- 🎯 **Explainability** through Grad-CAM 
- 🌐 Clean and interactive web interface for uploading and visualizing results

---

## 🚀 Features

- **Upload** any ultrasound image.
- **Classify** as Benign / Malignant / Normal.
- If abnormal: **highlight tumor region** with segmentation overlay.
- Built with **TensorFlow, Keras, OpenCV, Flask**.

---

## 🖥️ Tech Stack

| Area           | Tools Used                         |
|----------------|------------------------------------|
| Backend        | Flask, TensorFlow, Keras           |
| Frontend       | HTML, CSS (custom styling)         |
| Models         | EfficientNetB3, Attention U-Net    |
| Image Handling | OpenCV                             |

---

## 🛠️ Setup Instructions

1. **Clone the repo**
```bash
git clone https://github.com/Vyshnavi-Usha/Breast-Cancer-Analysis-Tumor-Classification-and-Segmentation.git
cd Breast-Cancer-Analysis-Tumor-Classification-and-Segmentation
```

2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 4. Download Model Files

Place the following `.h5` model files in the root directory:
- `efficientnet_model.h5`
- `tumor_segmentation_model.h5`

> 💡 These files are too large to be stored on GitHub directly.  
> You can download them from Hugging Face:
> - [EfficientNet Model](https://huggingface.co/Vysh-navi/breast-cancer-tumor-classification-segmentation/resolve/main/efficientnet_model.h5)
> - [Tumor Segmentation Model](https://huggingface.co/Vysh-navi/breast-cancer-tumor-classification-segmentation/resolve/main/tumor_segmentation_model.h5
)


5. **Run the app**
```bash
python app.py
```
Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📊 Model Overview

### 🧬 Classification (EfficientNetB3)
- Input shape: `(224, 224, 3)`
- Output classes: `Benign`, `Malignant`, `Normal`

### 🧠 Segmentation (Attention U-Net)
- Input shape: `(256, 256, 3)`
- Output: Tumor binary mask

---

## 📸 Sample Results

### Input  
![Input image](static/input.png)

### Output 
![Segmented Output](static/output.png)

---

## ✅ Requirements

All required Python packages are listed in `requirements.txt`.

---

## 🙌 Acknowledgements

- BUSI Dataset


---
