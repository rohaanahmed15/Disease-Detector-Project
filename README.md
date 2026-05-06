🏥 Medical AI Prediction System

An end-to-end Artificial Intelligence application designed to predict multiple medical conditions using Machine Learning and Deep Learning.

---

📌 Overview

This project is a multi-model medical prediction system that can analyze:

- Symptoms (for general disease prediction)
- Clinical data (for heart disease)
- Medical images like X-rays and MRI scans

It integrates multiple AI models into a single interactive web application.

---

Features

- Predicts 41+ diseases based on symptoms
   Detects pneumonia from chest X-rays
- Predicts heart disease using clinical data
- Classifies brain tumors from MRI scans
- Detects breast cancer (benign/malignant)
- Shows confidence scores
-  Provides disease descriptions & precautions
-  Interactive UI using Gradio

---

Models Used

1. MLPClassifier (Tabular Data)

Used for:

- General Disease Prediction
- Heart Disease Prediction
- Breast Cancer Detection

Works on numerical data like symptoms and medical records.

---

2. VGG16 CNN (Image Data)

Used for:

- Lung Disease Detection (X-rays)
- Brain Tumor Classification (MRI)

Uses Transfer Learning for better performance.

---

Dataset

The system uses 5 datasets:

- Disease Symptom Dataset
- Chest X-ray Dataset
- Heart Disease Dataset
- Brain MRI Dataset
- Breast Cancer Dataset

---

Tech Stack

- Python
- Scikit-learn
- TensorFlow / Keras
- NumPy & Pandas
- Gradio (for UI)
- Matplotlib / Seaborn

---

How It Works

1. User provides input (symptoms / image / clinical data)
2. Input is preprocessed
3. Model makes prediction
4. System returns:
   - Disease name
   - Confidence score
   - Description
   - Precautions

---

Results

Model| Accuracy
Disease Prediction| 100%
Lung Disease| 90%
Heart Disease| 87.5%
Brain Tumor| 83.5%
Breast Cancer| 87.5%

---

Disclaimer

This project is for educational purposes only.
It should not be used as a replacement for professional medical advice.

---

Future Improvements

- Add more diseases
- Combine image + clinical data
- Deploy on cloud
- Mobile application
- Explainable AI (Grad-CAM)

---

Authors

- Rohaan Ahmed
- Team Members

---

How to Run

pip install -r requirements.txt
python app.py

---

 Acknowledgment

Datasets sourced from Kaggle and other open repositories.
