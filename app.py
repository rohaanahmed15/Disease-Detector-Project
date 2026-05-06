import gradio as gr
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

print("✅ Libraries imported!")

files = [
    'model_MLPC.sav',
    'chest_xray_model.h5',
    'model_heart.sav',
    'scaler_heart.sav',
    'model_brain.h5',
    'model_cancer.sav',
    'scaler_cancer.sav',
    'Symptom-severity.csv',
    'model_breast_cnn.keras'
]

print("Checking files:")
for f in files:
    if os.path.exists(f):
        print(f"✅ {f}")
    else:
        print(f"❌ {f} NOT FOUND")

from keras.models import load_model
# Common Disease
model_MLPC = pickle.load(open('model_MLPC.sav', 'rb'))
severity = pd.read_csv('Symptom-severity.csv')
severity['Symptom'] = severity['Symptom'].str.replace('_', ' ').str.strip()
symptom_list = sorted(severity['Symptom'].unique().tolist())

# Load description and precaution files
description_df = pd.read_csv('symptom_Description.csv')
precaution_df = pd.read_csv('symptom_precaution.csv')

# Lung Disease
model_lung = load_model('chest_xray_model.h5')

# Heart Disease
model_heart = pickle.load(open('model_heart.sav', 'rb'))
scaler_heart = pickle.load(open('scaler_heart.sav', 'rb'))

# Brain Tumor
model_brain = load_model('model_brain.h5')

# Breast Cancer
model_cancer = pickle.load(open('model_cancer.sav', 'rb'))
scaler_cancer = pickle.load(open('scaler_cancer.sav', 'rb'))

print("✅ All models and files loaded!")
# 1 - Disease Prediction
def predict_disease(symptoms):
    if not symptoms:
        return "⚠️ Please select at least one symptom!"
    if len(symptoms) < 3:
        return "⚠️ Please select at least 3 symptoms!"

    weights = [0.0] * 17
    for i, sym in enumerate(symptoms[:17]):
        row = severity[severity['Symptom'] == sym]
        if not row.empty:
            weights[i] = float(row['weight'].values[0])

    features = np.array(weights).reshape(1, -1)
    prediction = model_MLPC.predict(features)[0]
    proba = model_MLPC.predict_proba(features)[0]
    confidence = max(proba) * 100
    classes = model_MLPC.classes_
    top3_idx = proba.argsort()[-3:][::-1]

    desc_row = description_df[description_df['Disease'].str.strip() == prediction.strip()]
    description = desc_row['Description'].values[0] if not desc_row.empty else "No description available"

    prec_row = precaution_df[precaution_df['Disease'].str.strip() == prediction.strip()]
    precautions = [str(prec_row[f'Precaution_{i}'].values[0]) for i in range(1, 5)] if not prec_row.empty else ["No precautions available"]

    if confidence < 70:
        conf_msg = "⚠️ Low confidence — add more symptoms"
    elif confidence < 85:
        conf_msg = "✅ Moderate confidence"
    else:
        conf_msg = "✅ High confidence"

    result  = f"🏥 Predicted Disease: {prediction}\n"
    result += f"📊 Confidence: {confidence:.2f}% — {conf_msg}\n\n"
    result += f"📋 Description:\n{description}\n\n"
    result += f"⚠️ Precautions:\n"
    for i, p in enumerate(precautions, 1):
        if str(p) != 'nan':
            result += f"  {i}. {p}\n"
    result += f"\n🔝 Top 3 Predictions:\n"
    for i, idx in enumerate(top3_idx, 1):
        result += f"  {i}. {classes[idx]} — {proba[idx]*100:.2f}%\n"

    return result

# 2 - Lung Disease
def predict_lung(img):
    if img is None:
        return "⚠️ Please upload an X-Ray image!"
    img_array = image.img_to_array(img) / 255.0
    img_array = tf.image.resize(img_array, [150, 150])
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_lung.predict(img_array)[0][0]
    if prediction > 0.6:
        return f"🔴 PNEUMONIA Detected\n📊 Confidence: {prediction*100:.2f}%\n\n⚠️ Please consult a doctor immediately!"
    else:
        return f"🟢 NORMAL — No Pneumonia\n📊 Confidence: {(1-prediction)*100:.2f}%\n\n✅ Lungs appear normal"

# 3 - Heart Disease
def predict_heart(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope):
    features = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]]
    scaled = scaler_heart.transform(features)
    prediction = model_heart.predict(scaled)[0]
    proba = model_heart.predict_proba(scaled)[0]
    confidence = max(proba) * 100
    if prediction == 1:
        return f"🔴 Heart Disease DETECTED\n📊 Confidence: {confidence:.2f}%\n\n⚠️ Please consult a cardiologist immediately!"
    else:
        return f"🟢 No Heart Disease\n📊 Confidence: {confidence:.2f}%\n\n✅ Heart appears healthy"

# 4 - Brain Tumor
def predict_brain(img):
    if img is None:
        return "⚠️ Please upload an MRI image!"
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    img_array = image.img_to_array(img) / 255.0
    img_array = tf.image.resize(img_array, [150, 150])
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model_brain.predict(img_array)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = max(prediction) * 100

    result = f"🧠 Detected: {predicted_class}\n"
    result += f"📊 Confidence: {confidence:.2f}%\n\n"
    result += "All Probabilities:\n"
    for name, prob in zip(class_names, prediction):
        result += f"  • {name}: {prob*100:.2f}%\n"

    if predicted_class == 'No Tumor':
        result += "\n✅ No tumor detected"
    else:
        result += "\n⚠️ Please consult a neurologist immediately!"

    return result

# 5 - Breast Cancer
def predict_breast(mean_radius, mean_texture, mean_perimeter, mean_area,
                   mean_smoothness, mean_compactness, mean_concavity,
                   mean_concave_points, mean_symmetry, mean_fractal):
    features = [[mean_radius, mean_texture, mean_perimeter, mean_area,
                 mean_smoothness, mean_compactness, mean_concavity,
                 mean_concave_points, mean_symmetry, mean_fractal,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    scaled = scaler_cancer.transform(features)
    prediction = model_cancer.predict(scaled)[0]
    proba = model_cancer.predict_proba(scaled)[0]
    confidence = max(proba) * 100
    if prediction == 0:
        return f"🔴 MALIGNANT — Cancer Detected\n📊 Confidence: {confidence:.2f}%\n\n⚠️ Please consult an oncologist immediately!"
    else:
        return f"🟢 BENIGN — No Cancer\n📊 Confidence: {confidence:.2f}%\n\n✅ No cancer detected"

print("✅ Functions defined!")

with gr.Blocks(title="Medical AI System", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏥 Medical AI Prediction System
    ### Powered by Machine Learning & Deep Learning
    ---
    """)

    with gr.Tab("🤒 Disease Prediction"):
        gr.Markdown("### Select your symptoms (minimum 3)")
        symptoms_input = gr.Dropdown(
            choices=symptom_list,
            multiselect=True,
            label="Symptoms",
            info="Select all symptoms you are experiencing"
        )
        disease_output = gr.Textbox(label="Result", lines=15)
        disease_btn = gr.Button("🔍 Predict Disease", variant="primary")
        disease_btn.click(predict_disease, symptoms_input, disease_output)

    with gr.Tab("🫁 Lung Disease"):
        gr.Markdown("### Upload a Chest X-Ray Image")
        lung_input = gr.Image(type="pil", label="Chest X-Ray")
        lung_output = gr.Textbox(label="Result", lines=5)
        lung_btn = gr.Button("🔍 Analyze X-Ray", variant="primary")
        lung_btn.click(predict_lung, lung_input, lung_output)

    with gr.Tab("❤️ Heart Disease"):
        gr.Markdown("### Enter Patient Details")
        with gr.Row():
            age = gr.Number(label="Age")
            sex = gr.Number(label="Sex (1=Male, 0=Female)")
            cp = gr.Number(label="Chest Pain (0-3)")
        with gr.Row():
            trestbps = gr.Number(label="Resting BP")
            chol = gr.Number(label="Cholesterol")
            fbs = gr.Number(label="Fasting Blood Sugar (1=Yes, 0=No)")
        with gr.Row():
            restecg = gr.Number(label="Resting ECG (0-2)")
            thalach = gr.Number(label="Max Heart Rate")
            exang = gr.Number(label="Exercise Angina (1=Yes, 0=No)")
        with gr.Row():
            oldpeak = gr.Number(label="Oldpeak")
            slope = gr.Number(label="ST Slope (0-2)")
        heart_output = gr.Textbox(label="Result", lines=5)
        heart_btn = gr.Button("🔍 Predict Heart Disease", variant="primary")
        heart_btn.click(
            predict_heart,
            [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope],
            heart_output
        )

    with gr.Tab("🧠 Brain Tumor"):
        gr.Markdown("### Upload a Brain MRI Image")
        brain_input = gr.Image(type="pil", label="Brain MRI")
        brain_output = gr.Textbox(label="Result", lines=8)
        brain_btn = gr.Button("🔍 Analyze MRI", variant="primary")
        brain_btn.click(predict_brain, brain_input, brain_output)

    with gr.Tab("🎗️ Breast Cancer"):
        gr.Markdown("### Enter Tumor Measurements")
        with gr.Row():
            mean_radius = gr.Number(label="Mean Radius")
            mean_texture = gr.Number(label="Mean Texture")
            mean_perimeter = gr.Number(label="Mean Perimeter")
        with gr.Row():
            mean_area = gr.Number(label="Mean Area")
            mean_smoothness = gr.Number(label="Mean Smoothness")
            mean_compactness = gr.Number(label="Mean Compactness")
        with gr.Row():
            mean_concavity = gr.Number(label="Mean Concavity")
            mean_concave_points = gr.Number(label="Mean Concave Points")
            mean_symmetry = gr.Number(label="Mean Symmetry")
        mean_fractal = gr.Number(label="Mean Fractal Dimension")
        cancer_output = gr.Textbox(label="Result", lines=5)
        cancer_btn = gr.Button("🔍 Predict Cancer", variant="primary")
        cancer_btn.click(
            predict_breast,
            [mean_radius, mean_texture, mean_perimeter, mean_area,
             mean_smoothness, mean_compactness, mean_concavity,
             mean_concave_points, mean_symmetry, mean_fractal],
            cancer_output
        )

    gr.Markdown("---\n⚠️ **Disclaimer:** For educational purposes only. Always consult a qualified doctor.")

demo.launch()

