import streamlit as st
st.set_page_config(page_title="Multimodal Medical Diagnosis", layout="centered") 

import tensorflow as tf
import numpy as np
import cv2
import json
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet import preprocess_input
from PIL import Image
import io
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add the background image
add_bg_from_local(r"E:\Multimodal_meddical_diagnosis_assistant\bg4.jpg") 

@st.cache_resource
def load_models():
    eye_model = load_model('final_resnet101_eye.keras')
    skin_model = load_model('final_resnet101_skin.keras')
    return eye_model, skin_model

@st.cache_data
def load_class_indices():
    with open("class_indices_resnet101_eye.json", "r") as f:
        idx_eye = json.load(f)
    with open("class_indices_resnet101.json", "r") as f:
        idx_skin = json.load(f)
    return {v: k for k, v in idx_eye.items()}, {v: k for k, v in idx_skin.items()}

@st.cache_resource
def load_nlp_assets():
    nlp_pipeline = joblib.load('nlp_pipeline.pkl')
    disease_encoder = joblib.load('disease_encoder.pkl')
    severity_encoder = joblib.load('severity_encoder.pkl')
    return nlp_pipeline, disease_encoder, severity_encoder


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    img = image.convert("RGB").resize(target_size)
    img_array = np.array(img).astype(np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)


def predict_image(image: Image.Image, model, idx_to_class):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_index = np.argmax(predictions)
    predicted_class = idx_to_class[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence


def predict_text(text, pipeline, disease_encoder, severity_encoder):
    pred = pipeline.predict([text])[0]
    return (
        disease_encoder.inverse_transform([pred[0]])[0],
        severity_encoder.inverse_transform([pred[1]])[0],
        0.95  
    )


eye_model, skin_model = load_models()
idx_eye, idx_skin = load_class_indices()
nlp_pipeline, disease_encoder, severity_encoder = load_nlp_assets()


st.title("⚕️ Multimodal Medical Diagnosis Assistant")

modality = st.selectbox("Select Image Type", ["Eye", "Skin"])
uploaded_image = st.file_uploader("Upload Eye/Skin Image", type=["jpg", "jpeg", "png"])
symptom_text = st.text_area("Describe your symptoms (optional)", height=100)

# if uploaded_image:
#     st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

if st.button("Diagnose"):
    has_image = uploaded_image is not None
    has_text = symptom_text.strip() != ""

    if not has_image and not has_text:
        st.warning("Please upload an image or enter symptoms.")
    else:
        final_disease, severity, source, conf_image, conf_nlp = None, None, None, None, None

        
        if has_image:
            image_pil = Image.open(uploaded_image)
            if modality == "Eye":
                final_disease_img, conf_image = predict_image(image_pil, eye_model, idx_eye)
            else:
                final_disease_img, conf_image = predict_image(image_pil, skin_model, idx_skin)

        
        if has_text:
            final_disease_nlp, severity, conf_nlp = predict_text(symptom_text, nlp_pipeline, disease_encoder, severity_encoder)

    
        if has_image and has_text:
            if conf_image >= conf_nlp:
                final_disease = final_disease_img
                source = f"{modality} Image ({conf_image*100:.2f}%)"
            else:
                final_disease = final_disease_nlp
                source = f"Symptoms ({conf_nlp*100:.2f}%)"
        elif has_image:
            final_disease, source = final_disease_img, f"{modality} Image ({conf_image*100:.2f}%)"
        elif has_text:
            final_disease, source = final_disease_nlp, f"Symptoms ({conf_nlp*100:.2f}%)"

        
        st.subheader("🩺 Diagnosis Result")
        st.success(f"🦠 Final Predicted Disease: **{final_disease}**")
        st.info(f"🔍 Based on: {source}")
        if has_text:
            st.info(f"📊 Severity (from symptoms): **{severity}**")
        st.warning("⚠ Please consult a doctor.")



