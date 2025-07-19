import tensorflow as tf
import numpy as np
import cv2
import json
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import load_model

model = load_model('final_resnet101_skin.keras')
with open("class_indices_resnet101.json", "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}
def preprocess_image(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image at path: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img.astype(np.float32), axis=0)
    img_preprocessed = preprocess_input(img_array)
    return img_preprocessed
def predict_image(image_path):
    processed_img = preprocess_image(image_path)
    predictions = model.predict(processed_img)
    predicted_index = np.argmax(predictions)
    predicted_class = idx_to_class[predicted_index]
    confidence = predictions[0][predicted_index]
    return predicted_class, confidence
image_path = r"E:\Multimodal_meddical_diagnosis_assistant\hemangioma-764.jpeg"
predicted_class, confidence = predict_image(image_path)
print(f" Predicted class: {predicted_class} ({confidence*100:.2f}%)")
