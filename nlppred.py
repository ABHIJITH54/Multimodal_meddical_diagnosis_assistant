import joblib
# Step 7: Load model and encoders (for future use)
pipeline = joblib.load('nlp_pipeline.pkl')
disease_encoder = joblib.load('disease_encoder.pkl')
severity_encoder = joblib.load('severity_encoder.pkl')

# Step 8: Prediction function
def predict_disease(symptom_text):
    prediction = pipeline.predict([symptom_text])[0]
    disease = disease_encoder.inverse_transform([prediction[0]])[0]
    severity = severity_encoder.inverse_transform([prediction[1]])[0]
    return {
        "Disease": disease,
        "Severity": severity,
        "Message": "Please consult a doctor."
    }

# Step 9: Test on new input
example = predict_disease("Constricted visual field and Difficulty adjusting to dark rooms")
print("\nExample Prediction:", example)


