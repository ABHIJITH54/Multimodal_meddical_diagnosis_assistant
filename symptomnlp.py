import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the dataset
df = pd.read_csv("generated_symptom_dataset.csv")

# Step 2: Encode labels
disease_encoder = LabelEncoder()
severity_encoder = LabelEncoder()
df['Disease Encoded'] = disease_encoder.fit_transform(df['Disease Name'])
df['Severity Encoded'] = severity_encoder.fit_transform(df['Severity'])

# Step 3: Split dataset
X = df['Symptom']
y = df[['Disease Encoded', 'Severity Encoded']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42)))
])

# Step 5: Train model
pipeline.fit(X_train, y_train)

# Step 6: Save model and encoders
joblib.dump(pipeline, 'nlp_pipeline.pkl')
joblib.dump(disease_encoder, 'disease_encoder.pkl')
joblib.dump(severity_encoder, 'severity_encoder.pkl')


# Step 10: Optional - Evaluate model performance
y_pred = pipeline.predict(X_test)
print("\nDisease Classification Report:\n", classification_report(
    y_test.iloc[:, 0], y_pred[:, 0], target_names=disease_encoder.classes_, zero_division=0))
print("\nSeverity Classification Report:\n", classification_report(
    y_test.iloc[:, 1], y_pred[:, 1], target_names=severity_encoder.classes_, zero_division=0))
