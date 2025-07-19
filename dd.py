import pandas as pd

df = pd.read_csv(r"C:\Users\ACER USER\OneDrive\Documents\sample_50_medical_dataset_with_severity.csv")
print(df['disease_name'].nunique())
print(df['disease_name'].unique())
