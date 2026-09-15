import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

RAW_FILE_PATH = "../data/raw/Breast_Cancer_Wisconsin.csv"
CLEAN_FILE_PATH = "../data/clean/Clean_Breast_Cancer_Wisconsin.csv"

df_raw = pd.read_csv(RAW_FILE_PATH, header=0)

df = df_raw.copy(deep=True)

df = df.drop(columns=['Unnamed: 32', 'id'], errors='ignore')

df = df.rename(columns={'diagnosis': 'Class'})

target = 'Class'

le = LabelEncoder()

df[target] = le.fit_transform(df[target].astype(str))

mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"Encoding used: {mapping}")

print(df.head())
print(df.shape)

os.makedirs(os.path.dirname(CLEAN_FILE_PATH), exist_ok=True)

df.to_csv(CLEAN_FILE_PATH, index=False)
print(f"\nCleaned dataset saved to: {CLEAN_FILE_PATH}")