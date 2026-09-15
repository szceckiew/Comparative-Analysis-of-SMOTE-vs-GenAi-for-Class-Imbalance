import pandas as pd
import os

RAW_FILE_PATH = "../data/raw/creditcard.csv"
CLEAN_FILE_PATH = "../data/clean/Clean_creditcard.csv"

df = pd.read_csv(RAW_FILE_PATH)

if 'Time' in df.columns:
    df = df.drop('Time', axis=1)

df = df.drop_duplicates()

if df['Class'].dtype == object:
    df['Class'] = df['Class'].astype(str).str.replace('"', '').astype(int)

os.makedirs(os.path.dirname(CLEAN_FILE_PATH), exist_ok=True)
df.to_csv(CLEAN_FILE_PATH, index=False)
print(f"Dataset saved to: {CLEAN_FILE_PATH}")