import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('../data/raw/covid.csv')

if 'Patient_ID' in df.columns:
    df = df.drop('Patient_ID', axis=1)

if 'Status' in df.columns:
    df = df.rename(columns={'Status': 'Class'})

if df['Class'].dtype == 'object':
    le = LabelEncoder()
    df['Class'] = le.fit_transform(df['Class'])

threshold = 0.7
missing_ratios = df.isnull().mean()
cols_to_drop = missing_ratios[missing_ratios > threshold].index

if 'Class' in cols_to_drop:
    cols_to_drop = cols_to_drop.drop('Class')

if len(cols_to_drop) > 0:
    print(f"Dropping sparse columns: {list(cols_to_drop)}")
    df = df.drop(columns=cols_to_drop)

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

bool_cols = df_clean.select_dtypes(include=['bool']).columns
if len(bool_cols) > 0:
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Data types in cleaned dataset:")
print(df_clean.dtypes.value_counts())

df_clean.to_csv('../data/clean/Clean_covid.csv', index=False)
print("\nCleaned dataset saved successfully to: ../data/clean/Clean_covid.csv")