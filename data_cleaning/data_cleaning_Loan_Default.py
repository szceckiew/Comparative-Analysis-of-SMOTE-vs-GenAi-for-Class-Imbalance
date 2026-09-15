import pandas as pd
import numpy as np

input_path = '../data/raw/Loan_Default.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")
print("Total missing values before cleaning:")
print(df.isnull().sum().sum())

if 'ID' in df.columns:
    df = df.drop('ID', axis=1)

if 'year' in df.columns:
    df = df.drop('year', axis=1)

if 'Status' in df.columns:
    df = df.rename(columns={'Status': 'Class'})

df = df.dropna(subset=['Class'])
df['Class'] = df['Class'].astype(int)

numeric_cols = df.select_dtypes(include=[np.number]).columns
object_cols = df.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols:
    if col != 'Class':
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

for col in object_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

if 'Class' in df_clean.columns:
    cols = [c for c in df_clean.columns if c != 'Class'] + ['Class']
    df_clean = df_clean[cols]

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
print("Sample columns:")
print(df_clean.columns.tolist()[:10])

output_path = '../data/clean/Clean_Loan_Default.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nProcessed dataset saved to: {output_path}")