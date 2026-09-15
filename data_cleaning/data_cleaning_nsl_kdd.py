import pandas as pd
import numpy as np

input_path = '../data/raw/nsl_kdd_full_dataset.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")

if 'difficulty' in df.columns:
    df = df.drop('difficulty', axis=1)

if 'class' in df.columns:
    df = df.rename(columns={'class': 'Class'})

df['Class'] = df['Class'].apply(lambda x: 0 if x == 'normal' else 1)

print("Class distribution (0=normal, 1=attack):")
print(df['Class'].value_counts())

numeric_cols = df.select_dtypes(include=[np.number]).columns
object_cols = df.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols:
    if col != 'Class':
        df[col] = df[col].fillna(df[col].median())

for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

if 'Class' in df_clean.columns:
    cols = [c for c in df_clean.columns if c != 'Class'] + ['Class']
    df_clean = df_clean[cols]

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print("Column preview (sample):")
print(df_clean.columns.tolist()[:10])

output_path = '../data/clean/Clean_nsl_kdd_full_dataset.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nCleaned file saved to: {output_path}")