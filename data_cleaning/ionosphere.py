import pandas as pd
import numpy as np

input_path = '../data/raw/ionosphere.csv'
df = pd.read_csv(input_path, header=None)

print(f"Original shape: {df.shape}")

feature_names = [f'Attr_{i}' for i in range(1, 35)]
df.columns = feature_names + ['Class']

df['Class'] = df['Class'].map({'g': 0, 'b': 1})

print("Class distribution (0=Good, 1=Bad):")
print(df['Class'].value_counts())

nunique = df.nunique()
cols_to_drop = nunique[nunique == 1].index

if len(cols_to_drop) > 0:
    print(f"\nDropping constant columns (no variance): {list(cols_to_drop)}")
    df = df.drop(cols_to_drop, axis=1)

if df.isnull().sum().sum() > 0:
    print("Imputing missing values with median...")
    df = df.fillna(df.median())

print("-" * 30)
print(f"Cleaned dataset shape: {df.shape}")
print("Preview (first 5 rows):")
print(df.head())

output_path = '../data/clean/Clean_ionosphere.csv'
df.to_csv(output_path, index=False)
print(f"\nProcessed file saved to: {output_path}")