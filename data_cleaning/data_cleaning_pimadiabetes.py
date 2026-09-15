import pandas as pd
import numpy as np

input_path = '../data/raw/pimadiabetes.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")

if 'Outcome' in df.columns:
    df = df.rename(columns={'Outcome': 'Class'})

cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("\nCount of zeros in columns (before fix):")
print((df[cols_with_zeros] == 0).sum())

df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

for col in cols_with_zeros:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

print("\nCount of zeros in columns (after fix - should be 0):")
print((df[cols_with_zeros] == 0).sum())

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print("Preview (first 5 rows):")
print(df_clean.head())

output_path = '../data/clean/Clean_pimadiabetes.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nProcessed file saved to: {output_path}")