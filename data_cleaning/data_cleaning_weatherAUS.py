import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

input_path = '../data/raw/weatherAUS.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df = df.drop('Date', axis=1)
    print("Processed date: extracted 'Month', dropped 'Date'.")

if 'RainTomorrow' in df.columns:
    df = df.rename(columns={'RainTomorrow': 'Class'})

df = df.dropna(subset=['Class'])

le = LabelEncoder()
df['Class'] = le.fit_transform(df['Class'])

threshold = 0.7
missing_ratios = df.isnull().mean()
cols_to_drop = missing_ratios[missing_ratios > threshold].index
if len(cols_to_drop) > 0:
    print(f"Dropping columns with >70% missing values: {list(cols_to_drop)}")
    df = df.drop(columns=cols_to_drop)

numeric_cols = df.select_dtypes(include=[np.number]).columns
object_cols = df.select_dtypes(exclude=[np.number]).columns

for col in numeric_cols:
    if col != 'Class':
        df[col] = df[col].fillna(df[col].median())

for col in object_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

if 'Class' in df_clean.columns:
    cols = [c for c in df_clean.columns if c != 'Class'] + ['Class']
    df_clean = df_clean[cols]

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print("Columns preview (first 10):")
print(df_clean.columns.tolist()[:10])

if 'Month' in df_clean.columns:
    print(f"Column 'Month' exists. Sample values: {df_clean['Month'].unique()[:5]}")

output_path = '../data/clean/Clean_weatherAUS.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nProcessed file saved to: {output_path}")