import pandas as pd
import numpy as np

input_path = '../data/raw/Titanic-Dataset.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")

cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
df = df.drop(columns=existing_cols_to_drop)

if 'Survived' in df.columns:
    df = df.rename(columns={'Survived': 'Class'})

if 'Age' in df.columns:
    df['Age'] = df['Age'].fillna(df['Age'].median())

if 'Embarked' in df.columns:
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

if 'Class' in df_clean.columns:
    cols = [c for c in df_clean.columns if c != 'Class'] + ['Class']
    df_clean = df_clean[cols]

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Remaining missing values: {df_clean.isnull().sum().sum()}")
print("Columns preview:")
print(df_clean.columns.tolist())

output_path = '../data/clean/Clean_Titanic-Dataset.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nProcessed file saved to: {output_path}")