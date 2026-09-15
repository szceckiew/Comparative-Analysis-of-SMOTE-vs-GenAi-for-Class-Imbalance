import pandas as pd

input_path = '../data/raw/heart.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")

if df.duplicated().sum() > 0:
    print(f"Found {df.duplicated().sum()} duplicates. Removing...")
    df = df.drop_duplicates()

if 'HeartDisease' in df.columns:
    df = df.rename(columns={'HeartDisease': 'Class'})

df_clean = pd.get_dummies(df, drop_first=True, dtype=int)

if 'Class' in df_clean.columns:
    cols = [c for c in df_clean.columns if c != 'Class'] + ['Class']
    df_clean = df_clean[cols]

print("-" * 30)
print(f"Cleaned dataset shape: {df_clean.shape}")
print("Sample columns:")
print(df_clean.columns.tolist()[:10])
print(f"Target class distribution:\n{df_clean['Class'].value_counts()}")

output_path = '../data/clean/Clean_heart.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nProcessed dataset saved to: {output_path}")