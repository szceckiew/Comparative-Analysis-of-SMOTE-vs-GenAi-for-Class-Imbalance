import pandas as pd

input_path = '../data/raw/shuttle_full_dataset.csv'
df = pd.read_csv(input_path)

print(f"Original shape: {df.shape}")
print("Current class distribution:")
counts = df['Class'].value_counts().sort_index()
print(counts)

majority_class = counts.idxmax()
print(f"\nMajority class (Normal): {majority_class} (Count: {counts[majority_class]})")

df['Class'] = df['Class'].apply(lambda x: 0 if x == majority_class else 1)

print("\nNew class distribution (0=Normal, 1=Anomaly):")
print(df['Class'].value_counts())

output_path = '../data/clean/Clean_shuttle_full_dataset.csv'
df.to_csv(output_path, index=False)
print("-" * 30)
print(f"Binarized file saved to: {output_path}")
print("SMOTE will no longer fail due to insufficient nearest neighbors.")