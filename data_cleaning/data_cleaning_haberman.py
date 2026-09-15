import pandas as pd

input_path = '../data/raw/haberman.csv'
df = pd.read_csv(input_path, header=None)

df.columns = ['Age', 'Op_Year', 'Axil_Nodes', 'Class']

df['Class'] = df['Class'].map({1: 0, 2: 1})

print("Class distribution after mapping (0=survived, 1=died):")
print(df['Class'].value_counts())

output_path = '../data/clean/Clean_haberman.csv'
df.to_csv(output_path, index=False)

print(f"Saved processed file to: {output_path}")