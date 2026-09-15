import pandas as pd
import numpy as np
import os

files_to_process = {
    'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv': 'Clean_DDos.csv',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv': 'Clean_Infiltration.csv'
}

base_input_dir = '../data/raw/'
base_output_dir = '../data/clean/'


def clean_and_save(input_filename, output_filename):
    input_path = os.path.join(base_input_dir, input_filename)
    output_path = os.path.join(base_output_dir, output_filename)

    print(f"\n---> Processing file: {input_filename}")

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        return

    df = pd.read_csv(input_path, low_memory=False)
    print(f"   Original shape: {df.shape}")

    df.columns = df.columns.str.strip().str.replace(' ', '_')

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'Class'})

    df['Class'] = df['Class'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    counts = df['Class'].value_counts()
    print(f"   Class distribution (0=Benign, 1=Attack):\n{counts}")

    initial_rows = df.shape[0]
    df = df.drop_duplicates()
    print(f"   Duplicates removed: {initial_rows - df.shape[0]}")

    os.makedirs(base_output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"   Saved to: {output_path}")


for in_file, out_file in files_to_process.items():
    clean_and_save(in_file, out_file)

print("\n" + "=" * 30)
print("Finished processing all datasets.")