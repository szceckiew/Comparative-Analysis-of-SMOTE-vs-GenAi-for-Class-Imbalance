import pandas as pd
import os

def process_spambase():
    raw_path = 'data/raw/spambase/spambase.data'
    out_path = 'data/clean/Clean_spambase.csv'
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return
        
    print("Processing Spambase...")
    
    columns = [
        "word_freq_make", "word_freq_address", "word_freq_all", "word_freq_3d", "word_freq_our", "word_freq_over",
        "word_freq_remove", "word_freq_internet", "word_freq_order", "word_freq_mail", "word_freq_receive",
        "word_freq_will", "word_freq_people", "word_freq_report", "word_freq_addresses", "word_freq_free",
        "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit", "word_freq_your",
        "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl", "word_freq_george",
        "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data",
        "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
        "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project",
        "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(",
        "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average",
        "capital_run_length_longest", "capital_run_length_total", "Class"
    ]
    
    df = pd.read_csv(raw_path, header=None, names=columns)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

def process_parkinsons():
    raw_path = 'data/raw/parkinsons/parkinsons.data'
    out_path = 'data/clean/Clean_parkinsons.csv'
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return
        
    print("Processing Parkinsons...")
    
    df = pd.read_csv(raw_path)
    
    # Remove patient/sample ID identifier column
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
        
    # Rename classification target to standard 'Class'
    if 'status' in df.columns:
        df = df.rename(columns={'status': 'Class'})
        
    # Reorder columns to place 'Class' at the end to match project conventions
    if 'Class' in df.columns:
        cols = [c for c in df.columns if c != 'Class'] + ['Class']
        df = df[cols]
        
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

def process_sonar():
    raw_path = 'data/raw/sonar.csv'
    out_path = 'data/clean/Clean_sonar.csv'
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return
        
    print("Processing Sonar...")
    
    # Raw file does not contain header
    df = pd.read_csv(raw_path, header=None)
    
    # Assign column names
    feature_cols = [f"Feature_{i}" for i in range(1, df.shape[1])]
    df.columns = feature_cols + ["Class"]
    
    # Map string labels 'M' (Mine), 'R' (Rock) to binary values 0/1
    # Setting M=1, R=0
    df['Class'] = df['Class'].map({'M': 1, 'R': 0})
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

def process_mdlon():
    raw_path = 'data/raw/mdlon.csv'
    out_path = 'data/clean/Clean_mdlon.csv'
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return
        
    print("Processing Madelon...")
    
    df = pd.read_csv(raw_path)
    
    # Rename target column from 'T' to standard 'Class'
    if 'T' in df.columns:
        df = df.rename(columns={'T': 'Class'})
        
    # Map target values from (-1, 1) to (0, 1) for scikit-learn pipeline compatibility
    df['Class'] = df['Class'].map({-1: 0, 1: 1})
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

def process_banknote():
    raw_path = 'data/raw/data_banknote_authentication.csv'
    out_path = 'data/clean/Clean_banknote.csv'
    
    if not os.path.exists(raw_path):
        print(f"File not found: {raw_path}")
        return
        
    print("Processing Banknote Authentication...")
    
    df = pd.read_csv(raw_path)
    # The dataset already has columns N1, N2, N3, N4, Class and classes are 0,1
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")

if __name__ == "__main__":
    print("--- Converting raw data to clean CSVs ---")
    process_spambase()
    print("-" * 40)
    process_parkinsons()
    print("-" * 40)
    process_sonar()
    print("-" * 40)
    process_mdlon()
    print("-" * 40)
    process_banknote()