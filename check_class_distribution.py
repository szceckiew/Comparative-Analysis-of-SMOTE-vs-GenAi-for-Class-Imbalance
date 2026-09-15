import os
import pandas as pd
import argparse

def check_class_distribution(folder_path):
    if not os.path.exists(folder_path):
        print(f"ERROR: Directory '{folder_path}' does not exist.")
        return

    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No .csv files found in directory '{folder_path}'.")
        return

    print(f"{'Dataset Name':<40} | {'Dimensions':<12} | {'Class Distribution (Target)'}")
    print("-" * 100)

    for file in sorted(csv_files):
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            
            # Attempt to infer the target column using common naming conventions
            target_col = None
            common_target_names = ['target', 'Class', 'class', 'Label', 'label', 'TARGET']
            for name in common_target_names:
                if name in df.columns:
                    target_col = name
                    break
            
            # Fall back to assuming the last column is the target
            if target_col is None:
                target_col = df.columns[-1]
            
            # Calculate class distribution
            counts = df[target_col].value_counts().to_dict()
            
            # Format for console output
            counts_str = ", ".join([f"Class {k}: {v}" for k, v in counts.items()])
            shape_str = f"{df.shape[0]}x{df.shape[1]}"
            
            print(f"{file:<40} | {shape_str:<12} | {target_col} -> {counts_str}")
            
        except Exception as e:
            print(f"{file:<40} | Error reading file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check class distributions for all datasets in the specified directory.")
    parser.add_argument("--folder", type=str, default="data", 
                        help="Path to the directory containing .csv files (default: 'data')")
    
    args = parser.parse_args()
    
    print(f"Checking directory: {args.folder}\n")
    check_class_distribution(args.folder)
