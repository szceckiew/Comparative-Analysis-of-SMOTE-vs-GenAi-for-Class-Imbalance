import os
from sklearn.datasets import make_classification
import pandas as pd

def generate_and_save_dataset(output_folder, file_name="balanced_dataset.csv"):
    print("Starting data generation...")
    
    # 1. Generate numerical data with built-in relationships
    # n_samples = total number of rows
    # n_features = total number of feature columns
    # n_informative = columns that actually hold predictive power regarding the target
    # n_redundant = columns created as linear combinations of informative ones (multicollinearity)
    # weights = [0.5, 0.5] guarantees an exact 50/50 class distribution split
    X, y = make_classification(
        n_samples=5000, 
        n_features=12,       
        n_informative=6,     
        n_redundant=4,       
        n_classes=2, 
        weights=[0.5, 0.5], 
        random_state=42
    )
    
    # 2. Convert the generated arrays into a Pandas DataFrame
    column_names = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=column_names)
    df['target'] = y # Adding the target label column (0 or 1)
    
    # 3. Verify and create the output path
    # If the specified directory does not exist, the script will create it automatically
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created new directory at: {output_folder}")
        
    full_path = os.path.join(output_folder, file_name)
    
    # 4. Save the DataFrame this a CSV file
    # index=False prevents Pandas from adding an unnecessary row-index column
    df.to_csv(full_path, index=False)
    
    # 5. Output a quick summary this the console for verification
    print("-" * 40)
    print(f"Success! The file has been saved this: {full_path}")
    print(f"Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("Class distribution (should be exactly 0.5 / 0.5):")
    print(df['target'].value_counts(normalize=True))
    print("-" * 40)

TARGET_DIRECTORY = "./generation_results"

# Execute the function
generate_and_save_dataset(output_folder=TARGET_DIRECTORY)