import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

def plot_meta_features(csv_path):
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist. Ensure meta-features have been generated.")
        return
        
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("CSV file is empty.")
        return
        
    numeric_df = df.drop(columns=['Dataset_Name']) if 'Dataset_Name' in df.columns else df
    
    numeric_df = numeric_df.dropna(axis=1, how='all')
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]
    
    feature_name_mapping = {
        'n_samples': 'Number of Samples',
        'n_features': 'Number of Features',
        'Minority_Ratio': 'Minority Ratio',
        'Imbalance_Ratio': 'Imbalance Ratio',
        'Samples_Per_Feature': 'Samples per Feature',
        'Avg_Correlation': 'Average Correlation',
        'Linearity_AUC': 'Linear AUC',
        'NonLinear_AUC': 'Nonlinear AUC',
        'Mean_Skewness': 'Mean Skewness',
        'Minority_Overlap_Ratio': 'MCOR',
        'Minority_Avg_Distance': 'Minority Avg. Distance'
    }
    
    numeric_df.rename(columns=feature_name_mapping, inplace=True)
    
    plt.figure(figsize=(12, 10))
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, square=True, linewidths=.5)
    plt.tight_layout()
    
    output_dir = os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.'
    plt.savefig(os.path.join(output_dir, 'meta_features_correlation_heatmap.pdf'))

    zscore_df = (numeric_df - numeric_df.mean()) / numeric_df.std()
    
    annot_df = pd.DataFrame(index=numeric_df.index, columns=numeric_df.columns)
    for col in numeric_df.columns:
        for idx in numeric_df.index:
            val = numeric_df.at[idx, col]
            if pd.isna(val):
                annot_df.at[idx, col] = ""
            elif col in ['Number of Samples', 'Number of Features']:
                annot_df.at[idx, col] = f"{int(val)}"
            else:
                annot_df.at[idx, col] = f"{val:.4f}"
    
    if 'Dataset_Name' in df.columns:
        def clean_dataset_name(name):
            str_name = str(name).replace('.csv', '')
            
            str_name = re.sub(r'(?i)clean', '', str_name)
            str_name = re.sub(r'(?i)dataset', '', str_name)
            str_name = re.sub(r'(?i)full', '', str_name)
            
            str_name = str_name.replace('_', ' ').replace('-', ' ')
            
            str_name = re.sub(r'\s+', ' ', str_name).strip()
            
            str_name = str_name.title()
            
            str_name = re.sub(r'(?i)nsl kdd', 'NSL KDD', str_name)
            str_name = re.sub(r'(?i)ddos', 'DDOS', str_name)
            
            return str_name
            
        dataset_names = df['Dataset_Name'].apply(clean_dataset_name)
        zscore_df.index = dataset_names
        annot_df.index = dataset_names
    
    plt.figure(figsize=(20, max(12, len(df) * 0.5)))
    sns.heatmap(zscore_df, cmap='coolwarm', center=0, annot=annot_df, fmt="", linewidths=.5, annot_kws={"size": 10},
                cbar_kws={'label': 'z-score (Standard Deviations from Mean)'})
    plt.ylabel('Dataset')
    plt.xlabel('Meta-feature')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'meta_features_by_dataset_heatmap.pdf'))
    
    print(f"Plots saved to directory: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    csv_file = './meta_features_dataset.csv'
    
    if not os.path.exists(csv_file):
        csv_file = 'meta_features_dataset.csv'
        
    plot_meta_features(csv_file)