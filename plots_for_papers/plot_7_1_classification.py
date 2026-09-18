import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math

def plot_7_1_classification(summary_csv_path, output_dir):
    print(f"Loading results analysis from: {summary_csv_path}")
    df = pd.read_csv(summary_csv_path)

    desired_metrics = {
        'MCC': 'MCC Comparison (Average over Classifiers)',
        'Recall (Malignant)': 'Recall (Class 1) Comparison',
        'F1-score (Malignant)': 'F1-score (Class 1) Comparison',
        'Balanced_Accuracy': 'Balanced Accuracy Comparison',
        'PR_AUC': 'PR AUC Comparison',
        'AUC (Binary)': 'AUC (Binary) Comparison',
    }

    available_metrics = {}
    for col, title in desired_metrics.items():
        if f"{col} (Mean)" in df.columns:
            available_metrics[f"{col} (Mean)"] = title
        elif col in df.columns:
            available_metrics[col] = title
            
    if not available_metrics:
        print(f"No suitable columns found in: {df.columns}")
        print("Error: The script will not generate any plots.")
        return

    subset = df.copy()

    def clean_dataset_name(name):
        str_name = str(name)
        is_iqr = bool(re.search(r'(?i)iqr', str_name))
        
        str_name = re.sub(r'(?i)clean', '', str_name)
        str_name = re.sub(r'(?i)dataset', '', str_name)
        str_name = re.sub(r'(?i)balanced', '', str_name)
        str_name = re.sub(r'(?i)iqr', '', str_name)
        str_name = str_name.replace('_', ' ').replace('-', ' ').strip().lower()
        
        base_name = ""
        if 'blood' in str_name:
            base_name = 'Blood'
        elif 'breast' in str_name and 'cancer' in str_name:
            base_name = 'Breast Cancer Wisconsin'
        elif 'covid' in str_name:
            base_name = 'Covid-19'
        elif 'credit' in str_name and 'card' in str_name:
            base_name = 'Credit Card Fraud'
        elif 'ddos' in str_name or 'cicids' in str_name:
            base_name = 'DDoS'
        elif 'haberman' in str_name:
            base_name = "Haberman's Survival"
        elif 'heart' in str_name:
            base_name = 'Heart Disease'
        elif 'infiltration' in str_name:
            base_name = 'Infiltration'
        elif 'ionosphere' in str_name:
            base_name = 'Ionosphere'
        elif 'loan' in str_name:
            base_name = 'Loan Default'
        elif 'madelon' in str_name or 'mdlon' in str_name:
            base_name = 'Madelon'
        elif 'nsl' in str_name or 'nsl kdd' in str_name:
            base_name = 'NSL-KDD'
        elif 'pima' in str_name or 'diabetes' in str_name:
            base_name = 'Pima Indians Diabetes'
        elif 'shuttle' in str_name:
            base_name = 'Shuttle'
        elif 'titanic' in str_name:
            base_name = 'Titanic'
        elif 'weather' in str_name or 'weatheraus' in str_name:
            base_name = 'WeatherAUS'
        else:
            base_name = ' '.join([w.capitalize() for w in str_name.split()])
            
        if is_iqr:
            return f"IQR {base_name}"
        return base_name

    subset['Dataset Name'] = subset['Dataset Name'].apply(clean_dataset_name)
    sample_datasets = subset['Dataset Name'].unique()

    def parse_mean_from_str(val):
        if pd.isna(val) or val == "-":
            return float('nan')
        if isinstance(val, str) and "(" in val:
            return float(val.split("(")[0].strip())
        return float(val)

    for col in available_metrics.keys():
        subset[col] = subset[col].apply(parse_mean_from_str)

    def clean_method_name(row):
        name = str(row['Method']).replace('-test_results', '').replace('unknown', 'baseline')
        
        name_map = {
            'gan-tutorial': 'gan 1',
            'gan from tutorial': 'gan 1',
            'gan-paper_2': 'gan 2',
            'gan-paper': 'gan 2*'
        }
        
        for key in sorted(name_map.keys(), key=len, reverse=True):
            pattern_str = re.sub(r'[-_]', r'[-_]', key)
            if re.search(pattern_str, name, flags=re.IGNORECASE):
                name = re.sub(pattern_str, name_map[key], name, flags=re.IGNORECASE)
                break
                
        if row.get('Filtered') == True:
            name += "_filtered"
        if row.get('IQR Used') == True:
            name = "IQR + " + name
            
        name = name.replace('_', ' ').replace('-', ' ')
        name = re.sub(r'\s+', ' ', name).strip()
        
        return name

    subset['Augmentation Method'] = subset.apply(clean_method_name, axis=1)

    plot_df = subset.groupby(['Dataset Name', 'Augmentation Method'])[list(available_metrics.keys())].mean().reset_index()

    os.makedirs(output_dir, exist_ok=True)

    plot_df['is_iqr'] = plot_df['Dataset Name'].str.startswith('IQR')
    plot_df = plot_df.sort_values(by=['is_iqr', 'Dataset Name', 'Augmentation Method'])
    plot_df = plot_df.drop(columns=['is_iqr'])

    sea_cmap = 'YlGnBu'

    def clean_colorbar_label(title_str):
        t = title_str.replace(' Comparison', '')
        t = re.sub(r'\(.*?\)', '', t)
        words = t.split()
        cleaned_words = []
        for w in words:
            if w.isupper():
                cleaned_words.append(w)
            else:
                cleaned_words.append(w.lower())
        return " ".join(cleaned_words)

    for col, title in available_metrics.items():
        plt.figure(figsize=(18, 10)) 
        pivot_data = plot_df.pivot(index="Augmentation Method", columns="Dataset Name", values=col)
        
        safe_name = col.replace(" (Mean)", "").replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        clean_label = clean_colorbar_label(title)
        
        sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap=sea_cmap, vmin=0.0, vmax=1.0, cbar_kws={'label': clean_label})
        plt.ylabel('Augmentation Method', fontsize=14)
        plt.xlabel('Dataset', fontsize=14)
        plt.tight_layout()
        
        heat_out = os.path.join(output_dir, f'7_1_{safe_name}_heatmap.pdf')
        plt.savefig(heat_out, dpi=300)
        plt.close()
        print(f"Saved {col} heatmap to: {heat_out}")

    first_col = list(available_metrics.keys())[0]

    n_datasets = len(sample_datasets)
    cols = min(2, n_datasets)
    rows = math.ceil(n_datasets / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows), sharex=True)
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, dataset in enumerate(sample_datasets):
        ax = axes[i]
        data_subset = plot_df[plot_df['Dataset Name'] == dataset]
        
        sns.barplot(data=data_subset, y='Augmentation Method', x=first_col, hue='Augmentation Method', ax=ax, palette=sea_cmap, orient="h", legend=False)
        ax.set_title(f'{dataset}', fontsize=14)
        ax.set_ylabel('')
        ax.set_xlabel('Score')
        ax.set_xlim(0.0, 1.0)
        
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    bar_out = os.path.join(output_dir, '7_1_horizontal_bars.pdf')
    plt.savefig(bar_out, dpi=300)
    plt.close()
    print(f"Saved horizontal bar plot to: {bar_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 7.1 classification plots.")
    parser.add_argument("--artificial", action="store_true", help="Generate plots for the artificial downsampling run")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.artificial:
        summary_path = os.path.join(base_dir, 'utils', 'ALL_DATASETS_ARTIFICIAL_SUMMARY.csv')
        out_dir = os.path.join(base_dir, 'plots_for_papers', 'artificial_runs')
    else:
        summary_path = os.path.join(base_dir, 'utils', 'ALL_DATASETS_SUMMARY.csv')
        out_dir = os.path.join(base_dir, 'plots_for_papers')
        
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(summary_path):
        plot_7_1_classification(summary_path, out_dir)
    else:
        print(f"Summary file does not exist yet: {summary_path}. Run the reporting script first.")