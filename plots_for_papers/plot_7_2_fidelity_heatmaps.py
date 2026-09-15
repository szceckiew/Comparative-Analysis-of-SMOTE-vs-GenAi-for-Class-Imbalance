import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import math

def plot_7_2_fidelity(summary_csv_path, output_dir):
    """
    Generates heatmaps and bar plots for data fidelity metrics
    (Wasserstein Mean, KL Divergence, etc.) to evaluate synthetic data quality.
    """
    print(f"Loading metrics analysis from: {summary_csv_path}")
    df = pd.read_csv(summary_csv_path)

    desired_metrics = {
        'Wasserstein_Mean': 'Wasserstein Distance',
        'KL_Divergence_Mean': 'KL Divergence',
        'Pairwise_Real_Synth': 'Real-Synth Pairwise Distance',
        'Wasserstein_vs_TRUE_Missing': 'Wasserstein vs DROPPED Orig',
    }

    available_metrics = {}
    for col, label in desired_metrics.items():
        if col in df.columns:
            available_metrics[col] = label
            
    if not available_metrics:
        print(f"No suitable columns found in: {df.columns}")
        print("Error: The script will not generate any plots.")
        return

    if 'Dataset' in df.columns:
        def clean_dataset_name(name):
            str_name = str(name)
            if str_name not in ['heart_balanced', 'iqr_heart_balanced', 'balanced_artificial_dataset', 'iqr_balanced_artificial_dataset']:
                str_name = re.sub(r'(?i)clean', '', str_name)
                str_name = re.sub(r'(?i)dataset', '', str_name)
                str_name = re.sub(r'(?i)balanced', '', str_name)
            
            str_name = str_name.replace('_', ' ').replace('-', ' ')
            return re.sub(r'\s+', ' ', str_name).strip()
            
        df['Dataset'] = df['Dataset'].apply(clean_dataset_name)

        def map_display_name(cleaned_name: str) -> str:
            s = str(cleaned_name).lower()
            if 'blood' in s:
                return 'Blood'
            if 'breast' in s and 'cancer' in s:
                return 'Breast Cancer Wisconsin'
            if 'covid' in s:
                return 'Covid-19'
            if 'credit' in s and 'card' in s:
                return 'Credit Card Fraud'
            if 'ddos' in s or 'cicids' in s:
                return 'DDoS'
            if 'haberman' in s:
                return "Haberman's Survival"
            if 'heart' in s:
                return 'Heart Disease'
            if 'infiltration' in s:
                return 'Infiltration'
            if 'ionosphere' in s:
                return 'Ionosphere'
            if 'loan' in s:
                return 'Loan Default'
            if 'nsl' in s or 'nsl kdd' in s:
                return 'NSL-KDD'
            if 'pima' in s or 'diabetes' in s:
                return 'Pima Indians Diabetes'
            if 'shuttle' in s:
                return 'Shuttle'
            if 'titanic' in s:
                return 'Titanic'
            if 'weather' in s or 'weatheraus' in s:
                return 'WeatherAUS'
            return str(cleaned_name).title()

        df['Dataset'] = df['Dataset'].apply(map_display_name)

    if 'Method' in df.columns:
        def clean_method_name(name):
            name = str(name).replace('-test_results', '').replace('unknown', 'baseline')
            
            name_map = {
                'gan-tutorial': 'gan 1',
                'gan_from_tutorial': 'gan 1',
                'gan from tutorial': 'gan 1',
                'gan-paper_2': 'gan 2',
                'gan-paper': 'gan 2*'
            }
            
            for key in sorted(name_map.keys(), key=len, reverse=True):
                pattern_str = re.sub(r'[-_]', r'[-_]', key)
                if re.search(pattern_str, name, flags=re.IGNORECASE):
                    name = re.sub(pattern_str, name_map[key], name, flags=re.IGNORECASE)
                    break
                    
            name = name.replace('_', ' ').replace('-', ' ')
            return re.sub(r'\s+', ' ', name).strip()

        df['Method'] = df['Method'].apply(clean_method_name)
        
        df = df[df['Method'] != 'baseline'].copy()

    plot_df = df.copy()
    
    if plot_df.empty:
        print("No data available to plot (all records might be baseline).")
        return

    distance_cmap = 'YlOrRd'

    for col, cbar_label in available_metrics.items():
        plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
        pivot_data = plot_df.pivot_table(index="Method", columns="Dataset", values=col, aggfunc='mean')
        
        safe_name = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        plt.figure(figsize=(24, 10)) 
        
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt=".2f", 
            annot_kws={"size": 8}, 
            cmap=distance_cmap, 
            cbar_kws={'label': cbar_label}
        )
        
        plt.ylabel('Augmentation Method', fontsize=14)
        plt.xlabel('Dataset', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        heat_out = os.path.join(output_dir, f'7_2_{safe_name}_heatmap.pdf')
        plt.savefig(heat_out, dpi=300)
        plt.close()
        print(f"Saved {col} heatmap to: {heat_out}")

    first_col = 'Wasserstein_Mean'
    if first_col in available_metrics:
        sample_datasets = plot_df['Dataset'].unique()
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
            data_subset = plot_df[plot_df['Dataset'] == dataset]
            
            sns.barplot(data=data_subset, y='Method', x=first_col, hue='Method', ax=ax, palette=distance_cmap, orient="h", legend=False)
            ax.set_title(f'{dataset}', fontsize=14)
            ax.set_ylabel('')
            ax.set_xlabel('Wasserstein Distance')
            
        for j in range(len(sample_datasets), len(axes)):
            fig.delaxes(axes[j])
            
        plt.tight_layout()
        bar_out = os.path.join(output_dir, '7_2_wasserstein_horizontal_bars.pdf')
        plt.savefig(bar_out, dpi=300)
        plt.close()
        print(f"Saved horizontal bar plot to: {bar_out}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate 7.2 fidelity plots.")
    parser.add_argument("--artificial", action="store_true", help="Generate for artificial_imbalance run")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.artificial:
        summary_path = os.path.join(base_dir, 'utils', 'ALL_DATASETS_ARTIFICIAL_FIDELITY_SUMMARY.csv')
        out_dir = os.path.join(base_dir, 'plots_for_papers', 'artificial_runs')
    else:
        summary_path = os.path.join(base_dir, 'utils', 'ALL_DATASETS_FIDELITY_SUMMARY.csv')
        out_dir = os.path.join(base_dir, 'plots_for_papers')
        
    os.makedirs(out_dir, exist_ok=True)
    
    if os.path.exists(summary_path):
        plot_7_2_fidelity(summary_path, out_dir)
    else:
        print(f"Summary file does not exist yet: {summary_path}. Generate the summary first.")