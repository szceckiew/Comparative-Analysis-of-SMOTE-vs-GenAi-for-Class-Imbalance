import pandas as pd
import glob
import os
import sys

RELATIVE_CSV_INPUT_FOLDER = "../results/cv_results/dataset_metrics"
RELATIVE_OUTPUT_FOLDER = "."
output_filename = "ALL_DATASETS_FIDELITY_SUMMARY.csv"

metric_columns = [
    'wasserstein_mean',
    'kl_divergence_mean',
    'pairwise_real_real',
    'pairwise_synth_synth',
    'pairwise_real_synth'
]

column_rename_map = {
    'wasserstein_mean': 'Wasserstein_Mean',
    'kl_divergence_mean': 'KL_Divergence_Mean',
    'pairwise_real_real': 'Pairwise_Real_Real',
    'pairwise_synth_synth': 'Pairwise_Synth_Synth',
    'pairwise_real_synth': 'Pairwise_Real_Synth'
}

def get_absolute_path(relative_path):
    try:
        script_dir = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        script_dir = os.getcwd()
    return os.path.normpath(os.path.join(script_dir, relative_path))

def analyze_metrics_in_folder(input_folder_path, output_folder_path, is_artificial=False):
    if not os.path.isdir(input_folder_path):
        print(f"\nERROR: Input folder does not exist: '{input_folder_path}'")
        sys.exit(1)

    os.makedirs(output_folder_path, exist_ok=True)
    print(f"Starting analysis of directory: {input_folder_path}\n")

    search_pattern = os.path.join(input_folder_path, "*.csv")
    csv_files = glob.glob(search_pattern)

    all_data = []
    for file_path in csv_files:
        if os.path.basename(file_path) == output_filename:
            continue
            
        try:
            df = pd.read_csv(file_path)
            if 'dataset' not in df.columns:
                base = os.path.basename(file_path).replace("_dataset_metrics.csv", "").replace("-dataset_metrics.csv", "")
                if 'balance_method' in df.columns:
                    method = df['balance_method'].iloc[0]
                    dataset = base.replace(f"{method}_", "").replace(f"-{method}", "").replace(f"{method}-", "")
                    df['dataset'] = dataset
                else:
                    df['dataset'] = "Unknown"
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    if not all_data:
        print("No valid data loaded.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    mask_metrics = [c for c in metric_columns if c in full_df.columns]
    
    grouped = full_df.groupby(['dataset', 'balance_method'])[mask_metrics].agg(['mean', 'std']).reset_index()
    grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
    
    renamed_df = grouped.rename(columns={"dataset": "Dataset", "balance_method": "Method"})

    for metric in mask_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in renamed_df.columns and std_col in renamed_df.columns:
            friendly_name = column_rename_map.get(metric, metric)
            renamed_df[f"{friendly_name}_String"] = renamed_df.apply(
                lambda row: f"{row[mean_col]:.4f} (± {row[std_col]:.4f})" if pd.notnull(row[mean_col]) else "NaN", 
                axis=1
            )
            renamed_df[friendly_name] = renamed_df[mean_col].apply(lambda x: round(x, 4) if pd.notnull(x) else x)
    
    if is_artificial:
        art_path = get_absolute_path("../artificial_imbalance_results.csv")
        if os.path.exists(art_path):
            try:
                art_df = pd.read_csv(art_path)
                if 'Method' in art_df.columns and 'Dataset' in art_df.columns:
                    art_df = art_df.rename(columns={'Wasserstein_Dist': 'Wasserstein_vs_TRUE_Missing'})
                    
                    renamed_df['Dataset_Key'] = renamed_df['Dataset'].astype(str).str.replace(".csv", "", regex=False)
                    art_df['Dataset_Key'] = art_df['Dataset'].astype(str).str.replace(".csv", "", regex=False)
                    
                    renamed_df = pd.merge(renamed_df, art_df, on=['Dataset_Key', 'Method'], how='left')
                    renamed_df = renamed_df.drop(columns=['Dataset_Key', 'Dataset_y'])
                    if 'Dataset_x' in renamed_df.columns:
                        renamed_df = renamed_df.rename(columns={'Dataset_x': 'Dataset'})
                        
                    print(f"--> Successfully merged artificial imbalance test statistics from: {art_path}")
            except Exception as e:
                print(f"Error merging with artificial_imbalance_results.csv: {e}")

    output_filepath = os.path.join(output_folder_path, output_filename)
    renamed_df.to_csv(output_filepath, index=False)
    print(f"\nSUCCESS! Created aggregated report with averaged values: {output_filepath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze dataset fidelity metrics.")
    parser.add_argument("--artificial", action="store_true", help="Use results from results_artificial_imbalance")
    args = parser.parse_args()

    if args.artificial:
        input_dir = "../results_artificial_imbalance/cv_results/dataset_metrics"
        out_name = "ALL_DATASETS_ARTIFICIAL_FIDELITY_SUMMARY.csv"
    else:
        input_dir = "../results/cv_results/dataset_metrics"
        out_name = "ALL_DATASETS_FIDELITY_SUMMARY.csv"

    output_filename = out_name

    target_input_path = get_absolute_path(input_dir)
    target_output_path = get_absolute_path(RELATIVE_OUTPUT_FOLDER)
    analyze_metrics_in_folder(target_input_path, target_output_path, is_artificial=args.artificial)