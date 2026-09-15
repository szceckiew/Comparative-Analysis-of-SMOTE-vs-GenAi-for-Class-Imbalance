import pandas as pd
import glob
import os
import numpy as np


def analyze_all_datasets(directory_path='.',
                         output_filename='full_metrics_summary.csv',
                         recursive=False):
    """
    Analyzes all CSV files in a directory, detects dataset names,
    aggregates results, and saves them to a single summary file.
    """
    output_path_local = os.path.abspath(os.path.join('.', output_filename))
    if recursive:
        search_path = os.path.join(directory_path, '**', '*test_results.csv')
        all_files = glob.glob(search_path, recursive=True)
    else:
        search_path = os.path.join(directory_path, '*test_results.csv')
        all_files = glob.glob(search_path)

    file_list = []

    print(f"Scanning directory: {directory_path}")

    for f in all_files:
        abs_path = os.path.abspath(f)
        filename = os.path.basename(f)

        if abs_path == output_path_local:
            continue
        if filename == output_filename:
            continue
        if "summary" in filename.lower():
            continue

        file_list.append(f)

    if not file_list:
        print(f"Error: No result files found in: {directory_path}")
        return

    print(f"Found {len(file_list)} result files. Processing...")

    all_stats_numeric = []
    all_numeric_cols = []

    for file_path in file_list:
        file_name = os.path.basename(file_path)

        base_name = file_name.replace('.csv', '').replace('-test_results', '')

        is_iqr = base_name.lower().startswith('iqr_')
        if is_iqr:
            base_name = base_name[4:]

        parts = base_name.rsplit('-', 2)
        if len(parts) == 3:
            dataset_part = parts[0]
            method_part = parts[1]
            classifier_part = parts[2]
        elif len(parts) == 2:
            dataset_part = parts[0]
            method_part = parts[1]
            classifier_part = "unknown"
        else:
            dataset_part = base_name
            method_part = "unknown"
            classifier_part = "unknown"

        is_filtered = '_filtered' in method_part.lower()
        if is_filtered:
            method_part = method_part.replace('_filtered', '')

        dataset_name = dataset_part.replace('Clean_', '')
        balancing_method = method_part.replace('_from_tutorial', '-tutorial').replace('_paper', '-paper')

        stats_row = {
            'Dataset Name': dataset_name,
            'Case File': file_name,
            'IQR Used': is_iqr,
            'Filtered': is_filtered,
            'Method': balancing_method,
            'Classifier': classifier_part
        }

        try:
            df = pd.read_csv(file_path)
            df_numeric = df.select_dtypes(include=['number'])

            if df_numeric.empty:
                continue

            for col in df_numeric.columns:
                if col not in all_numeric_cols:
                    all_numeric_cols.append(col)

            means = df_numeric.mean()
            if len(df_numeric) > 1:
                stds = df_numeric.std()
            else:
                stds = df_numeric.mean() * 0.0

            for col in df_numeric.columns:
                stats_row[f'{col} (Mean)'] = means.get(col)
                stats_row[f'{col} (Std)'] = stds.get(col)

            all_stats_numeric.append(stats_row)

        except Exception as e:
            print(f"    [SKIP] Error processing file {file_name}: {e}")

    if not all_stats_numeric:
        print("\nNo data available to generate the report.")
        return None

    summary_df = pd.DataFrame(all_stats_numeric)

    sort_metrics = ['Dataset Name']
    ascending_order = [True]

    if 'MCC (Mean)' in summary_df.columns:
        sort_metrics.append('MCC (Mean)')
        ascending_order.append(False)
    elif 'F1-score (Malignant) (Mean)' in summary_df.columns:
        sort_metrics.append('F1-score (Malignant) (Mean)')
        ascending_order.append(False)

    summary_df = summary_df.sort_values(by=sort_metrics, ascending=ascending_order)

    metadata_cols = ['Dataset Name', 'Method', 'IQR Used', 'Filtered']
    formatted_rows = []

    for _, row in summary_df.iterrows():
        new_row = {col: row[col] for col in metadata_cols}

        for metric in all_numeric_cols:
            mean_val = row.get(f'{metric} (Mean)')
            std_val = row.get(f'{metric} (Std)')

            if pd.isna(mean_val):
                new_row[metric] = "-"
            else:
                if pd.isna(std_val):
                    std_val = 0.0
                new_row[metric] = f"{mean_val:.4f} (± {std_val:.4f})"

        formatted_rows.append(new_row)

    final_df = pd.DataFrame(formatted_rows)

    priority_metrics = ['MCC', 'F1-score (Malignant)', 'AUC (Binary)', 'Recall (Malignant)']
    existing_priority = [m for m in priority_metrics if m in final_df.columns]
    other_metrics = [c for c in final_df.columns if c not in metadata_cols and c not in existing_priority]
    other_metrics.sort()

    final_cols = metadata_cols + existing_priority + other_metrics
    final_df = final_df[final_cols]

    final_df.to_csv(output_path_local, index=False, encoding='utf-8')
    print(f"\nSaved full report to: {output_filename}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'left')

    unique_datasets = final_df['Dataset Name'].unique()

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY ({len(unique_datasets)} Datasets)")
    print("=" * 80)

    for dataset in unique_datasets:
        print(f"\n>>> DATASET: {dataset}")
        print("-" * 80)

        dataset_subset = final_df[final_df['Dataset Name'] == dataset]

        print_cols = [c for c in final_cols if c != 'Dataset Name']

        print(dataset_subset[print_cols].to_string(index=False))
        print("-" * 80)

    return final_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze metrics from CSV files into a unified summary.")
    parser.add_argument("--artificial", action="store_true", help="Analyze results from results_artificial_imbalance")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.artificial:
        input_directory = os.path.join(script_dir, "..", "results_artificial_imbalance", "cv_results")
        output_name = os.path.join(script_dir, "ALL_DATASETS_ARTIFICIAL_SUMMARY.csv")
    else:
        input_directory = os.path.join(script_dir, "..", "results", "cv_results")
        output_name = os.path.join(script_dir, "ALL_DATASETS_SUMMARY.csv")

    analyze_all_datasets(
        directory_path=input_directory,
        output_filename=output_name,
        recursive=True
    )