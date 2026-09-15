import pandas as pd
import os
import glob
from sklearn.preprocessing import MinMaxScaler
import numpy as np

FOLDER_TEST_RESULTS = '../results/cv_results/hgbc_tuning/test_results'
FOLDER_DATASET_METRICS = '../results/cv_results/dataset_metrics'
FILE_META_FEATURES = '../meta_features_dataset.csv'

OUTPUT_ALL_SCORES = 'all_methods_scores.csv'
OUTPUT_BEST_METHODS = 'best_methods_summary.csv'

COL_MCC = 'MCC'
COL_RECALL = 'Recall (Malignant)'
COL_KENDALL = 'Frobenius_Kendall_Diff'
COL_WASSERSTEIN = 'wasserstein_mean'
COL_KL = 'kl_divergence_mean'

WEIGHTS = {
    'mcc': 0.4,
    'recall': 0.1,
    'wasserstein': 0.3,
    'kendall': 0.1,
    'kl_div': 0.1
}

TOLERANCE = 0.025


def get_method_complexity(method_name):
    """
    Returns method complexity tier (lower means simpler/preferred in case of a tie).
    1 = SMOTE / Interpolation (Most lightweight, stable)
    2 = Simple GANs (Vanilla)
    3 = Advanced GANs (WGAN, CTGAN, Hybrids)
    """
    name = str(method_name).lower()

    if 'smote' in name and 'smotified' not in name:
        return 1

    if any(x in name for x in ['wgangp', 'ctgan', 'smotified']):
        return 3

    if 'gan' in name:
        return 2

    return 10


def parse_filename_smart(filename, suffix='-test_results.csv'):
    """
    Intelligently parses the filename using a hyphen delimiter.
    Handles '_filtered' and 'iqr_'.
    """
    if not filename.endswith(suffix):
        return None, None, None

    base_name = filename.replace(suffix, '')
    parts = base_name.rsplit('-', 1)

    if len(parts) != 2:
        return None, None, None

    dataset_raw = parts[0]
    method_raw = parts[1]

    if dataset_raw.startswith('iqr_'):
        dataset_clean = dataset_raw.replace('iqr_', '')
        method_final = f"iqr_{method_raw}"
    else:
        dataset_clean = dataset_raw
        method_final = method_raw

    if not dataset_clean.endswith('.csv'):
        dataset_final = dataset_clean + '.csv'
    else:
        dataset_final = dataset_clean

    return method_final, dataset_final, dataset_raw


def load_and_merge_data():
    all_data = []
    test_files = glob.glob(os.path.join(FOLDER_TEST_RESULTS, '*.csv'))
    print(f"Found {len(test_files)} test result files.")

    for test_file_path in test_files:
        filename = os.path.basename(test_file_path)
        method, dataset, original_dataset_prefix = parse_filename_smart(filename)

        if method is None:
            continue

        base_name_parts = filename.replace('-test_results.csv', '').rsplit('-', 1)
        metric_method_part = base_name_parts[1]
        metric_dataset_part = base_name_parts[0]
        metrics_file_name = f"{metric_method_part}_{metric_dataset_part}_dataset_metrics.csv"
        metrics_file_path = os.path.join(FOLDER_DATASET_METRICS, metrics_file_name)

        if not os.path.exists(metrics_file_path):
            continue

        try:
            df_test = pd.read_csv(test_file_path)
            df_metrics = pd.read_csv(metrics_file_path)

            record = {
                'Dataset': dataset,
                'Method': method,
                'MCC': df_test[COL_MCC].mean(),
                'Recall': df_test[COL_RECALL].mean(),
                'Kendall_Diff': df_test[COL_KENDALL].mean(),
                'Wasserstein': df_metrics[COL_WASSERSTEIN].mean(),
                'KL_Div': df_metrics[COL_KL].mean()
            }
            all_data.append(record)
        except Exception as e:
            print(f"[ERROR] Error in {filename}: {e}")

    return pd.DataFrame(all_data)


def calculate_scores(df):
    df_scored = df.copy()
    metrics_config = {
        'MCC': True, 'Recall': True, 'Kendall_Diff': False,
        'Wasserstein': False, 'KL_Div': False
    }

    for dataset_name, group in df_scored.groupby('Dataset'):
        for metric, higher_is_better in metrics_config.items():
            values = group[metric].values.reshape(-1, 1)
            scaler = MinMaxScaler()

            if len(values) > 0:
                scaled_values = scaler.fit_transform(values).flatten()
            else:
                scaled_values = np.zeros(len(values))

            if not higher_is_better:
                scaled_values = 1 - scaled_values

            df_scored.loc[group.index, f'Norm_{metric}'] = scaled_values

    df_scored = df_scored.fillna(0.5)
    df_scored['Final_Score'] = (
            (df_scored['Norm_MCC'] * WEIGHTS['mcc']) +
            (df_scored['Norm_Recall'] * WEIGHTS['recall']) +
            (df_scored['Norm_Wasserstein'] * WEIGHTS['wasserstein']) +
            (df_scored['Norm_KL_Div'] * WEIGHTS['kl_div']) +
            (df_scored['Norm_Kendall_Diff'] * WEIGHTS['kendall'])
    )
    return df_scored


def select_best_method_smartly(df_scores, tolerance=TOLERANCE):
    """
    Selects the best method considering tolerance and complexity.
    Occam's razor: If performance scores are close, choose the simpler method.
    """
    best_methods_list = []

    for dataset_name, group in df_scores.groupby('Dataset'):
        max_score = group['Final_Score'].max()

        candidates = group[group['Final_Score'] >= (max_score - tolerance)].copy()

        candidates['Complexity'] = candidates['Method'].apply(get_method_complexity)

        winner = candidates.sort_values(by=['Complexity', 'Final_Score'], ascending=[True, False]).iloc[0]

        is_optimized = (winner['Final_Score'] < max_score)

        winner_record = {
            'Dataset': winner['Dataset'],
            'Best_Method': winner['Method'],
            'Final_Score': winner['Final_Score'],
            'Selection_Optimized': is_optimized
        }
        best_methods_list.append(winner_record)

    return pd.DataFrame(best_methods_list)


def normalize_key(name):
    name = str(name).lower()
    name = name.replace('.csv', '').replace('clean_', '').replace('iqr_', '').strip()
    return name


def main():
    print("--- Starting results aggregation ---")
    df_results = load_and_merge_data()

    if df_results.empty:
        print("No data found.")
        return

    print("--- Computing metrics and rankings ---")
    df_scores = calculate_scores(df_results)
    df_scores.to_csv(OUTPUT_ALL_SCORES, index=False)

    print(f"--- Selecting best methods (Tolerance: {TOLERANCE * 100}%) ---")

    best_methods = select_best_method_smartly(df_scores, tolerance=TOLERANCE)

    opt_count = best_methods['Selection_Optimized'].sum()
    print(
        f"[INFO] For {opt_count} datasets, a simpler method was selected instead of the absolute highest score (within tolerance limits).")

    if os.path.exists(FILE_META_FEATURES):
        print("--- Merging with meta-features ---")
        meta_df = pd.read_csv(FILE_META_FEATURES)

        meta_df['Join_Key'] = meta_df['Dataset_Name'].apply(normalize_key)
        best_methods['Join_Key'] = best_methods['Dataset'].apply(normalize_key)

        final_df = pd.merge(meta_df, best_methods, on='Join_Key', how='inner')
        final_df = final_df.drop(columns=['Dataset', 'Join_Key', 'Selection_Optimized'])

        if final_df.empty:
            print("[ERROR] Merged table is empty.")
        else:
            final_df.to_csv(OUTPUT_BEST_METHODS, index=False)
            print(f"\nSUCCESS! Training file saved to: {OUTPUT_BEST_METHODS}")
            print(final_df[['Dataset_Name', 'Best_Method']].head())
    else:
        print("[WARN] Meta-features file not found.")
        best_methods.to_csv(OUTPUT_BEST_METHODS, index=False)


if __name__ == "__main__":
    main()