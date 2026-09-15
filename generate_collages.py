import argparse
import pandas as pd
import numpy as np
import os
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Imports of balancing functions
from balance.smote import apply_smote
from balance.gan_paper import apply_gan_paper_oversample
from balance.gan_paper_2 import GBO
from balance.gan_from_tutorial import train_and_augment_gan_keras
from balance.wgangp import apply_wgangp_oversample
from balance.ct_gan import apply_ctgan_oversample
from balance.smotified_gan import apply_smotified_gan_early_stopping

from dataset_information_extraction.iqr_preprocessing import IQRCapper

# Imports of collage plotting utilities
from utils.histograms_collage import plot_dataset_collage_histograms
from utils.kendalls_collage import plot_dataset_collage_kendall


def apply_gan_paper_2_oversample(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gbo = GBO()
    X_res, y_res = gbo.fit_resample(X_train, y_train)
    return X_res, y_res


# --- Global Configuration ---
TARGET_COLUMN = "Class"
DATA_DIR = "data/clean"

AVAILABLE_DATASETS = [
    "Clean_blood.csv", "Clean_Breast_Cancer_Wisconsin.csv", "Clean_covid.csv",
    "Clean_creditcard.csv", "Clean_DDos.csv", "Clean_haberman.csv", "Clean_heart.csv",
    "Clean_Infiltration.csv", "Clean_ionosphere.csv", "Clean_Loan_Default.csv",
    "Clean_nsl_kdd_full_dataset.csv", "Clean_pimadiabetes.csv", "Clean_shuttle_full_dataset.csv",
    "Clean_Titanic-Dataset.csv", "Clean_weatherAUS.csv", "Clean_spambase.csv",
    "Clean_parkinsons.csv", "Clean_sonar.csv", "Clean_mdlon.csv", "heart_balanced.csv",
    "balanced_artificial_dataset.csv", "Clean_banknote.csv"
]

BALANCE_FUNCTIONS = {
    "smote": apply_smote,
    "gan_from_tutorial": train_and_augment_gan_keras,
    "gan_paper": apply_gan_paper_oversample,
    "gan_paper_2": apply_gan_paper_2_oversample,
    "wgangp": apply_wgangp_oversample,
    "ctgan": apply_ctgan_oversample,
    "smotified_gan": apply_smotified_gan_early_stopping
}

METHODS_REQUIRING_MINMAX = ["gan_from_tutorial", "gan_paper", "gan_paper_2", "smotified_gan"]

TEST_SIZE = 0.3
VAL_SIZE = 0.2

@dataclass
class DatasetSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    target_names: list[str]
    feature_names: list[str]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script generating exclusively histogram collages for datasets.")
    
    parser.add_argument("--datasets", nargs='+', default=["Clean_Breast_Cancer_Wisconsin.csv"], 
                        help="List of datasets (space-separated) or 'all'")
    parser.add_argument("--balances", nargs='+', default=["smote", "gan_from_tutorial"], 
                        help="List of balancing methods for collage (space-separated) or 'all'")
    
    parser.add_argument("--randomState", type=int, default=42)
    parser.add_argument("--use_iqr_preprocessing", action='store_true', help="Apply IQR capping")
    parser.add_argument("--artificial_imbalance_ratio", type=float, default=None)
    parser.add_argument("--results_dir", type=str, default="results/histograms_collages",
                        help="Target directory for PDF collage files")

    return parser.parse_args()


def apply_artificial_imbalance(X, y, minority_keep_ratio, random_state):
    from collections import Counter
    counts = Counter(y)
    sorted_classes = sorted(counts.keys(), key=lambda c: counts[c])
    minority_class = sorted_classes[0]
    majority_class = sorted_classes[-1]
    
    mask_majority = (y == majority_class)
    mask_minority = (y == minority_class)
    
    X_maj = X[mask_majority]
    y_maj = y[mask_majority]
    X_min = X[mask_minority]
    y_min = y[mask_minority]
    
    n_keep = int(len(X_min) * minority_keep_ratio)
    min_required_samples = 15
    if n_keep < min_required_samples:
        n_keep = min(min_required_samples, len(X_min))
        
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X_min))
    keep_indices = shuffled_indices[:n_keep]
    
    X_min_keep = X_min.iloc[keep_indices]
    y_min_keep = y_min.iloc[keep_indices]
    
    X_imbalanced = pd.concat([X_maj, X_min_keep]).reset_index(drop=True)
    y_imbalanced = pd.concat([y_maj, y_min_keep]).reset_index(drop=True)
    
    return X_imbalanced, y_imbalanced


def load_data(path: str, target_col: str, args: argparse.Namespace, requires_minmax: bool) -> DatasetSplits:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: File not found: {path}")
        return None

    y = df[target_col]
    X = df.drop(columns=[target_col])

    if args.artificial_imbalance_ratio is not None:
        X, y = apply_artificial_imbalance(X, y, args.artificial_imbalance_ratio, args.randomState)

    features = X.columns.tolist()
    target_names = list(np.unique(y).astype(str))

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.randomState, stratify=y
    )

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=args.randomState, stratify=y_train_val
    )

    if args.use_iqr_preprocessing:
        capper = IQRCapper(k=1.5)
        capper.fit(X_train)
        X_train = capper.transform(X_train)
        X_val = capper.transform(X_val)
        X_test = capper.transform(X_test)

    # Unified scaler for all methods on the given dataset
    if requires_minmax:
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaler.fit(X_train)
    return DatasetSplits(
        x_train=scaler.transform(X_train), y_train=y_train.values,
        x_val=scaler.transform(X_val), y_val=y_val.values,
        x_test=scaler.transform(X_test), y_test=y_test.values,
        target_names=target_names, feature_names=features
    )


def balance_data(data: DatasetSplits, method: str) -> tuple[np.ndarray, np.ndarray]:
    balance_func = BALANCE_FUNCTIONS.get(method)
    if balance_func is None:
        raise ValueError(f"ERROR: non-existent balance method: {method}")
        
    if method == "smotified_gan":
        return balance_func(data.x_train, data.y_train, data.x_val, data.y_val)
    else:
        return balance_func(data.x_train, data.y_train)


def main():
    args = parse_arguments()
    os.makedirs(args.results_dir, exist_ok=True)
    os.environ["RESULTS_DIR"] = args.results_dir

    datasets_to_run = AVAILABLE_DATASETS if "all" in args.datasets else args.datasets
    balances_to_run = list(BALANCE_FUNCTIONS.keys()) if "all" in args.balances else args.balances

    print("=== START OF HISTOGRAM COLLAGE GENERATOR ===")
    print(f"Datasets ({len(datasets_to_run)}): {datasets_to_run}")
    print(f"Methods ({len(balances_to_run)}): {balances_to_run}")

    for dataset_file in datasets_to_run:
        print(f"\n{'-'*60}\nProcessing dataset: {dataset_file}\n{'-'*60}")
        
        dataset_path = os.path.join(DATA_DIR, dataset_file)
        dataset_name = dataset_file.split('.')[0]
        if args.use_iqr_preprocessing:
            dataset_name = f"iqr_{dataset_name}"

        # Determine scaler for the entire dataset
        requires_minmax = any(m in METHODS_REQUIRING_MINMAX for m in balances_to_run)
        data = load_data(dataset_path, TARGET_COLUMN, args, requires_minmax)
        
        if data is None:
            continue

        # Number of generation repetitions to average over
        NUM_RUNS = 10
        
        # Collect collage data for the given dataset
        collage_data_for_dataset = {}
        n_orig = len(data.x_train)

        for balance_method in balances_to_run:
            print(f"  -> Generating data using method: {balance_method} ({NUM_RUNS} runs)...")
            runs_list = []
            
            try:
                for run_idx in range(NUM_RUNS):
                    # Change seed so that each run is distinct
                    np.random.seed(args.randomState + run_idx)
                    
                    x_train_balanced, y_train_balanced = balance_data(data, balance_method)
                    runs_list.append((x_train_balanced, y_train_balanced))

                # Store all runs for the given method
                collage_data_for_dataset[balance_method] = runs_list
                    
            except Exception as e:
                print(f"  [!] Error during balancing {balance_method}: {e}")


        # If data was successfully collected for at least one method, render collages
        if collage_data_for_dataset:
            print(f"\nPlotting collage for {dataset_name}...")
            plot_dataset_collage_histograms(
                X_train_original=data.x_train,
                y_train_original=data.y_train,
                X_test_original=data.x_test,
                y_test_original=data.y_test,
                methods_data=collage_data_for_dataset,
                feature_names=data.feature_names,
                dataset_name=dataset_name,
                save_path=args.results_dir
            )

            print(f"Plotting Kendall correlation collage for {dataset_name}...")
            plot_dataset_collage_kendall(
                X_train_original=data.x_train,
                y_train_original=data.y_train,
                methods_data=collage_data_for_dataset,
                feature_names=data.feature_names,
                dataset_name=dataset_name,
                save_path=args.results_dir
            )

        else:
            print(f"No successfully generated data for dataset {dataset_name}. Skipping plotting.")

    print("\nAll collages have been successfully generated!")


if __name__ == "__main__":
    main()