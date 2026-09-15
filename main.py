import argparse
import pandas as pd
import numpy as np
import os
import time

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

from utils.visualization import plot_class_distribution
from utils.evaluation import evaluate_model
from utils.visualize_dataset import plot_tsne_augmentation_comparison
from utils.post_balance_filter import apply_outlier_filter
from utils.post_balance_filter import save_dataset_as_csv

from balance.smote import apply_smote
from balance.gan_paper import apply_gan_paper_oversample
from balance.gan_paper_2 import GBO

from balance.gan_from_tutorial import train_and_augment_gan_keras
from balance.wgangp import apply_wgangp_oversample
from balance.ct_gan import apply_ctgan_oversample
from balance.smotified_gan import apply_smotified_gan_early_stopping

from classify.random_forest import train_random_forest
from classify.tune_hgb import tune_hgb
from classify.my_lightGBM import tune_lgb
from classify.my_xgboost import tune_xgb


from dataset_information_extraction.iqr_preprocessing import IQRCapper
from utils.dataset_metrics import evaluate_synthetic_quality
from utils.plot_feature_histograms import plot_feature_histograms
from utils.plot_kendall_correlation import plot_kendall_correlation_comparison

def apply_gan_paper_2_oversample(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gbo = GBO()
    X_res, y_res = gbo.fit_resample(X_train, y_train)
    return X_res, y_res

# --- Global Configuration ---
TARGET_COLUMN = "Class"
DATA_DIR = "data/clean"

# List of available datasets in data/clean
AVAILABLE_DATASETS = [
    "Clean_blood.csv",
    "Clean_Breast_Cancer_Wisconsin.csv",
    "Clean_covid.csv",
    "Clean_creditcard.csv",
    "Clean_DDos.csv",
    "Clean_haberman.csv",
    "Clean_heart.csv",
    "Clean_Infiltration.csv",
    "Clean_ionosphere.csv",
    "Clean_Loan_Default.csv",
    "Clean_nsl_kdd_full_dataset.csv",
    "Clean_pimadiabetes.csv",
    "Clean_shuttle_full_dataset.csv",
    "Clean_Titanic-Dataset.csv",
    "Clean_weatherAUS.csv",
    "Clean_spambase.csv",
    "Clean_parkinsons.csv",
    "Clean_sonar.csv",
    "Clean_mdlon.csv",
    "heart_balanced.csv",
    "balanced_artificial_dataset.csv",
    "Clean_banknote.csv"
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

# Methods requiring input features normalized to [0, 1] due to sigmoid generator output
METHODS_REQUIRING_MINMAX = ["gan_from_tutorial", "gan_paper", "gan_paper_2", "smotified_gan"]

TEST_SIZE = 0.3
VAL_SIZE = 0.2


# --- Dataclass ---
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


# --- Helper Functions ---

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Pipeline for imbalanced data classification")

    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="Clean_Breast_Cancer_Wisconsin.csv")

    parser.add_argument("--balance", choices=BALANCE_FUNCTIONS.keys(), default="gan_from_tutorial")
    
    # Build available classifiers list
    available_classifiers = ["random_forest", "tune_hgb", "lightgbm", "xgboost"]

    parser.add_argument("--classifier", choices=available_classifiers, default="tune_hgb")
    parser.add_argument("--use_gpu", action='store_true', help="Use GPU for LightGBM/XGBoost training")
    parser.add_argument("--saveResults", action='store_true', help="Save results to files")
    parser.add_argument("--randomState", type=int, default=42)
    parser.add_argument("--showPlots", action='store_true', help="Show plots during execution")
    parser.add_argument("--use_tsne", action='store_true', help="Use t-SNE visualization")
    parser.add_argument("--save_filtered_dataset", action='store_true', help="Save filtered dataset to CSV")
    parser.add_argument("--use_iqr_preprocessing", action='store_true',
                        help="Apply IQR capping (fit on train, transform all)")
    parser.add_argument("--skip_baseline", action='store_true', help="Skip baseline experiment")
    parser.add_argument("--skip_classification", action='store_true', help="Skip classification")
    parser.add_argument("--use_additional_eval", action='store_true', help="Run additional evaluation at the end")
    parser.add_argument("--plot_histograms", action='store_true', help="Plot feature histograms")
    parser.add_argument("--plot_kendall_correlation", action='store_true', help="Plot Kendall correlation")
    
    # Arguments for artificial imbalance experiments and output directories
    parser.add_argument("--artificial_imbalance_ratio", type=float, default=None,
                        help="Ratio to artificially imbalance the dataset (e.g. 0.1 for 10% minority retention). If set, forcefully drops minority records.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory to save the results")
    parser.add_argument("--skip_unfiltered", action="store_true", help="Skip evaluation and saving for unfiltered data (runs filtered variant only)")

    return parser.parse_args()


def apply_artificial_imbalance(X, y, minority_keep_ratio, random_state):
    """Artificially down-samples the minority class across the entire dataset."""
    from collections import Counter
    counts = Counter(y)
    
    # Sort classes by count (ascending) so [0] is minority, [-1] is majority
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
    
    # Guard against excessively small minority sample sizes for nearest-neighbor algorithms (e.g., SMOTE)
    # SMOTE requires at least 6 samples in the training set; with train/val/test splits, 15 ensures stability
    min_required_samples = 15
    if n_keep < min_required_samples:
        n_keep = min(min_required_samples, len(X_min))
        print(f" Minority guard: Ratio too small. Forcing n_keep={n_keep} to prevent errors in SMOTE.")
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(X_min))
    keep_indices = shuffled_indices[:n_keep]
    
    X_min_keep = X_min.iloc[keep_indices]
    y_min_keep = y_min.iloc[keep_indices]
    
    X_imbalanced = pd.concat([X_maj, X_min_keep]).reset_index(drop=True)
    y_imbalanced = pd.concat([y_maj, y_min_keep]).reset_index(drop=True)
    
    print(f"--- ARTIFICIAL IMBALANCE (Ratio={minority_keep_ratio}) ---")
    print(f"Original distribution: {counts}")
    print(f"After reduction, minority class ({minority_class}) has: {len(y_min_keep)} samples")
    
    return X_imbalanced, y_imbalanced


def load_data(path: str, target_col: str, args: argparse.Namespace) -> DatasetSplits:
    """
    Loads data, splits partitions, and applies the appropriate scaler based
    on the selected oversampling method (args.balance).
    """
    print(f"Loading data from: {path}")

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {path}")
        exit(1)

    # 1. Separate features and target
    try:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    except KeyError:
        print(f"ERROR: Target column '{target_col}' not found in file.")
        print(f"Available columns: {df.columns.tolist()}")
        exit(1)

    if args.artificial_imbalance_ratio is not None:
        X, y = apply_artificial_imbalance(X, y, args.artificial_imbalance_ratio, args.randomState)

    features = X.columns.tolist()
    target_names = list(np.unique(y).astype(str))

    # 2. Split partitions (Train / Val / Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=args.randomState,
        stratify=y
    )

    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=args.randomState,
        stratify=y_train_val
    )

    print("Data splitting completed.")
    print(f"X_train shape: {X_train.shape}")

    # 3. Optional IQR Preprocessing
    if args.use_iqr_preprocessing:
        print("\nApplying IQR preprocessing (Fit on Train, Transform all)...")
        capper = IQRCapper(k=1.5)
        capper.fit(X_train)
        X_train = capper.transform(X_train)
        X_val = capper.transform(X_val)
        X_test = capper.transform(X_test)

    # 4. Feature Scaling
    if args.balance in METHODS_REQUIRING_MINMAX:
        print(f"\n Selected method '{args.balance}' requires feature range [0, 1].")
        print("   Applying MinMaxScaler (Fit on Train, Transform all)...")
        scaler = MinMaxScaler()
    else:
        print(f"\n Selected method '{args.balance}' assumes standard-scaled features.")
        print("   Applying StandardScaler (Fit on Train, Transform all)...")
        scaler = StandardScaler()

    scaler.fit(X_train)
    x_train_scaled = scaler.transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    x_test_scaled = scaler.transform(X_test)
    print("Scaling completed.")

    return DatasetSplits(
        x_train=x_train_scaled,
        y_train=y_train.values,
        x_val=x_val_scaled,
        y_val=y_val.values,
        x_test=x_test_scaled,
        y_test=y_test.values,
        target_names=target_names,
        feature_names=features
    )


def balance_data(data: DatasetSplits, method: str) -> tuple[np.ndarray, np.ndarray]:
    """Applies the selected data balancing method."""
    balance_func = BALANCE_FUNCTIONS.get(method)
    if balance_func is None:
        print(f"ERROR: non-existent balance method: {method}")
        exit(1)

    print(f"\nApplying balancing method: {method}...")
    
    if method == "smotified_gan":
        return balance_func(data.x_train, data.y_train, data.x_val, data.y_val)
    else:
        return balance_func(data.x_train, data.y_train)


def run_experiment(
        args: argparse.Namespace,
        data: DatasetSplits,
        x_train_aug: np.ndarray,
        y_train_aug: np.ndarray,
        balance_method_name: str,
        dataset_name: str,
        generation_time: float
):
    print(f"\n--- Running experiment: {balance_method_name} ---")

    if balance_method_name != "baseline":
        evaluate_synthetic_quality(
            X_real=data.x_train,
            y_real=data.y_train,
            X_augmented=x_train_aug,
            y_augmented=y_train_aug,
            dataset_name=dataset_name,
            balance_method_name=balance_method_name
        )

        if args.plot_kendall_correlation:
            plot_kendall_correlation_comparison(
                X_train_original=data.x_train,
                y_train_original=data.y_train,
                X_augmented=x_train_aug,
                y_augmented=y_train_aug,
                feature_names=data.feature_names,
                dataset_name=dataset_name,
                balance_method_name=balance_method_name
            )

        if args.use_tsne:
            print("Generating t-SNE visualization...")
            plot_tsne_augmentation_comparison(
                os.environ.get("RESULTS_DIR", "results"),
                data.x_train,
                data.y_train,
                x_train_aug,
                y_train_aug,
                dataset_name,
                balance_method_name,
                args.randomState
            )

    if args.skip_classification:
        print("Skipping classifier training due to --skip_classification flag.")
        return

    print(f"Training classifier: {args.classifier}...")
    if args.classifier == "random_forest":
        model = train_random_forest(
            x_train_aug, y_train_aug,
            data.x_test, data.y_test,
            args.randomState,
            balance_method_name,
            dataset_name,
            generation_time=generation_time
        )
    elif args.classifier == "tune_hgb":
        model = tune_hgb(
            x_train_aug, y_train_aug,
            data.x_val, data.y_val,
            data.x_test, data.y_test,
            args.randomState,
            balance_method_name,
            dataset_name,
            generation_time=generation_time
        )
    elif args.classifier == "lightgbm":
        model = tune_lgb(
            x_train_aug, y_train_aug,
            data.x_val, data.y_val,
            data.x_test, data.y_test,
            args.randomState,
            balance_method_name,
            dataset_name,
            generation_time=generation_time,
            use_gpu=args.use_gpu
        )
    elif args.classifier == "xgboost":
        model = tune_xgb(
            x_train_aug, y_train_aug,
            data.x_val, data.y_val,
            data.x_test, data.y_test,
            args.randomState,
            balance_method_name,
            dataset_name,
            generation_time=generation_time,
            use_gpu=args.use_gpu
        )
    else:
        print(f"ERROR: Unknown classifier: {args.classifier}")
        return

    if args.use_additional_eval:
        print("Starting model evaluation...")
        evaluate_model(
            model, args.classifier, balance_method_name,
            data.x_test, data.y_test, data.target_names,
            args.saveResults, args.randomState, args.showPlots,
            dataset_name
        )

def main():
    args = parse_arguments()
    RANDOM_STATE = args.randomState
    
    # Export results directory path to environment variable
    os.environ["RESULTS_DIR"] = args.results_dir

    # 1. Configuration and data loading
    # Build complete path to file using DATA_DIR and dataset filename
    dataset_path = os.path.join(DATA_DIR, args.dataset)
    dataset_name = args.dataset.split('.')[0]

    if args.use_iqr_preprocessing:
        print("IQR preprocessing activated.")
        dataset_name = f"iqr_{dataset_name}"

    # load_data uses global TARGET_COLUMN constant and dynamic file path
    data = load_data(dataset_path, TARGET_COLUMN, args)

    # 2. Pre-balancing visualization
    print(" Class counts before balancing (training set):")
    print(pd.Series(data.y_train).value_counts())
    if args.showPlots:
        plot_class_distribution(data.y_train, "Before")
    if not args.skip_baseline:
        print("\n--- EXPERIMENT 0: Baseline (Original data without oversampling) ---")
        run_experiment(
            args=args,
            data=data,
            x_train_aug=data.x_train,
            y_train_aug=data.y_train,
            balance_method_name="baseline",
            dataset_name=dataset_name,
            generation_time=0.0
        )

    start_gen = time.time()
    x_train_balanced, y_train_balanced = balance_data(data, args.balance)
    generation_time = time.time() - start_gen

    print(f"\n Class counts after applying {args.balance}:")
    print(pd.Series(y_train_balanced).value_counts())
    print(f"⏱ Generation/balancing time: {generation_time:.4f} seconds")
    
    if args.plot_histograms:
        print("\nGenerating feature distribution plots...")
        plot_feature_histograms(
            X_train_original=data.x_train,
            y_train_original=data.y_train,
            X_test_original=data.x_test,
            y_test_original=data.y_test,
            X_augmented=x_train_balanced,
            y_augmented=y_train_balanced,
            feature_names=data.feature_names,
            dataset_name=dataset_name,
            balance_method_name=args.balance
        )

    n_orig = len(data.x_train)
    new_samples = x_train_balanced[n_orig:]

    # Safeguard against runs where no synthetic samples are produced
    if len(new_samples) > 0:
        std_devs = np.std(new_samples, axis=0)
        print("\n--- MODE COLLAPSE DIAGNOSTICS ---")
        print(f"Mean standard deviation of synthesized samples: {np.mean(std_devs):.6f}")

        if np.mean(std_devs) < 0.01:
            print("DIAGNOSIS: Complete mode collapse detected.")
        else:
            print("DIAGNOSIS: Acceptable sample diversity observed.")
    else:
        print("\n--- WARNING: No synthetic samples generated ---")

    x_train_balanced, y_train_balanced = shuffle(
        x_train_balanced,
        y_train_balanced,
        random_state=RANDOM_STATE
    )

    if not args.skip_unfiltered:
        # 4. EXPERIMENT 1
        run_experiment(
            args=args,
            data=data,
            x_train_aug=x_train_balanced,
            y_train_aug=y_train_balanced,
            balance_method_name=args.balance,
            dataset_name=dataset_name,
            generation_time=generation_time
        )

    # 5. Outlier Filtering
    print("\nApplying post-synthesis outlier filtration...")
    x_train_filtered, y_train_filtered = apply_outlier_filter(
        data.x_train,
        data.y_train,
        x_train_balanced,
        y_train_balanced,
        std_multiplier=3.0
    )

    if args.save_filtered_dataset:
        filtered_file_name = f"data/filtered/filtered_{args.balance}_rs{RANDOM_STATE}.csv"
        # Use TARGET_COLUMN for output serialization
        save_dataset_as_csv(
            x_train_filtered,
            y_train_filtered,
            TARGET_COLUMN,
            filtered_file_name
        )
        print(f"Filtered dataset saved to: {filtered_file_name}")

    # 6. EXPERIMENT 2
    run_experiment(
        args=args,
        data=data,
        x_train_aug=x_train_filtered,
        y_train_aug=y_train_filtered,
        balance_method_name=f"{args.balance}_filtered",
        dataset_name=dataset_name,
        generation_time=generation_time
    )

    print("\nPipeline executed successfully.")


if __name__ == "__main__":
    main()