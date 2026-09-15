import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import Tuple


def _filter_augmented_outliers(
        X_orig_minority: np.ndarray,
        X_aug_minority: np.ndarray,
        std_multiplier: float = 3.0
) -> np.ndarray:
    """
    [Helper function] Filters augmented samples that are considered outliers.

    Uses the X-sigma rule (default 3-sigma) on intra-class nearest neighbor
    distances of the original minority class as a cutoff threshold.
    """
    if len(X_orig_minority) < 2:
        print(
            "Outlier filter: Insufficient original minority samples to compute statistics. Returning all samples."
        )
        return X_aug_minority

    nn_intra = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_orig_minority)
    intra_distances, _ = nn_intra.kneighbors(X_orig_minority)

    nearest_neighbor_dists = intra_distances[:, 1]

    mean_dist = np.mean(nearest_neighbor_dists)
    std_dist = np.std(nearest_neighbor_dists)
    threshold = mean_dist + std_multiplier * std_dist

    print(f"Outlier filter: Intra-class NN mean distance: {mean_dist:.4f}")
    print(f"Outlier filter: Intra-class NN distance std dev: {std_dist:.4f}")
    print(f"Outlier filter: Cutoff threshold ({std_multiplier}-sigma): {threshold:.4f}")

    nn_inter = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_orig_minority)
    inter_distances, _ = nn_inter.kneighbors(X_aug_minority)

    inter_distances = inter_distances.ravel()

    inlier_mask = (inter_distances <= threshold)

    X_aug_minority_filtered = X_aug_minority[inlier_mask]

    n_original = len(X_aug_minority)
    n_filtered = len(X_aug_minority_filtered)
    n_removed = n_original - n_filtered

    if n_original > 0:
        print(
            f"Outlier filter: Removed {n_removed} out of {n_original} augmented samples ({n_removed / n_original * 100:.2f}%)."
        )
    else:
        print("Outlier filter: No samples to filter.")

    return X_aug_minority_filtered


def apply_outlier_filter(
        X_original: np.ndarray,
        y_original: np.ndarray,
        X_balanced: np.ndarray,
        y_balanced: np.ndarray,
        std_multiplier: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filters a balanced dataset by removing augmented samples (outliers)
    that deviate excessively from the original minority distribution.

    Returns:
        (X_filtered_balanced, y_filtered_balanced): Filtered feature and target arrays.
    """
    print("\nRunning outlier filtering for augmented samples...")

    counts = pd.Series(y_original).value_counts()
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    X_orig_majority = X_original[y_original == majority_label]
    y_orig_majority = y_original[y_original == majority_label]

    X_orig_minority = X_original[y_original == minority_label]
    y_orig_minority = y_original[y_original == minority_label]

    X_bal_minority = X_balanced[y_balanced == minority_label]

    if len(X_bal_minority) == 0 or len(X_orig_minority) == 0:
        print("Filtering error: No minority samples found.")
        return X_balanced, y_balanced

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_orig_minority)
    distances, _ = nn.kneighbors(X_bal_minority)
    is_new_mask = distances.ravel() > 1e-9

    X_aug_minority = X_bal_minority[is_new_mask]

    X_aug_minority_filtered = _filter_augmented_outliers(
        X_orig_minority,
        X_aug_minority,
        std_multiplier=std_multiplier
    )
    print("Filtering complete.")

    X_filtered_balanced = np.vstack([
        X_orig_majority,
        X_orig_minority,
        X_aug_minority_filtered
    ])

    y_filtered_balanced = np.concatenate([
        y_orig_majority,
        y_orig_minority,
        np.full(len(X_aug_minority_filtered), minority_label)
    ])

    return X_filtered_balanced, y_filtered_balanced


def save_dataset_as_csv(
        X_data: np.ndarray,
        y_data: np.ndarray,
        target_column: str,
        file_path: str
):
    """
    Saves features X and target y to a CSV file.
    Uses generic column names ('feature_0', 'feature_1', etc.) when original names are unavailable.
    """
    try:
        feature_names = [f'feature_{i}' for i in range(X_data.shape[1])]

        df_X = pd.DataFrame(X_data, columns=feature_names)
        df_y = pd.Series(y_data, name=target_column)

        final_df = pd.concat([df_X, df_y], axis=1)

        final_df.to_csv(file_path, index=False)

        print(f"Successfully saved filtered dataset to: {file_path}")

    except Exception as e:
        print(f"ERROR: Failed to save dataset to file {file_path}.")
        print(f"Error details: {e}")