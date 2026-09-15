import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math
import os


def plot_feature_histograms(
        X_train_original: np.ndarray, y_train_original: np.ndarray,
        X_test_original: np.ndarray, y_test_original: np.ndarray,
        X_augmented: np.ndarray, y_augmented: np.ndarray,
        feature_names: list[str],
        dataset_name: str,
        balance_method_name: str,
        save_path: str = None
):
    """
    Plots and saves kernel density estimation (KDE) distributions for each feature,
    comparing normal (majority) and abnormal (minority) classes across original training,
    original testing, and synthetic generated data.
    """
    if save_path is None:
        save_path = os.path.join(os.environ.get("RESULTS_DIR", "results"), "histograms")

    print(f"Generating comparative histograms (Fig. 2 style) for {balance_method_name}...")
    os.makedirs(save_path, exist_ok=True)

    n_features = X_train_original.shape[1]
    if n_features == 0:
        print("ERROR: No features to plot (X_train_original.shape[1] == 0)")
        return

    counts = pd.Series(y_train_original).value_counts()
    if len(counts) < 2:
        print(f"ERROR: Training data contains only {len(counts)} class. Cannot compare.")
        return

    minority_class = counts.idxmin()
    majority_class = counts.idxmax()
    print(f"Normal Class (Majority): {majority_class}")
    print(f"Abnormal Class (Minority): {minority_class}")

    n_original = len(X_train_original)
    if len(X_augmented) > n_original:
        X_generated = X_augmented[n_original:]
        y_generated = y_augmented[n_original:]
        print(f"Extracted {len(X_generated)} generated samples.")
    else:
        print("WARNING: No new samples found in the balanced set. Plotting original data only.")
        X_generated = np.array([]).reshape(0, n_features)
        y_generated = np.array([])

    y_gen_raw = y_augmented[n_original:]

    if len(y_gen_raw.shape) > 1:
        y_gen_raw = y_gen_raw.flatten()

    try:
        y_gen_clean = y_gen_raw.astype(float).astype(int)
        minority_class_clean = int(minority_class)
    except (ValueError, TypeError):
        y_gen_clean = y_gen_raw.astype(str)
        minority_class_clean = str(minority_class)

    mask_min = (y_gen_clean == minority_class_clean)

    print(f"[DEBUG] Found {np.sum(mask_min)} generated minority class samples.")

    X_train_maj = X_train_original[y_train_original == majority_class]
    X_test_maj = X_test_original[y_test_original == majority_class]
    X_gen_maj = X_generated[y_generated == majority_class]

    X_train_min = X_train_original[y_train_original == minority_class]
    X_test_min = X_test_original[y_test_original == minority_class]
    X_gen_min = X_generated[mask_min]

    fig, axes = plt.subplots(n_features, 2, figsize=(14, n_features * 5))

    if n_features == 1:
        axes = np.array([axes])

    for i in range(n_features):
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'

        ax_maj = axes[i, 0]
        if len(X_train_maj) > 0:
            sns.kdeplot(X_train_maj[:, i], ax=ax_maj, label="Training data", color="blue", fill=True, alpha=0.3)
        if len(X_test_maj) > 0:
            sns.kdeplot(X_test_maj[:, i], ax=ax_maj, label="Test data", color="orange", fill=True, alpha=0.3)
        if len(X_gen_maj) > 0:
            sns.kdeplot(X_gen_maj[:, i], ax=ax_maj, label="Generated data", color="green", fill=True, alpha=0.3)

        ax_maj.set_title(f"{feature_name}\n(Majority Class: {majority_class})")
        ax_maj.set_xlabel("Value (scaled)")
        ax_maj.set_ylabel("Density")
        if i == 0:
            ax_maj.legend()

        ax_min = axes[i, 1]
        if len(X_train_min) > 0:
            sns.kdeplot(X_train_min[:, i], ax=ax_min, label="Training data", color="blue", fill=True, alpha=0.3)
        if len(X_test_min) > 0:
            sns.kdeplot(X_test_min[:, i], ax=ax_min, label="Test data", color="orange", fill=True, alpha=0.3)
        if len(X_gen_min) > 0:
            feature_var = np.var(X_gen_min[:, i])

            if feature_var < 1e-6:
                mean_val = np.mean(X_gen_min[:, i])
                ax_min.axvline(mean_val, color="green", linestyle="--", linewidth=2,
                               label="Generated data (Constant!)")
                print(f"Feature {i}: Mode Collapse detected (variance ~ 0).")
            else:
                sns.kdeplot(X_gen_min[:, i], ax=ax_min, label="Generated data", color="green", fill=True,
                            alpha=0.3)

        ax_min.set_title(f"{feature_name}\n(Minority Class: {minority_class})")
        ax_min.set_xlabel("Value (scaled)")
        ax_min.set_ylabel("Density")
        if i == 0:
            ax_min.legend()

    plt.tight_layout()

    file_name = f"{save_path}/hist_compare_{dataset_name}_{balance_method_name}.pdf"
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Histograms saved to: {file_name}")
    plt.close(fig)