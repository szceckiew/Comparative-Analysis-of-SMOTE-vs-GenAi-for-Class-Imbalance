import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter


def plot_kendall_correlation_comparison(
        X_train_original: np.ndarray,
        y_train_original: np.ndarray,
        X_augmented: np.ndarray,
        y_augmented: np.ndarray,
        feature_names: list[str],
        dataset_name: str,
        balance_method_name: str,
        target_column: str = "target"
):
    """
    Generates and saves side-by-side heatmaps of Kendall rank correlation,
    comparing the real minority class with the balanced minority class.
    """
    print(f"Generating Kendall correlation maps for: {balance_method_name}...")

    class_counts = Counter(y_train_original)

    if len(class_counts) < 2:
        print("ERROR: At least 2 classes are required to compare correlations.")
        return

    minority_class_label = min(class_counts, key=class_counts.get)
    print(
        f"Minority class identification: {minority_class_label} (Count: {class_counts[minority_class_label]})"
    )

    original_df = pd.DataFrame(X_train_original, columns=feature_names)
    original_df[target_column] = y_train_original

    augmented_df = pd.DataFrame(X_augmented, columns=feature_names)
    augmented_df[target_column] = y_augmented

    real_minority_df = original_df[original_df[target_column] == minority_class_label].drop(columns=[target_column])
    augmented_minority_df = augmented_df[augmented_df[target_column] == minority_class_label].drop(
        columns=[target_column]
    )

    if real_minority_df.empty:
        print("ERROR: No minority class samples found in the original dataset.")
        return
    if augmented_minority_df.empty:
        print(f"ERROR: No minority class samples found in the balanced dataset for {balance_method_name}.")
        return

    try:
        real_corr = real_minority_df.corr(method='kendall')
        aug_corr = augmented_minority_df.corr(method='kendall')
    except Exception as e:
        print(f"ERROR calculating correlation: {e}")
        return

    mask = np.triu(np.ones_like(real_corr, dtype=bool))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12), sharey=True)

    sns.heatmap(
        real_corr,
        mask=mask,
        ax=ax1,
        cmap='PuOr',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .7, "label": "Kendall's τ"},
        annot=False
    )
    ax1.set_title(f"Real data (Class: {minority_class_label})", fontsize=16)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=10)

    sns.heatmap(
        aug_corr,
        mask=mask,
        ax=ax2,
        cmap='PuOr',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=.5,
        cbar=False,
        annot=False
    )
    ax2.set_title(f"Balanced data ({balance_method_name})", fontsize=16)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
    ax2.set_yticks([])

    plt.tight_layout()

    output_dir = os.path.join(os.environ.get("RESULTS_DIR", "results"), "kendall_correlation")
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/corr_kendall_{dataset_name}_{balance_method_name}.pdf"

    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Correlation plot saved to: {filename}")
    except Exception as e:
        print(f"ERROR saving plot: {e}")

    plt.close(fig)