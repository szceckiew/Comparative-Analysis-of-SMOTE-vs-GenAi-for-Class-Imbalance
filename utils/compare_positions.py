import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compare_positions(
    x_before, y_before, x_after, y_after,
    method_name=None, save_plot=False, random_state=42
):
    """
    Compare the distribution of samples before and after resampling
    (e.g. SMOTE, GAN) using 2D PCA visualization with improved clarity.
    """

    title = "Comparison of samples before and after resampling"

    # --- Convert inputs safely ---
    y_before = np.array(y_before)
    y_after = np.array(y_after)

    # --- Scale data before PCA ---
    scaler = StandardScaler()
    x_before_scaled = scaler.fit_transform(x_before)
    x_after_scaled = scaler.transform(x_after)

    # --- Perform PCA on scaled data ---
    pca = PCA(n_components=2, random_state=random_state)
    x_before_2d = pca.fit_transform(x_before_scaled)
    x_after_2d = pca.transform(x_after_scaled)

    # --- Identify new (synthetic) points ---
    n_before = len(x_before)
    new_points_mask = np.arange(len(x_after)) >= n_before

    # --- Create dataframes for plotting ---
    df_before = pd.DataFrame({
        "PC1": x_before_2d[:, 0],
        "PC2": x_before_2d[:, 1],
        "class": y_before,
        "set": "Before resampling"
    })

    df_after = pd.DataFrame({
        "PC1": x_after_2d[:, 0],
        "PC2": x_after_2d[:, 1],
        "class": y_after,
        "new_point": new_points_mask,
        "set": "After resampling"
    })

    # --- Determine majority and minority classes ---
    class_counts = pd.Series(y_before).value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()

    palette = {
        majority_class: "#1f77b4",  # blue
        minority_class: "#ff7f0e",  # orange
    }

    # --- Create the plot ---
    plt.figure(figsize=(10, 7))
    plt.title(f"{method_name.upper()}: {title}" if method_name else title, fontsize=14)
    plt.xlabel("Principal Component 1 (PC1)")
    plt.ylabel("Principal Component 2 (PC2)")

    # Majority class (after resampling)
    sns.scatterplot(
        data=df_after[df_after["class"] == majority_class],
        x="PC1", y="PC2",
        color=palette[majority_class],
        alpha=0.4,
        s=35,
        label="Majority class"
    )

    # Minority class (after resampling, including synthetic)
    sns.scatterplot(
        data=df_after[df_after["class"] == minority_class],
        x="PC1", y="PC2",
        color=palette[minority_class],
        alpha=0.6,
        s=40,
        label="Minority class (after resampling)"
    )

    plt.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="gray",
        fontsize=9
    )

    plt.tight_layout()

    # --- Save or show plot ---
    if save_plot:
        safe_name = method_name.lower().replace(" ", "_") if method_name else "balancing_comparison"
        filename = f"results/{random_state}-{safe_name}-compare_positions_plot.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f" Plot saved as: {filename}")
    else:
        plt.show()
