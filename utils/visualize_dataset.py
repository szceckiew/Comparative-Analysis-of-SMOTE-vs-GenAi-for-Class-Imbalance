import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from typing import List, Dict, Optional, Tuple
from sklearn.neighbors import NearestNeighbors


def plot_tsne_augmentation_comparison(
        file_path,
        X_original,
        y_original,
        X_balanced,
        y_balanced,
        dataset_name,
        balance_method_name="Augmented",
        random_state=42
):
    """
    Creates a t-SNE plot comparing the majority class,
    original minority samples, and augmented minority samples.
    """
    counts = pd.Series(y_original).value_counts()
    minority_label = counts.idxmin()
    majority_label = counts.idxmax()

    X_orig_majority = X_original[y_original == majority_label]
    X_orig_minority = X_original[y_original == minority_label]
    X_bal_minority = X_balanced[y_balanced == minority_label]

    if len(X_bal_minority) == 0 or len(X_orig_minority) == 0:
        print("Error: No minority samples found in original or balanced dataset.")
        return

    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X_orig_minority)
    distances, _ = nn.kneighbors(X_bal_minority)
    is_new_mask = distances.ravel() > 1e-9
    X_aug_minority = X_bal_minority[is_new_mask]

    X_to_plot = np.vstack([X_orig_majority, X_orig_minority, X_aug_minority])

    n_maj = len(X_orig_majority)
    n_orig_min = len(X_orig_minority)
    n_aug_min = len(X_aug_minority)

    print(f"Found {n_maj} majority class samples.")
    print(f"Found {n_orig_min} original minority samples.")
    print(f"Found {n_aug_min} augmented minority samples.")

    y_to_plot = np.concatenate([
        np.full(n_maj, 0),
        np.full(n_orig_min, 1),
        np.full(n_aug_min, 2)
    ])

    unique_classes = np.unique(y_to_plot)

    n_samples = len(X_to_plot)
    perplexity_value = min(30.0, n_samples - 1.0)

    if perplexity_value <= 0:
        print(f"Error: Too few samples to run t-SNE (n={n_samples}).")
        return

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        perplexity=perplexity_value,
        init='pca',
        learning_rate='auto',
        n_iter_without_progress=1000
    )

    print("Running t-SNE... (this may take a while)")
    X_tsne = tsne.fit_transform(X_to_plot)
    print("t-SNE completed.")

    plt.figure(figsize=(10, 8))

    colors = ['blue', 'darkolivegreen', 'red']
    markers = ['o', '^', 's']

    target_names = [
        f'Majority class (n={n_maj})',
        f'Minority class (Original) (n={n_orig_min})',
        f'Minority class (Augmented) (n={n_aug_min})'
    ]

    for i in unique_classes:
        class_index = int(i)
        indices_to_plot = (y_to_plot == i)

        color_idx = class_index % len(colors)
        marker_idx = class_index % len(markers)

        label_name = target_names[class_index]
        layer_priority = {0: 1, 1: 3, 2: 2}.get(class_index, 1)

        plt.scatter(
            X_tsne[indices_to_plot, 0],
            X_tsne[indices_to_plot, 1],
            c=colors[color_idx],
            marker=markers[marker_idx],
            label=label_name,
            s=100,
            alpha=0.9,
            zorder=layer_priority
        )

    plt.xlabel('x1', fontsize=18)
    plt.ylabel('x2', fontsize=18)

    plt.legend(loc='lower left', fontsize=16)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(file_path + f"tsne_{dataset_name}_{balance_method_name}_rs{random_state}.pdf")


def perform_and_plot_tsne(
        file_path: str,
        target_column: str = 'diagnosis',
        target_mapping: Optional[Dict[str, int]] = {'M': 1, 'B': 0},
        columns_to_drop: Optional[List[str]] = ['id'],
        n_components: int = 2,
        perplexity: float = 30,
        n_iter_without_progress: int = 300,
        random_state: int = 42,
        save_file: bool = True,
        save_file_name: str = 'tsne_custom_dataset.png'
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Loads data, scales features, performs dimensionality reduction via t-SNE,
    and generates/saves a scatter plot of the results.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERROR: File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return None

    if target_column not in df.columns:
        print(f"ERROR: Target column '{target_column}' not found.")
        return None

    y = df[target_column]
    target_names = ['Benign', 'Malignant']

    if target_mapping and df[target_column].dtype == 'object':
        try:
            y = y.map(target_mapping).values
        except KeyError as e:
            print(f"ERROR: Label value {e} not found in mapping: {target_mapping}")
            return None
    else:
        y = y.values

    cols_to_drop = [target_column] + (columns_to_drop if columns_to_drop else [])
    X = df.drop(columns=cols_to_drop, errors='ignore')
    X = X.select_dtypes(include=np.number)

    if X.empty:
        print("ERROR: No numerical columns available for analysis after dropping specified columns.")
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Starting t-SNE (n_components={n_components}, perplexity={perplexity}, random_state={random_state})...")
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter_without_progress=n_iter_without_progress,
        random_state=random_state,
        n_jobs=-1
    )
    X_tsne = tsne.fit_transform(X_scaled)
    print("t-SNE completed.")

    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        print(
            "WARNING: More than 2 classes detected. Plot will only cycle the first two colors/markers."
        )

    plt.figure(figsize=(10, 8))

    colors = ['blue', 'darkolivegreen']
    markers = ['o', '^']

    for i in unique_classes:
        class_index = int(i)
        indices_to_plot = y == i

        count = np.sum(indices_to_plot)

        color_idx = class_index % len(colors)
        marker_idx = class_index % len(markers)

        label_name_base = target_names[class_index] if class_index < len(target_names) else f'Class {i}'
        label_name = f'{label_name_base} (n={count})'

        plt.scatter(
            X_tsne[indices_to_plot, 0],
            X_tsne[indices_to_plot, 1],
            c=colors[color_idx],
            marker=markers[marker_idx],
            label=label_name,
            s=100,
            alpha=0.9,
            zorder=class_index + 2
        )

    plt.xlabel('x1', fontsize=18)
    plt.ylabel('x2', fontsize=18)

    plt.legend(loc='lower left', fontsize=16)
    plt.grid(False)
    plt.tight_layout()

    if save_file:
        plt.savefig(save_file_name)
        print(f"\nPlot saved to '{save_file_name}'.")

    return X_tsne, y


if __name__ == '__main__':
    file_path_example = '../data/clean/Clean_Breast_Cancer_Wisconsin.csv'
    target_col = 'diagnosis'

    tsne_results, original_labels = perform_and_plot_tsne(
        file_path=file_path_example,
        target_column=target_col,
        columns_to_drop=['id'],
        perplexity=30,
        random_state=42,
        save_file_name='Breast_Cancer.png'
    )