import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import os

def plot_dataset_collage_histograms(
        X_train_original: np.ndarray, y_train_original: np.ndarray,
        X_test_original: np.ndarray, y_test_original: np.ndarray,
        methods_data: dict[str, list[tuple[np.ndarray, np.ndarray]]],
        feature_names: list[str],
        dataset_name: str,
        save_path: str = None
):
    """
    Plots a multi-column collage of feature histograms comparing the original training
    and test distributions against synthetic data distributions generated across multiple runs.
    """
    if save_path is None:
        save_path = os.path.join(os.environ.get("RESULTS_DIR", "results"), "histograms")

    print(f"Generating histogram collage for dataset: {dataset_name}...")
    os.makedirs(save_path, exist_ok=True)

    n_features = X_train_original.shape[1]
    if n_features == 0:
        return

    name_map = {
        'gan_from_tutorial': 'gan 1',
        'gan from tutorial': 'gan 1',
        'gan_paper': 'gan 2',
        'gan_paper_2': 'gan 2*',
        'smotified_gan': 'smotified gan'
    }

    counts = pd.Series(y_train_original).value_counts()
    if len(counts) < 2:
        return

    minority_class = counts.idxmin()
    majority_class = counts.idxmax()

    X_train_maj = X_train_original[y_train_original == majority_class]
    X_test_maj = X_test_original[y_test_original == majority_class]
    X_train_min = X_train_original[y_train_original == minority_class]
    X_test_min = X_test_original[y_test_original == minority_class]

    method_names = list(methods_data.keys())
    n_methods = len(method_names)
    n_cols = 1 + n_methods

    fig, axes = plt.subplots(n_features, n_cols, figsize=(5 * n_cols, n_features * 4.0))
    axes = np.array(axes).reshape(n_features, n_cols)

    for i in range(n_features):
        feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'

        ax_maj = axes[i, 0]
        if len(X_train_maj) > 0:
            if np.var(X_train_maj[:, i]) < 1e-6:
                mean_val = np.mean(X_train_maj[:, i])
                ax_maj.axvline(mean_val, color="blue", linestyle=":", linewidth=2, label="Train (Orig) Constant", alpha=0.5)
            else:
                sns.kdeplot(X_train_maj[:, i], ax=ax_maj, label="Train (Orig)", color="blue", fill=True, alpha=0.2)
                
        if len(X_test_maj) > 0:
            if np.var(X_test_maj[:, i]) < 1e-6:
                mean_val = np.mean(X_test_maj[:, i])
                ax_maj.axvline(mean_val, color="orange", linestyle=":", linewidth=2, label="Test (Orig) Constant", alpha=0.5)
            else:
                sns.kdeplot(X_test_maj[:, i], ax=ax_maj, label="Test (Orig)", color="orange", fill=True, alpha=0.2)
        
        ax_maj.set_ylabel(feature_name, fontsize=12, weight='bold', labelpad=10)
        
        if i == n_features - 1:
            ax_maj.set_xlabel(f"Majority Class\n({majority_class})", fontsize=12, weight='bold', labelpad=10)
        else:
            ax_maj.set_xlabel("")

        if i == 0:
            ax_maj.legend(loc="upper right", fontsize=8)

        for m_idx, method_name in enumerate(method_names):
            col_idx = m_idx + 1
            ax_min = axes[i, col_idx]
            display_method_name = name_map.get(method_name, method_name)

            if len(X_train_min) > 0:
                if np.var(X_train_min[:, i]) < 1e-6:
                    mean_val = np.mean(X_train_min[:, i])
                    ax_min.axvline(mean_val, color="blue", linestyle=":", linewidth=2, label="Train (Orig) Constant" if i==0 else None, alpha=0.5)
                else:
                    sns.kdeplot(X_train_min[:, i], ax=ax_min, label="Train (Orig)" if i==0 else None, color="blue", fill=True, alpha=0.15)
            
            if len(X_test_min) > 0:
                if np.var(X_test_min[:, i]) < 1e-6:
                    mean_val = np.mean(X_test_min[:, i])
                    ax_min.axvline(mean_val, color="orange", linestyle=":", linewidth=2, label="Test (Orig) Constant" if i==0 else None, alpha=0.5)
                else:
                    sns.kdeplot(X_test_min[:, i], ax=ax_min, label="Test (Orig)" if i==0 else None, color="orange", fill=True, alpha=0.15)

            runs_list = methods_data[method_name]
            n_original = len(X_train_original)

            for run_idx, (X_augmented, y_augmented) in enumerate(runs_list):
                if len(X_augmented) > n_original:
                    X_generated = X_augmented[n_original:]
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
                    X_gen_min = X_generated[mask_min]

                    if len(X_gen_min) > 0:
                        feature_var = np.var(X_gen_min[:, i])
                        
                        gen_label = "Generated" if run_idx == 0 else None
                        mc_label = "Gen (Mode Collapse)" if run_idx == 0 else None

                        if feature_var < 1e-6:
                            mean_val = np.mean(X_gen_min[:, i])
                            ax_min.axvline(mean_val, color="green", linestyle="--", linewidth=1.5, alpha=0.5, label=mc_label)
                        else:
                            sns.kdeplot(X_gen_min[:, i], ax=ax_min, label=gen_label, color="green", fill=False, alpha=0.3, linewidth=1.5)

            ax_min.set_ylabel("")
            
            if i == n_features - 1:
                ax_min.set_xlabel(display_method_name, fontsize=12, weight='bold', labelpad=10)
            else:
                ax_min.set_xlabel("")

            if i == 0:
                ax_min.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    file_name = f"{save_path}/collage_hist_{dataset_name}.pdf"
    plt.savefig(file_name, bbox_inches='tight')
    print(f"Saved histogram collage (multi-run) to: {file_name}")
    plt.close(fig)