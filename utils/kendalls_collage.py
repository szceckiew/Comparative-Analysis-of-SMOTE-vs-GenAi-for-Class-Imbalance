import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math
from collections import Counter

def plot_dataset_collage_kendall(
        X_train_original: np.ndarray, y_train_original: np.ndarray,
        methods_data: dict[str, list[tuple[np.ndarray, np.ndarray]]],
        feature_names: list[str],
        dataset_name: str,
        save_path: str = None
):
    """
    Plots individual Kendall rank correlation matrix collages (mean correlation and instability standard deviation)
    for minority class samples across different oversampling methods.
    """
    if save_path is None:
        save_path = os.path.join(os.environ.get("RESULTS_DIR", "results"), "kendall_collages")

    print(f"Generating Kendall correlation collages (Mean and Instability) for: {dataset_name}...")
    os.makedirs(save_path, exist_ok=True)

    name_map = {
        'gan_from_tutorial': 'gan 1',
        'gan from tutorial': 'gan 1',
        'gan_paper': 'gan 2',
        'gan_paper_2': 'gan 2*'
    }

    class_counts = Counter(y_train_original)
    if len(class_counts) < 2:
        return

    minority_class = min(class_counts, key=class_counts.get)
    
    original_df = pd.DataFrame(X_train_original, columns=feature_names)
    original_df['target'] = y_train_original
    real_minority_df = original_df[original_df['target'] == minority_class].drop(columns=['target'])
    
    if real_minority_df.empty:
        return

    try:
        real_corr = real_minority_df.corr(method='kendall')
    except Exception as e:
        print(f"ERROR calculating original data correlation: {e}")
        return

    mean_plot_items = [("Real Data (Minority)", real_corr, False)]
    std_plot_items = [("Real Data (Reference)", real_corr, False)]

    n_original = len(X_train_original)
    
    for method_name, data_payload in methods_data.items():
        display_name = name_map.get(method_name, method_name)
        runs_list = data_payload if isinstance(data_payload, list) else [data_payload]
        if not runs_list:
            continue

        run_corr_matrices = []

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
                    aug_minority_df = pd.DataFrame(X_gen_min, columns=feature_names)
                    try:
                        aug_corr = aug_minority_df.corr(method='kendall').fillna(0)
                        run_corr_matrices.append(aug_corr.values)
                    except Exception:
                        pass

        if run_corr_matrices:
            mean_corr_array = np.mean(run_corr_matrices, axis=0)
            mean_corr_df = pd.DataFrame(mean_corr_array, columns=feature_names, index=feature_names)
            mean_plot_items.append((f"{display_name}\nMean (Avg of {len(run_corr_matrices)} runs)", mean_corr_df, False))

            if len(run_corr_matrices) > 1:
                std_corr_array = np.std(run_corr_matrices, axis=0)
                std_corr_df = pd.DataFrame(std_corr_array, columns=feature_names, index=feature_names)
                std_plot_items.append((f"{display_name}\nInstability Map (Std Dev)", std_corr_df, True))

    def _draw_and_save_grid(items, suffix, title):
        n_plots = len(items)
        max_cols = 4
        n_cols = min(n_plots, max_cols)
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))
        
        if n_plots == 1:
            axes = [axes]
        else:
            axes = np.array(axes).flatten()

        mask = np.triu(np.ones_like(real_corr, dtype=bool))

        for idx, (plot_title, corr_matrix, is_std_dev) in enumerate(items):
            ax = axes[idx]
            
            cmap = 'Reds' if is_std_dev else 'PuOr'
            vmin = 0 if is_std_dev else -1
            vmax = 1.0
            center = None if is_std_dev else 0

            show_cbar = (idx == 0) or (idx == 1 and is_std_dev)
            cbar_label = "Std Deviation" if is_std_dev else "Kendall's τ"

            sns.heatmap(
                corr_matrix, mask=mask, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, center=center,
                square=True, linewidths=.5, cbar=show_cbar,
                cbar_kws={"shrink": .8, "label": cbar_label} if show_cbar else None, annot=False
            )
            
            ax.set_title(plot_title, fontsize=14, weight='semibold', pad=10)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=9)
            
            if idx % n_cols == 0:
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
            else:
                ax.set_yticks([])

        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        file_name = f"{save_path}/collage_kendall_{suffix}_{dataset_name}.pdf"
        plt.savefig(file_name, bbox_inches='tight')
        print(f"  Saved: {file_name}")
        plt.close(fig)

    if len(mean_plot_items) > 1:
        _draw_and_save_grid(
            mean_plot_items, 
            "MEAN", 
            f"Kendall's Rank Correlation (Mean) - Class: {minority_class}\nDataset: {dataset_name}"
        )
        
    if len(std_plot_items) > 1:
        _draw_and_save_grid(
            std_plot_items, 
            "INSTABILITY", 
            f"Kendall's Correlation Instability (Std Dev) - Class: {minority_class}\nDataset: {dataset_name}"
        )