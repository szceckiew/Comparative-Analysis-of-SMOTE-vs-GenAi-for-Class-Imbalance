import pandas as pd
import numpy as np
import os

from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import pairwise_distances


def evaluate_synthetic_quality(
        X_real: np.ndarray,
        y_real: np.ndarray,
        X_augmented: np.ndarray,
        y_augmented: np.ndarray,
        dataset_name: str,
        balance_method_name: str,
        output_csv_path: str = None,
):
    """
    Computes quality metrics for generated synthetic data.
    Compares ORIGINAL minority class samples with NEW synthetic samples.
    Optionally saves the results to a CSV file.
    """
    print(f"\n--- Evaluating synthetic data quality for: {balance_method_name} ---")

    if output_csv_path is None:
        output_csv_path = os.path.join(os.environ.get("RESULTS_DIR", "results"), "cv_results", "dataset_metrics")

    if not output_csv_path.endswith('.csv'):
        output_csv_path = os.path.join(output_csv_path, f"{dataset_name}-{balance_method_name}-dataset_metrics.csv")

    try:
        mean_dist_real = np.nan
        mean_dist_synth = np.nan
        mean_dist_real_synth = np.nan

        counts = pd.Series(y_real).value_counts()
        minority_class = counts.idxmin()
        n_original = len(y_real)
        X_real_minority = X_real[y_real == minority_class]

        X_new_samples = X_augmented[n_original:]
        y_new_samples = y_augmented[n_original:]
        X_synthetic_minority = X_new_samples[y_new_samples == minority_class]

        if len(X_real_minority) == 0:
            print("ERROR: No minority class samples found in real data.")
            return
        if len(X_synthetic_minority) == 0:
            print("INFO: No new synthetic samples available for comparison.")
            return

        print(
            f"Comparing {len(X_real_minority)} real samples (class {minority_class}) with {len(X_synthetic_minority)} synthetic samples."
        )

        n_features = X_real.shape[1]
        w_distances = []
        for i in range(n_features):
            w_dist = wasserstein_distance(
                X_real_minority[:, i],
                X_synthetic_minority[:, i]
            )
            w_distances.append(w_dist)

        mean_w_distance = np.mean(w_distances)
        print(f"\n[Metric] Mean Wasserstein Distance (per feature): {mean_w_distance:.4f}")

        n_bins = 30
        kl_divergences = []
        for i in range(n_features):
            min_val = min(X_real_minority[:, i].min(), X_synthetic_minority[:, i].min())
            max_val = max(X_real_minority[:, i].max(), X_synthetic_minority[:, i].max())
            bins = np.linspace(min_val, max_val, n_bins + 1)

            hist_real, _ = np.histogram(X_real_minority[:, i], bins=bins, density=False)
            pmf_real = (hist_real / hist_real.sum()) + 1e-9

            hist_synth, _ = np.histogram(X_synthetic_minority[:, i], bins=bins, density=False)
            pmf_synth = (hist_synth / hist_synth.sum()) + 1e-9

            kl_div = entropy(pmf_real, pmf_synth)
            kl_divergences.append(kl_div)

        mean_kl_divergence = np.mean(kl_divergences)
        print(f"[Metric] Mean KL Divergence (Entropy):        {mean_kl_divergence:.4f}")

        sample_size = min(500, len(X_real_minority), len(X_synthetic_minority))
        if sample_size < 10:
            print("Insufficient samples to calculate Pairwise Distance.")
        else:
            idx_real = np.random.choice(len(X_real_minority), sample_size, replace=False)
            idx_synth = np.random.choice(len(X_synthetic_minority), sample_size, replace=False)

            X_real_sample = X_real_minority[idx_real]
            X_synth_sample = X_synthetic_minority[idx_synth]

            dist_real = pairwise_distances(X_real_sample)
            mean_dist_real = np.mean(dist_real[np.triu_indices(sample_size, k=1)])

            dist_synth = pairwise_distances(X_synth_sample)
            mean_dist_synth = np.mean(dist_synth[np.triu_indices(sample_size, k=1)])

            dist_real_synth = pairwise_distances(X_real_sample, X_synth_sample)
            mean_dist_real_synth = np.mean(dist_real_synth)

            print("\n[Metric] Mean Pairwise Distance (on sampled subsets):")
            print(f"  Real-Real:   {mean_dist_real:.4f}")
            print(f"  Synth-Synth: {mean_dist_synth:.4f} (closer to Real-Real is better)")
            print(f"  Real-Synth:  {mean_dist_real_synth:.4f}")

        print("--- End of quality evaluation ---")

        if output_csv_path:
            try:
                output_directory = os.path.dirname(output_csv_path)
                if output_directory:
                    os.makedirs(output_directory, exist_ok=True)
            except Exception as e:
                print(f"WARNING: Could not create directory {output_directory}. Error: {e}")

            results = {
                "dataset": dataset_name,
                "balance_method": balance_method_name,
                "wasserstein_mean": mean_w_distance,
                "kl_divergence_mean": mean_kl_divergence,
                "pairwise_real_real": mean_dist_real,
                "pairwise_synth_synth": mean_dist_synth,
                "pairwise_real_synth": mean_dist_real_synth
            }

            df_results = pd.DataFrame([results])
            file_exists = os.path.isfile(output_csv_path)

            df_results.to_csv(
                output_csv_path,
                mode='a',
                header=not file_exists,
                index=False,
                encoding='utf-8'
            )
            print(f"Successfully saved quality metrics to: {output_csv_path}")

    except Exception as e:
        print(f"ERROR calculating synthetic data quality metrics: {e}")
        import traceback
        traceback.print_exc()