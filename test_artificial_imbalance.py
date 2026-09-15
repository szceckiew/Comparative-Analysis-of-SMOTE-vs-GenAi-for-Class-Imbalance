import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cosine
import time

# Import balancing techniques
from balance.smote import apply_smote
from balance.gan_paper import apply_gan_paper_oversample
from balance.gan_from_tutorial import train_and_augment_gan_keras
from balance.wgangp import apply_wgangp_oversample
from balance.ct_gan import apply_ctgan_oversample
from balance.smotified_gan import apply_smotified_gan

def run_artificial_imbalance_experiment(csv_path, target_col, minority_keep_ratio=0.1):
    print(f"=== Reconstructing balanced dataset: {csv_path} ===")
    
    # 1. Data loading
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Restrict to binary 0/1 tasks for simplicity; designate least frequent class as minority
    counts = Counter(y)
    print(f"Original distribution: {counts}")
    
    # Sort classes by count (ascending) so [0] is minority, [-1] is majority
    sorted_classes = sorted(counts.keys(), key=lambda c: counts[c])
    minority_class = sorted_classes[0]
    majority_class = sorted_classes[-1]
    
    # 2. Artificial down-sampling
    mask_majority = (y == majority_class)
    mask_minority = (y == minority_class)
    
    X_maj = X[mask_majority]
    y_maj = y[mask_majority]
    
    X_min = X[mask_minority]
    y_min = y[mask_minority]
    
    n_keep = int(len(X_min) * minority_keep_ratio)
    
    # Safeguard against excessive sample removal (SMOTE requires k_neighbors=5, i.e., at least 6 samples)
    if n_keep < 6:
        n_keep = min(6, len(X_min))
        print(f" Minority guard: Ratio too small. Forcing n_keep={n_keep} to avoid SMOTE errors.")
    
    # Preserved samples (shuffled for uniformity)
    shuffled_indices = np.random.permutation(len(X_min))
    keep_indices = shuffled_indices[:n_keep]
    dropped_indices = shuffled_indices[n_keep:]
    
    X_min_keep = X_min.iloc[keep_indices]
    y_min_keep = y_min.iloc[keep_indices]
    
    X_min_dropped = X_min.iloc[dropped_indices]  # Ground-truth removed samples for fidelity evaluation
    
    X_imbalanced = pd.concat([X_maj, X_min_keep]).reset_index(drop=True)
    y_imbalanced = pd.concat([y_maj, y_min_keep]).reset_index(drop=True)
    
    print(f"After reduction, minority class ({minority_class}) contains: {len(y_min_keep)} samples")
    print(f"Samples missing for ground-truth reconstruction: {len(X_min_dropped)}")
    
    # Methods
    methods = {
        "smote": apply_smote,
        "ctgan": apply_ctgan_oversample,
        "gan_tutorial": train_and_augment_gan_keras,
        "gan_paper": apply_gan_paper_oversample,
        "wgangp": apply_wgangp_oversample,
        "smotified_gan": apply_smotified_gan
    }
    
    results = {}
    
    # 3. Model Fitting and Balancing
    for name, func in methods.items():
        print(f"\n--- Running method: {name} ---")
        start = time.time()
        try:
            # Ensure methods receive NumPy arrays where necessary for index masking
            if name != "ctgan" and name != "smote": # SMOTE and CTGAN natively support DataFrames; others expect NumPy
                X_res, y_res = func(X_imbalanced.values, y_imbalanced.values)
                # Convert back to DataFrame if required
                if isinstance(X_res, np.ndarray):
                    X_res = pd.DataFrame(X_res, columns=X.columns)
            else:
                X_res, y_res = func(X_imbalanced, y_imbalanced)
                if isinstance(X_res, np.ndarray):
                    X_res = pd.DataFrame(X_res, columns=X.columns)
            
            calc_time = time.time() - start
            
            # Target outputs may be returned as NumPy arrays
            if isinstance(y_res, pd.Series):
                y_res = y_res.values
            if isinstance(X_res, pd.DataFrame):
                pass
            
            mask_res_min = (y_res == minority_class)
            X_res_min = X_res[mask_res_min]
            
            # Compute evaluation statistics against ground-truth removed samples
            # Measure mean Wasserstein distance across all features
            w_dists = []
            for col in X.columns:
                wd = wasserstein_distance(X_min_dropped[col], X_res_min[col])
                w_dists.append(wd)
                
            mean_wd = np.mean(w_dists)
            
            # Difference of means via cosine similarity
            real_mean = X_min_dropped.mean(axis=0)
            gen_mean = X_res_min.mean(axis=0)
            cos_sim = 1 - cosine(real_mean, gen_mean)
            
            print(f"[{name}] Time: {calc_time:.2f}s | Mean Wasserstein Distance (lower is better): {mean_wd:.4f} | Mean Cosine Similarity (1=max): {cos_sim:.4f}")
            results[name] = {"Time(s)": calc_time, "Wasserstein_Dist": mean_wd, "Cosine_Similarity": cos_sim}
            
        except Exception as e:
            print(f"[{name}] Error executing method: {e}")
            
    # Summary of results
    print("\n=== SUMMARY ===")
    res_df = pd.DataFrame(results).T
    print(res_df.sort_values(by="Wasserstein_Dist"))
    
    # Append metadata columns to facilitate downstream evaluation
    res_df.index.name = 'Method'
    res_df['Dataset'] = csv_path.split('/')[-1].split('\\')[-1]
    res_df['Target_Col'] = target_col
    res_df['Keep_Ratio'] = minority_keep_ratio
    
    import os
    out_file = "artificial_imbalance_results.csv"
    
    # Append if file exists, otherwise initialize new CSV
    if os.path.exists(out_file):
        res_df.to_csv(out_file, mode='a', header=False, index=True)
        print(f"\nResults appended to {out_file}")
    else:
        res_df.to_csv(out_file, mode='w', header=True, index=True)
        print(f"\nResults written to newly created {out_file}")
    
    return res_df

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python test_artificial_imbalance.py <path_to_balanced_csv> <target_column_name>")
        sys.exit(1)
        
    path = sys.argv[1]
    target = sys.argv[2]
    
    # Optional parameter indicating retained minority percentage
    ratio = float(sys.argv[3]) if len(sys.argv) >= 4 else 0.1
    
    run_artificial_imbalance_experiment(path, target, ratio)