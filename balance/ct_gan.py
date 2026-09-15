import pandas as pd
import numpy as np


def apply_ctgan_oversample(x_train, y_train, epochs=10):
    """
    Wrapper for CTGAN (library version > 0.5.0).
    """
    # --- Lazy imports to prevent deadlocks ---
    import torch
    from ctgan import CTGAN

    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running CTGAN on device: {device}")

    # 1. Convert to DataFrame
    if isinstance(x_train, pd.DataFrame):
        X_df = x_train.copy()
        X_df.columns = [str(col) for col in X_df.columns]
    else:
        feature_columns = [str(i) for i in range(x_train.shape[1])]
        X_df = pd.DataFrame(x_train, columns=feature_columns)

    # 2. Identify minority class
    unique, counts = np.unique(y_train, return_counts=True)
    minority_class = unique[np.argmin(counts)]
    majority_count = np.max(counts)
    minority_count = np.min(counts)
    n_to_generate = majority_count - minority_count

    if n_to_generate <= 0:
        print("Data is balanced. Skipping CTGAN.")
        return x_train, y_train

    print(f"Minority class: {minority_class}. Generating {n_to_generate} samples.")

    # 3. Extract minority partition
    minority_indices = np.where(y_train == minority_class)[0]
    minority_df = X_df.iloc[minority_indices]

    discrete_columns = []
    # Inspect the full feature matrix (X_df) to determine unique value counts
    for col in X_df.columns:
        # If a column has binary or few unique values, treat as discrete.
        # A threshold of 10 provides a standard baseline for tabular data.
        if X_df[col].nunique() <= 10: 
            discrete_columns.append(col)

    if discrete_columns:
        print(f"Detected discrete columns ({len(discrete_columns)}): {discrete_columns}")
    else:
        print("No discrete columns detected. All features treated as continuous.")

    # 4. Initialization and training
    ctgan = CTGAN(epochs=epochs, verbose=True, cuda=(device == "cuda"))

    print(f"Starting CTGAN training ({epochs} epochs)...")
    
    # Pass identified discrete columns to CTGAN fit procedure
    ctgan.fit(minority_df, discrete_columns=discrete_columns)

    # 5. Synthetic generation
    generated_df = ctgan.sample(n_to_generate)

    # 6. Concatenation
    generated_data = generated_df.values
    generated_labels = np.full((n_to_generate,), minority_class)

    x_balanced = np.vstack([x_train, generated_data])
    y_balanced = np.hstack([y_train, generated_labels])

    print(f"CTGAN completed. Added {len(generated_data)} samples.")

    return x_balanced, y_balanced