import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def calculate_advanced_metrics(df, target_col='Class'):
    metrics = {}

    y = df[target_col]
    X = df.drop(columns=[target_col])

    metrics['Missing_Values_Ratio'] = df.isnull().mean().mean()

    numeric_df = X.select_dtypes(include=[np.number])

    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr().abs()
        upper_tri_values = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        ).stack()
        metrics['Avg_Feature_Correlation'] = upper_tri_values.mean()
    else:
        metrics['Avg_Feature_Correlation'] = 0.0

    if not numeric_df.empty:
        metrics['Mean_Skewness'] = numeric_df.skew().abs().mean()
    else:
        metrics['Mean_Skewness'] = 0.0

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_encoded = X_encoded.fillna(0)

    if len(y.unique()) > 1:
        try:
            lr_model = LogisticRegression(solver='liblinear', max_iter=100, random_state=42)
            lr_scores = cross_val_score(lr_model, X_encoded, y, cv=3, scoring='roc_auc')
            metrics['Linear_Separability_AUC'] = lr_scores.mean()
        except Exception:
            metrics['Linear_Separability_AUC'] = 0.5

        try:
            knn_model = KNeighborsClassifier(n_neighbors=1)
            knn_scores = cross_val_score(knn_model, X_encoded, y, cv=3, scoring='roc_auc')
            metrics['NonLinear_Separability_AUC'] = knn_scores.mean()
        except Exception:
            metrics['NonLinear_Separability_AUC'] = 0.5
    else:
        metrics['Linear_Separability_AUC'] = 0.0
        metrics['NonLinear_Separability_AUC'] = 0.0

    try:
        counts = y.value_counts()
        minority_class = counts.idxmin()
        minority_indices = np.where(y == minority_class)[0]
        minority_X = X_encoded.iloc[minority_indices]

        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_encoded)

        _, neighbor_indices = nn.kneighbors(minority_X)

        nearest_neighbor_classes = y.iloc[neighbor_indices[:, 1]].values

        metrics['Minority_Class_Overlap_Ratio'] = np.mean(nearest_neighbor_classes != minority_class)

        if len(minority_X) > 1:
            distances_min = pdist(minority_X, metric='euclidean')
            metrics['Minority_Avg_Distance'] = distances_min.mean()
        else:
            metrics['Minority_Avg_Distance'] = 0.0

    except Exception:
        metrics['Minority_Class_Overlap_Ratio'] = 0.0
        metrics['Minority_Avg_Distance'] = 0.0

    return metrics


def analyze_datasets(input_folder_path, output_csv_path):
    results = []
    folder = Path(input_folder_path)
    files = list(folder.glob("*.csv"))

    print(f"Found {len(files)} files. Starting analysis...\n")

    for file_path in files:
        try:
            df = pd.read_csv(file_path)

            if 'Class' not in df.columns:
                print(f"[SKIP] File {file_path.name} does not contain 'Class' column. Skipping.")
                continue

            y = df['Class']
            X = df.drop(columns=['Class'])

            n_samples = len(df)
            n_features = X.shape[1]

            counts = y.value_counts()
            minority_count = counts.min()
            majority_count = counts.max()

            minority_ratio = minority_count / n_samples
            imbalance_ratio = majority_count / minority_count if minority_count > 0 else 0

            cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
            cat_ratio = len(cat_cols) / n_features if n_features > 0 else 0

            samples_per_feature = n_samples / n_features if n_features > 0 else 0

            adv_metrics = calculate_advanced_metrics(df, target_col='Class')

            row = {
                'Dataset_Name': file_path.name,
                'n_samples': n_samples,
                'n_features': n_features,
                'Minority_Ratio': minority_ratio,
                'Imbalance_Ratio': imbalance_ratio,
                'Categorical_Ratio': cat_ratio,
                'Samples_Per_Feature': samples_per_feature,
                'Missing_Values_Ratio': adv_metrics['Missing_Values_Ratio'],
                'Avg_Correlation': adv_metrics['Avg_Feature_Correlation'],
                'Linearity_AUC': adv_metrics['Linear_Separability_AUC'],
                'NonLinear_AUC': adv_metrics['NonLinear_Separability_AUC'],
                'Mean_Skewness': adv_metrics['Mean_Skewness'],
                'Minority_Overlap_Ratio': adv_metrics['Minority_Class_Overlap_Ratio'],
                'Minority_Avg_Distance': adv_metrics['Minority_Avg_Distance']
            }

            results.append(row)
            print(f"[OK] Processed: {file_path.name}")

        except Exception as e:
            print(f"[ERROR] Error processing file {file_path.name}: {e}")

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='n_samples')

        results_df.to_csv(output_csv_path, index=False)
        print("\n--- COMPLETED ---")
        print(f"Report saved to: {output_csv_path}")
        print("Sample 5 rows:")
        print(results_df.head())
    else:
        print("\nFailed to generate report (no files found or errors occurred).")


input_folder = 'data/clean/'
output_file = 'meta_features_dataset.csv'

analyze_datasets(input_folder, output_file)