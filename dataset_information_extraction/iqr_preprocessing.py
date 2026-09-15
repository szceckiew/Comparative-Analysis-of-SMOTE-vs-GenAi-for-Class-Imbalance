import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """
    Scikit-learn style transformer that applies IQR capping.

    Learns statistics (fences) from the training dataset (fit method)
    and applies them to any dataset (transform method).
    This prevents data leakage.
    """

    def __init__(self, k=1.5):
        self.k = k
        self.fences_ = {}
        self.features_ = []

    def fit(self, X, y=None):
        """
        Computes and stores IQR fences for each feature in X.

        Args:
            X (pd.DataFrame): Training dataset used to learn statistics.
        """
        print("Learning IQR statistics (fences)...")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.features_ = X.columns.tolist()
        self.fences_ = {}

        for col in self.features_:
            try:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1

                if iqr == 0:
                    lower_fence = X[col].min()
                    upper_fence = X[col].max()
                else:
                    lower_fence = q1 - (self.k * iqr)
                    upper_fence = q3 + (self.k * iqr)

                self.fences_[col] = (lower_fence, upper_fence)

            except Exception as e:
                print(f"Error fitting column {col} (skipping): {e}")

        return self

    def transform(self, X):
        """
        Applies learned IQR fences to the dataset X.

        Args:
            X (pd.DataFrame): Dataset to transform (e.g., train, val, or test).
        """
        if not self.fences_:
            raise RuntimeError("Transformer has not been fitted. Call .fit() before .transform().")

        X_capped = X.copy()
        total_capped_count = 0

        if not isinstance(X_capped, pd.DataFrame):
            X_capped = pd.DataFrame(X_capped, columns=self.features_)

        for col in self.features_:
            if col in self.fences_:
                lower, upper = self.fences_[col]

                count_lower = (X_capped[col] < lower).sum()
                count_upper = (X_capped[col] > upper).sum()
                total_capped_count += (count_lower + count_upper)

                X_capped[col] = np.clip(X_capped[col], lower, upper)
            else:
                print(f"Warning: Column {col} was not present during .fit(), skipping.")

        print(f"   ... applied IQR capping, bounded {total_capped_count} values.")
        return X_capped