import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

INPUT_FILE = 'best_methods_summary.csv'


def simplify_method_name(method_name):
    """
    Maps specific methods into 3 primary families.
    """
    method = str(method_name).lower()

    if any(x in method for x in ['ctgan', 'wgangp', 'smotified_gan']):
        return 'Advanced_GAN'
    elif any(x in method for x in ['gan_paper', 'gan_from_tutorial', 'vanilla']):
        return 'Simple_GAN'
    elif 'smote' in method:
        return 'SMOTE'
    else:
        return 'Other'


def preprocess_features(X):
    """
    Transforms features to facilitate learning (Log Transform for skewed features).
    """
    X_new = X.copy()

    skewed_features = ['n_samples', 'Imbalance_Ratio', 'Samples_Per_Feature']

    for col in skewed_features:
        if col in X_new.columns:
            X_new[col] = np.log1p(X_new[col])
            print(f"Applied log-transformation to column: {col}")

    return X_new


def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File not found {INPUT_FILE}")
        return

    df['Method_Family'] = df['Best_Method'].apply(simplify_method_name)

    df = df[df['Method_Family'] != 'Other']

    print("--- Class distribution (Target) ---")
    print(df['Method_Family'].value_counts())

    cols_to_drop = ['Dataset_Name', 'Best_Method', 'Method_Family', 'Join_Key']
    X = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    y = df['Method_Family']

    X = preprocess_features(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"Classes: {class_names}")

    print("\n--- LOOCV Validation (Random Forest) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    loo = LeaveOneOut()

    y_true = []
    y_pred = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        rf.fit(X_train, y_train)
        pred = rf.predict(X_test)[0]

        y_true.append(y_test[0])
        y_pred.append(pred)

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    print("\n--- Generating Decision Tree (PoC) ---")
    clf = DecisionTreeClassifier(max_depth=2, random_state=42, class_weight='balanced')
    clf.fit(X, y_encoded)

    plt.figure(figsize=(16, 10))
    plot_tree(clf,
              feature_names=X.columns,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=11)

    plt.savefig("meta_learning_tree_3groups.pdf", bbox_inches='tight')
    print("Plot saved to: meta_learning_tree_3groups.pdf")

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features driving selection:")
    print(importances.head(5))


if __name__ == "__main__":
    main()