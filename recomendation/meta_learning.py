import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

INPUT_FILE = 'best_methods_summary.csv'
MAX_DEPTH = 3


def main():
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: File not found {INPUT_FILE}")
        return

    print(f"Loaded {len(df)} datasets for training.")

    if 'Dataset_Name' in df.columns:
        dataset_names = df['Dataset_Name']
        X = df.drop(columns=['Dataset_Name', 'Best_Method'])
    else:
        dataset_names = df.index
        X = df.drop(columns=['Best_Method'])

    y = df['Best_Method']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    print(f"Detected classes (methods): {class_names}")
    print(f"Features used for decisions: {list(X.columns)}")

    print("\n--- Starting Leave-One-Out Validation ---")

    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []
    mistakes = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        clf = DecisionTreeClassifier(
            criterion='gini',
            max_depth=MAX_DEPTH,
            random_state=42,
            min_samples_leaf=1
        )
        clf.fit(X_train, y_train)

        prediction = clf.predict(X_test)[0]
        true_val = y_test[0]

        y_true_all.append(true_val)
        y_pred_all.append(prediction)

        if prediction != true_val:
            d_name = dataset_names.iloc[test_index[0]]
            actual_str = class_names[true_val]
            pred_str = class_names[prediction]
            mistakes.append(f"Dataset: {d_name} | Expected: {actual_str} -> Model predicted: {pred_str}")

    accuracy = accuracy_score(y_true_all, y_pred_all)
    print(f"\nAVERAGE ACCURACY: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all, target_names=class_names, zero_division=0))

    if mistakes:
        print("\n--- Error Analysis (Misclassified cases) ---")
        for m in mistakes:
            print(m)
    else:
        print("\nThe model predicted the correct method for every dataset!")

    print("\n--- Generating Final Decision Tree (PoC) ---")
    final_clf = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=42)
    final_clf.fit(X, y_encoded)

    tree_rules = export_text(final_clf, feature_names=list(X.columns))
    print("\nDiscovered Decision Rules:")
    print(tree_rules)

    plt.figure(figsize=(20, 10))
    plot_tree(final_clf,
              feature_names=X.columns,
              class_names=class_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title("Decision Tree Recommending Oversampling Method")

    output_img = 'meta_learning_tree.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Decision tree visualization saved to file: {output_img}")

    importances = pd.Series(final_clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nFeature Importance Ranking (Decision Drivers):")
    print(importances[importances > 0])


if __name__ == "__main__":
    main()