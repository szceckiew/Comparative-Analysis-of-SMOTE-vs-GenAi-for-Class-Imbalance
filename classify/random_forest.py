import os
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef, average_precision_score, balanced_accuracy_score

def train_random_forest(x_train_balanced, y_train_balanced, X_test, y_test_np, random_state, balancing_method, dataset_name, generation_time=0.0):
    # Train classifier
    model = RandomForestClassifier(random_state=42)
    train_start = time.time()
    model.fit(x_train_balanced, y_train_balanced)
    train_time = time.time() - train_start

    # Prepare directories for result serialization
    try:
        abspath_curr = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        abspath_curr = os.getcwd()

    name = 'rf_tuning' # Consistent with naming schema
    results_dir = os.environ.get("RESULTS_DIR", "results")
    base_results_dir = os.path.join(abspath_curr, "..", results_dir, "cv_results", name)
    test_results_dir = os.path.join(base_results_dir, 'test_results')

    os.makedirs(test_results_dir, exist_ok=True)

    metrics_list = []
    
    # ------------------ EVALUATION (TEST METRICS) ------------------
    y_pred = model.predict(X_test)
    y_pred_proba_all = model.predict_proba(X_test)

    if y_pred_proba_all.shape[1] == 2:
        y_pred_proba = y_pred_proba_all[:, 1]
    else:
        y_pred_proba = y_pred_proba_all[:, 0]

    precision, recall, fscore, support = precision_recall_fscore_support(y_test_np, y_pred, zero_division=0)
    
    try:
        auc_binary = roc_auc_score(y_test_np, y_pred)
        auc_proba = roc_auc_score(y_test_np, y_pred_proba)
    except ValueError:
        auc_binary = 0.0
        auc_proba = 0.0

    mcc = matthews_corrcoef(y_test_np, y_pred)

    try:
        pr_auc = average_precision_score(y_test_np, y_pred_proba)
    except ValueError:
        pr_auc = 0.0
        
    bacc = balanced_accuracy_score(y_test_np, y_pred)

    data_real = np.column_stack((X_test, y_test_np))
    data_pred = np.column_stack((X_test, y_pred))

    corr_matrix_real = pd.DataFrame(data_real).corr(method='kendall').fillna(0).values
    corr_matrix_pred = pd.DataFrame(data_pred).corr(method='kendall').fillna(0).values
    diff_matrix = corr_matrix_real - corr_matrix_pred

    frobenius_kendall = np.linalg.norm(diff_matrix, ord='fro')
    
    precision_benign = precision[0] if len(precision) > 0 else 0
    precision_malignant = precision[1] if len(precision) > 1 else 0
    recall_benign = recall[0] if len(recall) > 0 else 0
    recall_malignant = recall[1] if len(recall) > 1 else 0
    fscore_benign = fscore[0] if len(fscore) > 0 else 0
    fscore_malignant = fscore[1] if len(fscore) > 1 else 0

    test_metrics = [
        precision_benign, precision_malignant,
        recall_benign, recall_malignant,
        fscore_benign, fscore_malignant,
        auc_binary,
        auc_proba,
        mcc,
        pr_auc,
        bacc,
        frobenius_kendall,
        train_time,
        generation_time,
        "RandomForestBaseline"
    ]
    metrics_list.append(test_metrics)

    columns = [
        'Precision (Benign)', 'Precision (Malignant)',
        'Recall (Benign)', 'Recall (Malignant)',
        'F1-score (Benign)', 'F1-score (Malignant)',
        'AUC (Binary)', 'AUC (Probability)',
        'MCC',
        'PR_AUC',
        'Balanced_Accuracy',
        'Frobenius_Kendall_Diff',
        'Training_Time',
        'Generation_Time',
        'Model'
    ]
    test_results = pd.DataFrame(metrics_list, columns=columns)

    test_filename = f"{dataset_name}-{balancing_method}-rf-test_results.csv"
    test_csv_path = os.path.join(test_results_dir, test_filename)

    if os.path.exists(test_csv_path):
        test_results.to_csv(path_or_buf=test_csv_path, index=False, mode='a', header=False)
    else:
        test_results.to_csv(path_or_buf=test_csv_path, index=False, mode='w', header=True)

    return model