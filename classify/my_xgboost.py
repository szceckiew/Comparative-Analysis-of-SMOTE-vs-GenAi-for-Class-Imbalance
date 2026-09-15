import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, \
    matthews_corrcoef, average_precision_score, balanced_accuracy_score


def get_train_val_ps(X_train, y_train, X_val, y_val):
    """
    Get the:
    - feature matrix (X) and target vector (y) in the combined training and validation data
    - PredefinedSplit object (ps) for cross-validation
    """

    # Combine the feature matrix
    X_train_val = np.vstack((X_train, X_val))

    # Combine the target vector and ensure it is flat (1D)
    y_train_val = np.hstack((y_train.reshape(-1), y_val.reshape(-1)))

    # Get the indices for PredefinedSplit: -1 for training, 0 for validation
    train_val_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_val.shape[0], 0))

    # The PredefinedSplit object
    ps = PredefinedSplit(train_val_idxs)

    return X_train_val, y_train_val, ps


def training_validation_test(X_train_val, y_train_val, X_test, y_test, ps, abspath_curr, name, pipes, param_grids,
                           balancing_method, dataset_name, generation_time=0.0):
    """
    Training (with grid search and validation) and testing the models.
    """
    import numpy as np
    import pandas as pd
    import os
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, matthews_corrcoef

    # ************************************************************************************************
    # --- DIRECTORY CONFIGURATION ---
    base_results_dir = os.path.join(abspath_curr, name)
    grid_search_dir = os.path.join(base_results_dir, 'grid_search')
    test_results_dir = os.path.join(base_results_dir, 'test_results')

    os.makedirs(grid_search_dir, exist_ok=True)
    os.makedirs(test_results_dir, exist_ok=True)
    # ************************************************************************************************

    # ************************************************************************************************
    # Training and validation (Grid Search)

    best_score_param_estimator_gs = []
    acronym = 'xgb'

    # GridSearchCV
    gs = GridSearchCV(estimator=pipes[acronym],
                      param_grid=param_grids[acronym],
                      scoring='f1_macro',
                      n_jobs=1,  # Use 1 for GPU (no parallelism conflict)
                      cv=ps,
                      return_train_score=True,
                      verbose=1)

    # Fit the pipeline
    print(f"   -> Fitting {acronym} model with GridSearchCV...")
    train_start = time.time()
    gs = gs.fit(X_train_val, y_train_val)
    train_time = time.time() - train_start

    # Update best_score_param_estimator_gs
    best_score_param_estimator_gs.append([gs.best_score_, gs.best_params_, gs.best_estimator_])

    # Sort cv_results and save to CSV
    cv_results = pd.DataFrame.from_dict(gs.cv_results_).sort_values(by=['rank_test_score', 'std_test_score'])

    grid_filename = f"{dataset_name}-{balancing_method}-{acronym}-grid_result.csv"
    grid_csv_path = os.path.join(grid_search_dir, grid_filename)

    cv_results.to_csv(path_or_buf=grid_csv_path, index=False)
    print(f"   -> Saved Grid Search results to: {grid_csv_path}")

    print(f"   -> Best F1-score on validation set: {gs.best_score_:.4f}")
    print(f"   -> Best parameters: {gs.best_params_}")

    # ************************************************************************************************
    # Test

    metrics_list = []

    for best_score, best_param, best_estimator in best_score_param_estimator_gs:
        # Get the binary prediction
        y_pred = best_estimator.predict(X_test)

        # Compute standard classification metrics
        precision, recall, fscore, support = precision_recall_fscore_support(
            y_test, y_pred,
            average=None,
            labels=[0, 1],
            zero_division=0
        )
        auc_binary = roc_auc_score(y_test, y_pred)
        auc_proba = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[:, 1])
        mcc = matthews_corrcoef(y_test, y_pred)

        try:
            pr_auc = average_precision_score(y_test, best_estimator.predict_proba(X_test)[:, 1])
        except ValueError:
            pr_auc = 0.0
            
        bacc = balanced_accuracy_score(y_test, y_pred)

        # ************************************************************************************************
        # --- METRIC: Frobenius norm of Kendall rank correlation matrix difference ---
        # Assemble matrices: Features + Ground-Truth Target
        data_real = np.column_stack((X_test, y_test))
        # Assemble matrices: Features + Predicted Target
        data_pred = np.column_stack((X_test, y_pred))

        # Compute Kendall rank correlation matrices
        corr_matrix_real = pd.DataFrame(data_real).corr(method='kendall').fillna(0).values
        corr_matrix_pred = pd.DataFrame(data_pred).corr(method='kendall').fillna(0).values

        # Compute difference matrix
        diff_matrix = corr_matrix_real - corr_matrix_pred

        # Calculate Frobenius norm of matrix difference
        frobenius_kendall = np.linalg.norm(diff_matrix, ord='fro')
        # ************************************************************************************************

        precision_benign = precision[0]
        precision_malignant = precision[1]
        recall_benign = recall[0]
        recall_malignant = recall[1]
        fscore_benign = fscore[0]
        fscore_malignant = fscore[1]

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
            best_estimator
        ]
        metrics_list.append(test_metrics)

        # Save test evaluation metrics to CSV
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

        display_results = test_results.drop(columns=['Model']).round(4)

        print("\n**Test Results Summary (XGBoost):**")
        print(display_results.to_markdown(index=False))

        test_filename = f"{dataset_name}-{balancing_method}-xgb-test_results.csv"
        test_csv_path = os.path.join(test_results_dir, test_filename)

        if os.path.exists(test_csv_path):
            print(f"   -> Appending results to existing file: {test_csv_path}")
            test_results.to_csv(path_or_buf=test_csv_path, index=False, mode='a', header=False)
        else:
            print(f"   -> Creating new results file: {test_csv_path}")
            test_results.to_csv(path_or_buf=test_csv_path, index=False, mode='w', header=True)

        return best_estimator


def tune_xgb(x_train_balanced, y_train_balanced, x_val_scaled, y_val_np, x_test_scaled, y_test_np, random_state,
             balancing_method, dataset_name, generation_time=0.0, use_gpu=True):
    import time
    """
    Main function for XGBoost tuning with GPU support.
    
    Parameters:
    -----------
    use_gpu : bool
        If True, uses GPU for training (requires CUDA)
    """
    from xgboost import XGBClassifier
    from sklearn.pipeline import Pipeline

    # 1. Define Model, Pipeline, and Hyperparameter Grids
    
    # XGBoost GPU parameters
    xgb_params = {
        'random_state': random_state,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbosity': 0,
    }
    
    # Add GPU parameters
    if use_gpu:
        try:
            xgb_params['device'] = 'cuda'  # XGBoost 3.2.0
            xgb_params['tree_method'] = 'hist'  # Hist method works with CUDA
            print("   -> GPU mode (CUDA) enabled for XGBoost")
        except Exception as e:
            print(f"   -> GPU not available: {e}, falling back to CPU")
            xgb_params['device'] = 'cpu'
            xgb_params['tree_method'] = 'hist'
    else:
        xgb_params['device'] = 'cpu'
        xgb_params['tree_method'] = 'hist'
    
    xgb = XGBClassifier(**xgb_params)

    # Pipeline
    pipes = {'xgb': Pipeline([('model', xgb)])}

    # Grids - optimized for XGBoost (reduced from 180 to ~36 parameter combinations)
    param_grids = {
        'xgb': [
            {
                'model__learning_rate': [0.05, 0.1, 0.2],
                'model__max_depth': [5, 6, 7],
                'model__subsample': [0.8, 0.9],
                'model__colsample_bytree': [0.8, 0.9],
            }
        ]
    }

    # 2. Prepare combined data and PredefinedSplit
    print("   -> Preparing data for PredefinedSplit...")
    X_train_val, y_train_val, ps = get_train_val_ps(
        x_train_balanced, y_train_balanced, x_val_scaled, y_val_np
    )

    # 3. Training, Validation (Tuning), and Testing
    results_dir = os.environ.get("RESULTS_DIR", "results")
    try:
        abspath_curr = os.path.dirname(os.path.abspath(__file__)) + f'/../{results_dir}/cv_results/'
    except NameError:
        abspath_curr = os.path.abspath(os.path.join(os.getcwd(), '..', results_dir, 'cv_results')) + '/'
        print(f"Warning: __file__ not defined. Using relative path: {abspath_curr}")

    name = 'xgb_tuning'
    
    device_str = "GPU" if use_gpu else "CPU"
    print(f"   -> Training XGBoost on {device_str}")

    best_estimator = training_validation_test(
        X_train_val, y_train_val, x_test_scaled, y_test_np,
        ps, abspath_curr, name, pipes, param_grids,
        balancing_method, dataset_name, generation_time
    )

    return best_estimator