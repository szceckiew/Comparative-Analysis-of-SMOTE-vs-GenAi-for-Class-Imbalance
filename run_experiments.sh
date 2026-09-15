#!/bin/bash

# Stops the script if any of the processes encounters an error
#set -e

# --- Experiment Configuration ---

# Name of your Python script
PYTHON_SCRIPT="main.py"

# Baseline option: true/false. If false, we pass --skip_baseline to prevent redundant calculations.
# (Due to loops for 6 generative methods, gridsearch under baseline could run 6 times instead of once)
RUN_BASELINE=false
BASELINE_FLAG=""
if [ "$RUN_BASELINE" == "false" ] ; then
    BASELINE_FLAG="--skip_baseline"
fi

# Classifiers to test
CLASSIFIERS=(
    # "tune_hgb"
    "xgboost"
    # "lightgbm"
    # "random_forest"
)

# Number of repetitions for each combination (used as random seed)
TOTAL_RUNS=10

# Argument lists for iteration
DATASETS=(
    # --- Older datasets (commented out to avoid recalculating everything from scratch) ---
    # "Clean_blood.csv"
    # "Clean_Breast_Cancer_Wisconsin.csv"
    # "Clean_covid.csv"
    # "Clean_creditcard.csv"
    # "Clean_DDos.csv"
    # "Clean_haberman.csv"
    # "Clean_heart.csv"
    # "Clean_Infiltration.csv"
    # "Clean_ionosphere.csv"
    # "Clean_Loan_Default.csv"
    # "Clean_nsl_kdd_full_dataset.csv"
    # "Clean_pimadiabetes.csv"
    # "Clean_shuttle_full_dataset.csv"
    # "Clean_Titanic-Dataset.csv"
    # "Clean_weatherAUS.csv"
    
    # --- New, ideally balanced datasets for artificial down-sampling operations ---
    "Clean_spambase.csv"
    "Clean_parkinsons.csv"
    "Clean_sonar.csv"
    "Clean_mdlon.csv"
    "heart_balanced.csv"
    "balanced_artificial_dataset.csv"
    "Clean_banknote.csv"
)

# New 50/50 dataset split configuration
ARTIFICIAL_RATIO=0.1
RESULTS_DIR="results_artificial_imbalance"

BALANCE_METHODS=(
    # "smote"
    # "gan_paper"
    # "gan_paper_2"
    # "gan_from_tutorial"
    # "wgangp"
    # "ctgan"
    "smotified_gan"
)

# --- Main Loop ---

echo "Starting the sequence of experiments..."
echo "Python script: $PYTHON_SCRIPT"
echo "Number of runs per combination: $TOTAL_RUNS"
echo "-----------------------------------------------------"

# Loop from 1 to TOTAL_RUNS
for (( run=1; run<=$TOTAL_RUNS; run++ )); do

    echo "=== STARTING RUN $run/$TOTAL_RUNS ==="

    # Loop over classifiers
    for classifier in "${CLASSIFIERS[@]}"; do
        
        echo "--- CLASSIFIER: $classifier ---"

        # Loop over datasets
        for dataset in "${DATASETS[@]}"; do

            # Loop over balancing methods
            for balance in "${BALANCE_METHODS[@]}"; do

                # --- RUN 1: Standard (WITHOUT IQR) ---
                echo
                echo "--> Running (without IQR): Run=$run, Classifier=$classifier, Dataset=$dataset, Balance=$balance"

                python $PYTHON_SCRIPT \
                    --dataset "$dataset" \
                    --balance "$balance" \
                    --classifier "$classifier" \
                    --randomState "$run" \
                    --saveResults \
                    --results_dir "$RESULTS_DIR" \
                    --artificial_imbalance_ratio "$ARTIFICIAL_RATIO" \
                    --skip_baseline
                    # --skip_unfiltered \
                    # --plot_histograms \
                    

                echo "--> Completed (without IQR): Run=$run, Classifier=$classifier, Dataset=$dataset, Balance=$balance"
                echo "---------------------"

                # --- RUN 2: Additional (with IQR) ---
                echo
                echo "--> Running (with IQR): Run=$run, Classifier=$classifier, Dataset=$dataset, Balance=$balance"

                python $PYTHON_SCRIPT \
                    --dataset "$dataset" \
                    --balance "$balance" \
                    --classifier "$classifier" \
                    --randomState "$run" \
                    --saveResults \
                    --use_iqr_preprocessing \
                    --results_dir "$RESULTS_DIR" \
                    --artificial_imbalance_ratio "$ARTIFICIAL_RATIO" \
                    --skip_baseline
                    
                    # $BASELINE_FLAG  # <-- ADDED FLAG

                echo "--> Completed (with IQR): Run=$run, Classifier=$classifier, Dataset=$dataset, Balance=$balance"
                echo "---------------------"

            done # End of balance loop
        done # End of dataset loop
    done # End of classifier loop

    echo "=== COMPLETED RUN $run/$TOTAL_RUNS ==="
    echo "-----------------------------------------------------"

done # End of run loop

echo "All $TOTAL_RUNS experiment cycles (with and without IQR variants) have been completed."