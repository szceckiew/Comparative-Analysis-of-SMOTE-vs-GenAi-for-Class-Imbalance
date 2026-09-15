#!/bin/bash

# --- Visualization Experiment Configuration for Artificial Imbalance ---

PYTHON_SCRIPT="main.py"

CLASSIFIER="xgboost"
RUN=42 

DATASETS=(
    "Clean_spambase.csv"
    "Clean_parkinsons.csv"
    "Clean_sonar.csv"
    "Clean_mdlon.csv"
    "heart_balanced.csv"
    "balanced_artificial_dataset.csv"
    "Clean_banknote.csv"
)

BALANCE_METHODS=(
    "smote"
    "gan_from_tutorial"
    "gan_paper"
    "gan_paper_2"
    "wgangp"
    "ctgan"
    "smotified_gan"
)

ARTIFICIAL_RATIO=0.1
RESULTS_DIR="results_artificial_imbalance"

echo "Starting visualization generation (Kendall correlation, histograms) for ARTIFICIAL IMBALANCE..."

for dataset in "${DATASETS[@]}"; do
    for balance in "${BALANCE_METHODS[@]}"; do
        
        echo "--> Generating (without IQR) for: Dataset=$dataset, Balance=$balance"
        
        python $PYTHON_SCRIPT \
            --dataset "$dataset" \
            --balance "$balance" \
            --classifier "$CLASSIFIER" \
            --randomState "$RUN" \
            --skip_baseline \
            --skip_classification \
            --plot_histograms \
            --plot_kendall_correlation \
            --artificial_imbalance_ratio "$ARTIFICIAL_RATIO" \
            --results_dir "$RESULTS_DIR"

        echo "--> Completed (without IQR): $dataset, $balance"
        echo "---------------------"

        echo "--> Generating (with IQR) for: Dataset=$dataset, Balance=$balance"
        
        python $PYTHON_SCRIPT \
            --dataset "$dataset" \
            --balance "$balance" \
            --classifier "$CLASSIFIER" \
            --randomState "$RUN" \
            --use_iqr_preprocessing \
            --skip_baseline \
            --skip_classification \
            --plot_histograms \
            --plot_kendall_correlation \
            --artificial_imbalance_ratio "$ARTIFICIAL_RATIO" \
            --results_dir "$RESULTS_DIR"

        echo "--> Completed (with IQR): $dataset, $balance"
        echo "---------------------"

    done
done

echo "All artificial imbalance visualization tasks completed!"