#!/bin/bash

# Terminate the script if any process returns an error
#set -e

# --- Visualization Experiment Configuration ---

PYTHON_SCRIPT="main.py"

# Use a single classifier and fixed seed to generate representative statistics
CLASSIFIER="random_forest"
RUN=42 # Fixed random state

DATASETS=(
    "Clean_blood.csv"
    "Clean_Breast_Cancer_Wisconsin.csv"
    "Clean_covid.csv"
    "Clean_creditcard.csv"
    "Clean_DDos.csv"
    "Clean_haberman.csv"
    "Clean_heart.csv"
    "Clean_Infiltration.csv"
    "Clean_ionosphere.csv"
    "Clean_Loan_Default.csv"
    "Clean_nsl_kdd_full_dataset.csv"
    "Clean_pimadiabetes.csv"
    "Clean_shuttle_full_dataset.csv"
    "Clean_Titanic-Dataset.csv"
    "Clean_weatherAUS.csv"
)

# Full iteration over available augmentation strategies
BALANCE_METHODS=(
    "smote"
    "gan_from_tutorial"
    "gan_paper"
    "gan_paper_2"
    "wgangp"
    "ctgan"
    "smotified_gan"
)

echo "Starting generation of visualization plots (t-SNE, Kendall correlation, histograms)..."
echo "Classifier used for pipeline execution: $CLASSIFIER"
echo "Seed: $RUN"
echo "-----------------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    for balance in "${BALANCE_METHODS[@]}"; do
        
        # --- 1. Without IQR ---
        echo
        echo "--> Generating (without IQR) for: Dataset=$dataset, Balance=$balance"
        
        python $PYTHON_SCRIPT \
            --dataset "$dataset" \
            --balance "$balance" \
            --classifier "$CLASSIFIER" \
            --randomState "$RUN" \
            --skip_baseline \
            --skip_classification \
            --plot_histograms \
            --plot_kendall_correlation

        echo "--> Completed (without IQR): $dataset, $balance"
        echo "---------------------"

        # --- 2. With IQR ---
        echo
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
            --plot_kendall_correlation

        echo "--> Completed (with IQR): $dataset, $balance"
        echo "---------------------"

    done
done

echo "All visualization execution jobs have been submitted!"