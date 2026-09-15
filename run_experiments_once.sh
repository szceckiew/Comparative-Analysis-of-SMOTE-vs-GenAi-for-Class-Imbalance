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

BALANCE_METHODS=(
#    "smote"
#    "gan_paper_2"
#    "gan_from_tutorial"
#    "wgangp"
    "ctgan"
    # "smotified_gan"
)

CLASSIFIER=(
    # "random_forest"
    # "tune_hgb"
    "lightgbm"
    # "xgboost"
)


for dataset in "${DATASETS[@]}"; do

    # Loop over balancing methods
    for balance in "${BALANCE_METHODS[@]}"; do

        # Loop over classifiers
        for classifier in "${CLASSIFIER[@]}"; do

            python wrapper.py \
                --dataset="$dataset" \
                --balance="$balance" \
                --classifier="$classifier" \
                --saveResults \
                --use_additional_eval \
                --plot_histograms \
                --plot_kendall_correlation

            echo "--> Completed (without IQR): Dataset=$dataset, Balance=$balance, Classifier=$classifier"
            echo "---------------------"

            # --- RUN 2: Additional (with IQR) ---
            echo
            echo "--> Running (with IQR): Dataset=$dataset, Balance=$balance, Classifier=$classifier"

            python wrapper.py \
                --dataset="$dataset" \
                --balance="$balance" \
                --classifier="$classifier" \
                --saveResults \
                --use_additional_eval \
                --plot_histograms \
                --plot_kendall_correlation \
                --use_iqr_preprocessing

            echo "--> Completed (with IQR): Dataset=$dataset, Balance=$balance, Classifier=$classifier"
            echo "---------------------"

        done # End of classifier loop
    done # End of balance loop
done # End of dataset loop