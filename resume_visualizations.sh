#!/bin/bash

PYTHON_SCRIPT="main.py"

echo "Resume: Dataset=Clean_creditcard.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_creditcard.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_creditcard.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_creditcard.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_creditcard.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_creditcard.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_creditcard.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_creditcard.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_creditcard.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_creditcard.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_creditcard.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_creditcard.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_DDos.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_DDos.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_haberman.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_haberman.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_heart.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_heart.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Infiltration.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Infiltration.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_ionosphere.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_ionosphere.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Loan_Default.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Loan_Default.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_nsl_kdd_full_dataset.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_nsl_kdd_full_dataset.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_pimadiabetes.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_pimadiabetes.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_shuttle_full_dataset.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_shuttle_full_dataset.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_Titanic-Dataset.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_Titanic-Dataset.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=smote, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "smote" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=smote, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "smote" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=gan_from_tutorial, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=gan_from_tutorial, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "gan_from_tutorial" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=gan_paper, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=gan_paper, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "gan_paper" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=gan_paper_2, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=gan_paper_2, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "gan_paper_2" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=wgangp, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "wgangp" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=wgangp, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "wgangp" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=ctgan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "ctgan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=ctgan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "ctgan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=smotified_gan, IQR=False"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42"  --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

echo "Resume: Dataset=Clean_weatherAUS.csv, Balance=smotified_gan, IQR=True"
python $PYTHON_SCRIPT --dataset "Clean_weatherAUS.csv" --balance "smotified_gan" --classifier "random_forest" --randomState "42" --use_iqr_preprocessing --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation

