import os

datasets = [
    'Clean_blood.csv', 'Clean_Breast_Cancer_Wisconsin.csv', 'Clean_covid.csv', 
    'Clean_creditcard.csv', 'Clean_DDos.csv', 'Clean_haberman.csv', 'Clean_heart.csv', 
    'Clean_Infiltration.csv', 'Clean_ionosphere.csv', 'Clean_Loan_Default.csv', 
    'Clean_nsl_kdd_full_dataset.csv', 'Clean_pimadiabetes.csv', 'Clean_shuttle_full_dataset.csv', 
    'Clean_Titanic-Dataset.csv', 'Clean_weatherAUS.csv'
]
methods = [
    'smote', 'gan_from_tutorial', 'gan_paper', 'gan_paper_2', 'wgangp', 'ctgan', 'smotified_gan'
]

run=42
classifier='random_forest'

missing_runs = 0
with open('run_missing_visualizations.sh', 'in') as f:
    f.write('#!/bin/bash\n\n')
    for d in datasets:
        for m in methods:
            for use_iqr in [False, True]:
                dataset_name = d.replace('.csv', '')
                if use_iqr:
                    dataset_name = 'iqr_' + dataset_name
                
                tsne_file = f'results/tsne_{dataset_name}_{m}_rs42.pdf'
                kendall_file = f'results/kendall_correlation/corr_kendall_{dataset_name}_{m}.pdf'
                # Default path for histograms is likely results/feature_histograms based on make_image_collages.py
                hist_file = f'results/feature_histograms/hist_compare_{dataset_name}_{m}.pdf'
                
                if not (os.path.exists(tsne_file) and os.path.exists(kendall_file) and os.path.exists(hist_file)):
                    missing_runs += 1
                    f.write(f'echo "Running: {d}, {m}, IQR={use_iqr}"\n')
                    iqr_flag = '--use_iqr_preprocessing' if use_iqr else ''
                    f.write(f'python main.py --dataset "{d}" --balance "{m}" --classifier "{classifier}" --randomState "{run}" {iqr_flag} --skip_baseline --skip_classification --use_tsne --plot_histograms --plot_kendall_correlation\n\n')
print(f"Found missing runs: {missing_runs}")
