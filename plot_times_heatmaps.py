import os
import re
import glob
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def extract_info_from_filename(filename):
    basename = os.path.basename(filename)
    name_no_ext = basename.replace('-test_results.csv', '')
    parts = name_no_ext.split('-')
    if len(parts) >= 3:
        classifier = parts[-1]
        method = parts[-2]
        dataset = "-".join(parts[:-2])
        
        name_map = {
            'gan_from_tutorial': 'gan 1',
            'gan from tutorial': 'gan 1',
            'gan_paper': 'gan 2',
            'gan_paper_2': 'gan 2*'
        }
        if method in name_map:
            method = name_map[method]

        # Normalize dataset display name to match paper naming
        def map_display_name(raw: str) -> str:
            s = str(raw)
            s = re.sub(r'(?i)clean', '', s)
            s = re.sub(r'(?i)dataset', '', s)
            s = re.sub(r'(?i)balanced', '', s)
            s = s.replace('_', ' ').replace('-', ' ').strip().lower()
            if 'blood' in s:
                return 'Blood'
            if 'breast' in s and 'cancer' in s:
                return 'Breast Cancer Wisconsin'
            if 'covid' in s:
                return 'Covid-19'
            if 'credit' in s and 'card' in s:
                return 'Credit Card Fraud'
            if 'ddos' in s or 'cicids' in s:
                return 'DDoS'
            if 'haberman' in s:
                return "Haberman's Survival"
            if 'heart' in s:
                return 'Heart Disease'
            if 'infiltration' in s:
                return 'Infiltration'
            if 'ionosphere' in s:
                return 'Ionosphere'
            if 'loan' in s:
                return 'Loan Default'
            if 'nsl' in s or 'nsl kdd' in s:
                return 'NSL-KDD'
            if 'pima' in s or 'diabetes' in s:
                return 'Pima Indians Diabetes'
            if 'shuttle' in s:
                return 'Shuttle'
            if 'titanic' in s:
                return 'Titanic'
            if 'weather' in s or 'weatheraus' in s:
                return 'WeatherAUS'
            return ' '.join([w.capitalize() for w in s.split()])

        dataset = map_display_name(dataset)
            
        return dataset, method
    return None, None

def main():
    search_paths = ['results/**/*.csv', 'results_artificial_imbalance/**/*.csv']
    
    all_files = []
    for path in search_paths:
        all_files.extend(glob.glob(path, recursive=True))
        
    test_files = [f for f in all_files if 'test_results' in f]
    
    data_list = []
    
    for f in test_files:
        dataset, balance = extract_info_from_filename(f)
        if not dataset or not balance:
            continue
            
        try:
            df = pd.read_csv(f)
            if 'Training_Time' in df.columns and 'Generation_Time' in df.columns:
                avg_train = df['Training_Time'].mean()
                avg_gen = df['Generation_Time'].mean()
                data_list.append({
                    'Dataset': dataset,
                    'Method': balance,
                    'Training_Time': avg_train,
                    'Generation_Time': avg_gen
                })
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not data_list:
        print("No valid data found.")
        return
        
    results_df = pd.DataFrame(data_list)
    
    # Average again grouping by Dataset and Method
    agg_df = results_df.groupby(['Dataset', 'Method']).mean().reset_index()
    
    # Pivot for heatmaps
    train_pivot = agg_df.pivot(index='Dataset', columns='Method', values='Training_Time')
    gen_pivot = agg_df.pivot(index='Dataset', columns='Method', values='Generation_Time')
    
    if 'baseline' in train_pivot.columns:
        train_pivot['has_baseline'] = train_pivot['baseline'].notna()
        train_pivot = train_pivot.sort_values(by=['has_baseline', 'Dataset'], ascending=[False, True])
        train_pivot = train_pivot.drop(columns=['has_baseline'])
        
    if 'baseline' in gen_pivot.columns:
        gen_pivot['has_baseline'] = gen_pivot['baseline'].notna()
        gen_pivot = gen_pivot.sort_values(by=['has_baseline', 'Dataset'], ascending=[False, True])
        gen_pivot = gen_pivot.drop(columns=['has_baseline'])
    
    # Plot Training Time
    plt.figure(figsize=(14, 18))
    sns.heatmap(train_pivot, annot=True, cmap='viridis', fmt='.1f', annot_kws={"size": 8})
    plt.xlabel('Oversampling Method')
    plt.ylabel('Dataset')
    plt.tight_layout()
    plt.savefig('training_time_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Plot Generation Time
    plt.figure(figsize=(14, 18))
    sns.heatmap(gen_pivot, annot=True, cmap='plasma', fmt='.1f', annot_kws={"size": 8})
    plt.xlabel('Oversampling Method')
    plt.ylabel('Dataset')
    plt.tight_layout()
    plt.savefig('generation_time_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
