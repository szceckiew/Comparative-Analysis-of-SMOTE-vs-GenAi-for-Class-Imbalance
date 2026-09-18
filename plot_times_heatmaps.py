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
        
        is_filtered = False
        if method.endswith('_filtered'):
            is_filtered = True
            method = method.replace('_filtered', '')
        elif method.endswith(' filtered'):
            is_filtered = True
            method = method.replace(' filtered', '')

        name_map = {
            'gan_from_tutorial': 'gan 1',
            'gan from tutorial': 'gan 1',
            'gan_paper': 'gan 2',
            'gan paper': 'gan 2',
            'gan_paper_2': 'gan 2*',
            'gan paper 2': 'gan 2*',
            'smotified_gan': 'smotified gan'
        }
        
        if method in name_map:
            method = name_map[method]
        else:
            method = method.replace('_', ' ')

        if is_filtered:
            method = f"{method} filtered"

        def map_display_name(raw: str) -> str:
            s = str(raw)
            is_iqr = bool(re.search(r'(?i)iqr', s))
            
            s = re.sub(r'(?i)clean', '', s)
            s = re.sub(r'(?i)dataset', '', s)
            s = re.sub(r'(?i)balanced', '', s)
            s = re.sub(r'(?i)iqr', '', s)
            s = s.replace('_', ' ').replace('-', ' ').strip().lower()
            
            base_name = ""
            if 'blood' in s:
                base_name = 'Blood'
            elif 'breast' in s and 'cancer' in s:
                base_name = 'Breast Cancer Wisconsin'
            elif 'covid' in s:
                base_name = 'Covid-19'
            elif 'credit' in s and 'card' in s:
                base_name = 'Credit Card Fraud'
            elif 'ddos' in s or 'cicids' in s:
                base_name = 'DDoS'
            elif 'haberman' in s:
                base_name = "Haberman's Survival"
            elif 'heart' in s:
                base_name = 'Heart Disease'
            elif 'infiltration' in s:
                base_name = 'Infiltration'
            elif 'ionosphere' in s:
                base_name = 'Ionosphere'
            elif 'madelon' in s or 'mdlon' in s:
                base_name = 'Madelon'
            elif 'loan' in s:
                base_name = 'Loan Default'
            elif 'nsl' in s or 'nsl kdd' in s:
                base_name = 'NSL-KDD'
            elif 'pima' in s or 'diabetes' in s:
                base_name = 'Pima Indians Diabetes'
            elif 'shuttle' in s:
                base_name = 'Shuttle'
            elif 'titanic' in s:
                base_name = 'Titanic'
            elif 'weather' in s or 'weatheraus' in s:
                base_name = 'WeatherAUS'
            else:
                base_name = ' '.join([w.capitalize() for w in s.split()])
                
            if is_iqr:
                return f"IQR {base_name}"
            return base_name

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
    
    agg_df = results_df.groupby(['Dataset', 'Method']).mean().reset_index()
    
    train_pivot = agg_df.pivot(index='Dataset', columns='Method', values='Training_Time')
    gen_pivot = agg_df.pivot(index='Dataset', columns='Method', values='Generation_Time')
    
    train_pivot = train_pivot.reset_index()
    train_pivot['is_iqr'] = train_pivot['Dataset'].str.startswith('IQR')
    if 'baseline' in train_pivot.columns:
        train_pivot['has_baseline'] = train_pivot['baseline'].notna()
        train_pivot = train_pivot.sort_values(by=['has_baseline', 'is_iqr', 'Dataset'], ascending=[False, True, True])
        train_pivot = train_pivot.drop(columns=['has_baseline'])
    else:
        train_pivot = train_pivot.sort_values(by=['is_iqr', 'Dataset'], ascending=[True, True])
    train_pivot = train_pivot.drop(columns=['is_iqr']).set_index('Dataset')
        
    gen_pivot = gen_pivot.reset_index()
    gen_pivot['is_iqr'] = gen_pivot['Dataset'].str.startswith('IQR')
    if 'baseline' in gen_pivot.columns:
        gen_pivot['has_baseline'] = gen_pivot['baseline'].notna()
        gen_pivot = gen_pivot.sort_values(by=['has_baseline', 'is_iqr', 'Dataset'], ascending=[False, True, True])
        gen_pivot = gen_pivot.drop(columns=['has_baseline'])
    else:
        gen_pivot = gen_pivot.sort_values(by=['is_iqr', 'Dataset'], ascending=[True, True])
    gen_pivot = gen_pivot.drop(columns=['is_iqr']).set_index('Dataset')
    
    plt.figure(figsize=(14, 18))
    sns.heatmap(train_pivot, annot=True, cmap='viridis', fmt='.1f', annot_kws={"size": 8})
    plt.xlabel('Oversampling Method')
    plt.ylabel('Dataset')
    plt.tight_layout()
    plt.savefig('training_time_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(14, 18))
    sns.heatmap(gen_pivot, annot=True, cmap='plasma', fmt='.1f', annot_kws={"size": 8})
    plt.xlabel('Oversampling Method')
    plt.ylabel('Dataset')
    plt.tight_layout()
    plt.savefig('generation_time_heatmap.pdf', format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()