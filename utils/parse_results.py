import os
import csv

RESULTS_DIR = '../results'
OUTPUT_CSV = 'basic_results_summary.csv'

def parse_classification_report(filepath):
    metrics = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            
            if parts[0] == 'accuracy':
                metrics['accuracy'] = float(parts[1])
            elif len(parts) >= 5 and parts[0] == 'macro' and parts[1] == 'avg':
                metrics['macro_precision'] = float(parts[2])
                metrics['macro_recall'] = float(parts[3])
                metrics['macro_f1'] = float(parts[4])
            elif len(parts) >= 5 and parts[0] == 'weighted' and parts[1] == 'avg':
                metrics['weighted_precision'] = float(parts[2])
                metrics['weighted_recall'] = float(parts[3])
                metrics['weighted_f1'] = float(parts[4])
            elif parts[0] in ['0', '1', '0.0', '1.0'] and len(parts) >= 4:
                prefix = f'class_{parts[0]}'
                metrics[f'{prefix}_precision'] = float(parts[1])
                metrics[f'{prefix}_recall'] = float(parts[2])
                metrics[f'{prefix}_f1'] = float(parts[3])
                
    return metrics

def main():
    all_data = []
    
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.txt'):
            continue
            
        filepath = os.path.join(RESULTS_DIR, filename)
        
        name_without_ext = filename[:-4]
        name_parts = name_without_ext.split('-')
        
        if len(name_parts) >= 4:
            seed = name_parts[0]
            dataset = name_parts[1]
            method = name_parts[2]
            classifier = '-'.join(name_parts[3:])
        else:
            seed = "N/A"
            dataset = "N/A"
            method = "N/A"
            classifier = name_without_ext
            
        row = {
            'filename': filename,
            'seed': seed,
            'dataset': dataset,
            'method': method,
            'classifier': classifier
        }
        
        metrics = parse_classification_report(filepath)
        row.update(metrics)
        
        all_data.append(row)

    if not all_data:
        print("No results found.")
        return

    fieldnames = ['filename', 'seed', 'dataset', 'method', 'classifier', 'accuracy',
                  'macro_precision', 'macro_recall', 'macro_f1', 
                  'weighted_precision', 'weighted_recall', 'weighted_f1',
                  'class_0_precision', 'class_0_recall', 'class_0_f1',
                  'class_1_precision', 'class_1_recall', 'class_1_f1']

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_data)
        
    print(f"Successfully saved results to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()