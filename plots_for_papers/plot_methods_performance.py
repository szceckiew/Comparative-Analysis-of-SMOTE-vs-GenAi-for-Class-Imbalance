import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_method_performance(csv_path):
    df = pd.read_csv(csv_path)
    
    df['Dataset'] = df['Dataset'].str.replace('Clean_', '').str.replace('.csv', '')
    
    pivot_df = df.pivot(index='Dataset', columns='Method', values='Final_Score')
    
    plt.figure(figsize=(14, 8))
    
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
    
    plt.title('Final Score Across Different Methods and Datasets', fontsize=16)
    plt.ylabel('Dataset', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(csv_path), '..', 'plots_for_papers', 'methods_performance_heatmap.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.show()
    
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "..", "recomendation", "all_methods_scores.csv")
    plot_method_performance(csv_file)