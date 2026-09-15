import os
import glob
from PIL import Image
import math

def create_collages_for_datasets():
    """
    Finds all generated plot files in results/ and aggregates them into composite boards
    (one per dataset, categorized by plot type: t-SNE, Kendall, Histograms).
    """
    results_dirs = {
        'tsne': 'results/',
        'kendall': 'results/kendall_correlation/',
        'histograms': 'results/feature_histograms/'
    }

    output_dir = 'plots_for_papers/collages'
    os.makedirs(output_dir, exist_ok=True)

    plot_types = {
        'tsne': {'prefix': 'tsne_', 'suffix': '.png'},
        'kendall': {'prefix': 'corr_kendall_', 'suffix': '.png'},
        'histograms': {'prefix': 'hist_compare_', 'suffix': '.png'}
    }

    rs_suffix = "_rs42"

    print("Starting collage generation...")

    for plot_type, format_info in plot_types.items():
        base_dir = results_dirs[plot_type]
        if not os.path.exists(base_dir):
            continue
            
        print(f"\n--- Processing image type: {plot_type.upper()} ---")
        
        search_pattern = os.path.join(base_dir, f"{format_info['prefix']}*{format_info['suffix']}")
        all_images = glob.glob(search_pattern)

        if not all_images:
            print(f"No images found for category: {plot_type}.")
            continue

        known_methods = ['baseline', 'smote', 'gan_from_tutorial', 'gan_paper', 'gan_paper_2', 'wgangp', 'ctgan', 'smotified_gan']
        methods_with_filtered = known_methods + [m + '_filtered' for m in known_methods]
        
        image_groups = {}
        
        for img_path in all_images:
            filename = os.path.basename(img_path)
            clean_name = filename.replace(format_info['prefix'], '').replace(format_info['suffix'], '')
            if plot_type == 'tsne':
                clean_name = clean_name.replace(rs_suffix, '')
                
            matched_method = None
            for method in sorted(methods_with_filtered, key=len, reverse=True):
                if clean_name.endswith(f"_{method}"):
                    matched_method = method
                    dataset_name = clean_name[:-(len(method)+1)]
                    break
                    
            if not matched_method:
                continue
                
            if dataset_name not in image_groups:
                image_groups[dataset_name] = {}
            image_groups[dataset_name][matched_method] = img_path
            
        for dataset_name, methods_dict in image_groups.items():
            if len(methods_dict) <= 1:
                continue
                
            sorted_methods = sorted(methods_dict.keys())
            images = [Image.open(methods_dict[m]) for m in sorted_methods]
            
            thumb_w, thumb_h = images[0].size
            
            n = len(images)
            cols = 2 if n <= 4 else 3
            rows = math.ceil(n / cols)
            
            collage_w = cols * thumb_w
            collage_h = rows * thumb_h
            
            collage = Image.new('RGB', (collage_w, collage_h), (255, 255, 255))
            
            for i, img in enumerate(images):
                row = i // cols
                col = i % cols
                collage.paste(img, (col * thumb_w, row * thumb_h))
                
            collage_name = f"{output_dir}/collage_{plot_type}_{dataset_name}.png"
            collage.save(collage_name)
            print(f"Created collage: {collage_name} (contains {n} methods)")

if __name__ == "__main__":
    create_collages_for_datasets()