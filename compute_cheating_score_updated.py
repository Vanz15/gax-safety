"""
Multi-Metric Cheating Score Calculator
======================================
Difference from previous version:
The original script calculated a single "Base Cheating Score" by measuring all 
positive GAX attributions outside the segmented lung mask. 

This updated version introduces a rigorous multi-metric evaluation framework to 
account for saliency map noise and provide stricter proof of shortcut learning:
1. Base Score: Measures total positive attribution outside the ROI.
2. Thresholded Score: Filters out low-level noise by evaluating only the top 20% 
   strongest attributions, isolating the model's core focus.
3. Pointing Game: A strict binary test that checks if the absolute most important 
   pixel (max attribution) is located outside the lung mask.
"""

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import argparse

def get_positive_heatmap(raw_mask):
    """Flattens the 3-channel GAX mask and extracts only positive (red) attributions."""
    if raw_mask.shape[-1] in [1, 3]:
        h_sum = np.sum(raw_mask, axis=2)
    else:
        h_sum = raw_mask
        
    positive_attr = np.maximum(h_sum, 0)
    return positive_attr

def compute_score(positive_heatmap, binary_mask):
    """Calculates the percentage of ALL positive attribution outside the lung mask."""
    inverse_mask = 1.0 - binary_mask
    total_attribution = np.sum(positive_heatmap)
    if total_attribution == 0:
        return 0.0 
        
    outside_attribution = np.sum(positive_heatmap * inverse_mask)
    return float(outside_attribution / total_attribution)

def compute_thresholded_score(positive_heatmap, binary_mask, threshold_ratio=0.20):
    """Calculates the score using ONLY the top 20% strongest attributions, filtering out noise."""
    max_val = np.max(positive_heatmap)
    threshold_val = max_val * threshold_ratio
    
    # Zero out all weak/noisy pixels
    strong_heatmap = np.where(positive_heatmap >= threshold_val, positive_heatmap, 0)
    
    inverse_mask = 1.0 - binary_mask
    total_strong_attr = np.sum(strong_heatmap)
    
    if total_strong_attr == 0:
        return 0.0
        
    outside_strong_attr = np.sum(strong_heatmap * inverse_mask)
    return float(outside_strong_attr / total_strong_attr)

def compute_pointing_game(positive_heatmap, binary_mask):
    """Returns True if the absolute highest attribution pixel is outside the lungs."""
    # Find the 2D coordinates of the maximum value
    max_idx = np.unravel_index(np.argmax(positive_heatmap, axis=None), positive_heatmap.shape)
    
    # Check the binary mask at that exact coordinate (0 = Background/Cheating, 1 = Lungs)
    is_cheating = binary_mask[max_idx] == 0
    return bool(is_cheating)

def main():
    parser = argparse.ArgumentParser(description="Compute multi-metric cheating scores for a given model's GAX output")
    parser.add_argument("--gax_dir", type=str, default="results/resnet34_v3/gax_images", help="Directory containing the GAX heatmaps")
    parser.add_argument("--output_csv", type=str, default="results/resnet34_v3/cheating_scores_v2.csv", help="Path to save the results CSV")
    args = parser.parse_args()

    # --- Configurations ---
    MASK_DIR = "masked_dataset/test"
    GAX_DIR = args.gax_dir
    OUTPUT_CSV = args.output_csv
    
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    results = []
    
    if not os.path.exists(MASK_DIR):
        print(f"Error: Could not find {MASK_DIR}")
        return
        
    class_folders = [f for f in os.listdir(MASK_DIR) if os.path.isdir(os.path.join(MASK_DIR, f))]
    
    print(f"Starting Multi-Metric Cheating Score calculation...\nScanning GAX Directory: {GAX_DIR}")
    
    for class_name in class_folders:
        class_path = os.path.join(MASK_DIR, class_name)
        img_files = [f for f in os.listdir(class_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        for img_name in tqdm(img_files, desc=f"Scoring {class_name}"):
            mask_path = os.path.join(class_path, img_name)
            
            # Check for BOTH potential naming conventions from your previous scripts
            gax_filename_mult = f"op.{img_name}.test.mult.npy"
            gax_filename_sum = f"op.{img_name}.test.sum.npy"
            
            gax_path_mult = os.path.join(GAX_DIR, gax_filename_mult)
            gax_path_sum = os.path.join(GAX_DIR, gax_filename_sum)
            
            if os.path.exists(gax_path_mult):
                gax_path = gax_path_mult
            elif os.path.exists(gax_path_sum):
                gax_path = gax_path_sum
            else:
                continue # Skip if GAX file isn't found
                
            try:
                # 1. Load and Threshold the Masked Image
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, binary_mask = cv2.threshold(mask_img, 1, 1, cv2.THRESH_BINARY)
                binary_mask_resized = cv2.resize(binary_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                
                # 2. Load GAX Heatmap
                gax_data = np.load(gax_path)
                final_gax_mask = gax_data[-1] 
                positive_heatmap = get_positive_heatmap(final_gax_mask)
                
                # 3. Compute the Three Metrics
                base_score = compute_score(positive_heatmap, binary_mask_resized)
                thresh_score = compute_thresholded_score(positive_heatmap, binary_mask_resized, threshold_ratio=0.20)
                pointing_failed = compute_pointing_game(positive_heatmap, binary_mask_resized)
                
                # 4. Flag severe shortcut learning
                is_cheating = base_score > 0.50
                is_severe_shortcut = (thresh_score > 0.50) or pointing_failed
                
                results.append({
                    'image_name': img_name,
                    'true_class': class_name,
                    'base_cheating_score': round(base_score, 4),
                    'thresholded_cheating_score': round(thresh_score, 4),
                    'pointing_game_failed': pointing_failed,
                    'is_shortcut_learning': is_cheating,
                    'is_severe_shortcut': is_severe_shortcut
                })
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                
    # --- Save and Summarize ---
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nFinished! Results saved to {OUTPUT_CSV}")
        
        total_images = len(df)
        cheating_count = df['is_shortcut_learning'].sum()
        severe_count = df['is_severe_shortcut'].sum()
        pointing_count = df['pointing_game_failed'].sum()
        
        average_cheating_score = df['base_cheating_score'].mean()
        average_thresh_score = df['thresholded_cheating_score'].mean()
        
        print(f"\n======================================")
        print(f"         FINAL MULTI-METRIC SUMMARY   ")
        print(f"======================================")
        print(f"Total Images Scored: {total_images}")
        
        class_counts = df['true_class'].value_counts()
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count} images")
            
        print(f"\n1. BASE METRICS (>50% total attribution outside lungs)")
        print(f"   Shortcut Detected: {cheating_count} images ({(cheating_count/total_images)*100:.1f}%)")
        print(f"   Average Base Score: {average_cheating_score:.4f} ({average_cheating_score*100:.1f}%)")
        
        cheating_class_counts = df[df['is_shortcut_learning']]['true_class'].value_counts()
        for cls in class_counts.keys():
            c_count = cheating_class_counts.get(cls, 0)
            print(f"     - {cls}: {c_count} fails")

        print(f"\n2. THRESHOLDED & POINTING GAME (The 'Severe' Tests)")
        print(f"   Severe Shortcut Detected: {severe_count} images ({(severe_count/total_images)*100:.1f}%)")
        print(f"   Average Thresholded Score: {average_thresh_score:.4f} ({average_thresh_score*100:.1f}%)")
        print(f"   Pointing Game Fails (Max pixel outside): {pointing_count} images ({(pointing_count/total_images)*100:.1f}%)")
        
        severe_class_counts = df[df['is_severe_shortcut']]['true_class'].value_counts()
        for cls in class_counts.keys():
            s_count = severe_class_counts.get(cls, 0)
            print(f"     - {cls}: {s_count} severe fails")
            
    else:
        print("\nNo matching GAX and Mask files found. Ensure your batch scripts completed successfully.")

if __name__ == "__main__":
    main()