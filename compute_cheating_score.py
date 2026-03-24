import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
import argparse

def get_positive_heatmap(raw_mask):
    """Flattens the 3-channel GAX mask and extracts only positive (red) attributions."""
    if raw_mask.shape[-1] in [1, 3]:
        # Sum the channels to match the 2D visualization methodology
        h_sum = np.sum(raw_mask, axis=2)
    else:
        h_sum = raw_mask
        
    # We only penalize the model for pixels that INCREASED its confidence
    positive_attr = np.maximum(h_sum, 0)
    return positive_attr

def compute_score(positive_heatmap, binary_mask):
    """Calculates the percentage of positive attribution outside the lung mask."""
    # Invert the mask: Lungs = 0, Background = 1
    inverse_mask = 1.0 - binary_mask
    
    total_attribution = np.sum(positive_heatmap)
    if total_attribution == 0:
        return 0.0 # Safety check for empty heatmaps
        
    # Multiply the heatmap by the inverse mask to isolate outside attributions
    outside_attribution = np.sum(positive_heatmap * inverse_mask)
    
    # Return the ratio
    return float(outside_attribution / total_attribution)

def main():
    parser = argparse.ArgumentParser(description="Compute cheating scores for a given model's GAX output")
    parser.add_argument("--gax_dir", type=str, default="results/resnet34_v3/gax_images", help="Directory containing the GAX heatmaps")
    parser.add_argument("--output_csv", type=str, default="results/resnet34_v3/cheating_scores.csv", help="Path to save the results CSV")
    args = parser.parse_args()

    # --- Configurations ---
    MASK_DIR = "masked_dataset/test"
    GAX_DIR = args.gax_dir
    OUTPUT_CSV = args.output_csv
    
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    results = []
    
    # Find the class folders (PNEUMONIA, NORMAL)
    if not os.path.exists(MASK_DIR):
        print(f"Error: Could not find {MASK_DIR}")
        return
        
    class_folders = [f for f in os.listdir(MASK_DIR) if os.path.isdir(os.path.join(MASK_DIR, f))]
    
    print("Starting Cheating Score calculation...")
    
    for class_name in class_folders:
        class_path = os.path.join(MASK_DIR, class_name)
        img_files = [f for f in os.listdir(class_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        for img_name in tqdm(img_files, desc=f"Scoring {class_name}"):
            mask_path = os.path.join(class_path, img_name)
            
            # Reconstruct the exact GAX filename your previous script generated
            gax_filename = f"op.{img_name}.test.mult.npy"
            gax_path = os.path.join(GAX_DIR, gax_filename)
            
            # Skip if the GAX file wasn't generated for this specific image
            if not os.path.exists(gax_path):
                continue
                
            try:
                # 1. Load the Masked Image
                # Since the background was blacked out, we threshold it to create a pure 1 or 0 matrix
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                _, binary_mask = cv2.threshold(mask_img, 1, 1, cv2.THRESH_BINARY)
                
                # 2. Load the GAX Heatmap
                gax_data = np.load(gax_path)
                final_gax_mask = gax_data[-1] # Grab the final 100th iteration
                
                # 3. Resize Binary Mask to match GAX dimensions (224x224)
                # We use INTER_NEAREST to ensure the edges stay strictly 0 or 1 without blurring
                binary_mask_resized = cv2.resize(binary_mask, (224, 224), interpolation=cv2.INTER_NEAREST)
                
                # 4. Extract Positive Attributions
                positive_heatmap = get_positive_heatmap(final_gax_mask)
                
                # 5. Compute the Math
                cheating_score = compute_score(positive_heatmap, binary_mask_resized)
                
                # Flag it based on the 25% threshold
                is_cheating = cheating_score > 0.50
                
                results.append({
                    'image_name': img_name,
                    'true_class': class_name,
                    'cheating_score': round(cheating_score, 4),
                    'is_shortcut_learning': is_cheating
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
        cheating_percentage = (cheating_count / total_images) * 100 if total_images > 0 else 0
        average_cheating_score = df['cheating_score'].mean()
        
        print(f"\n--- Final Summary ---")
        print(f"Total Images Scored: {total_images}")
        
        class_counts = df['true_class'].value_counts()
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count} images")
            
        print(f"Shortcut Learning Detected: {cheating_count} images ({cheating_percentage:.1f}%)")
        
        cheating_class_counts = df[df['is_shortcut_learning']]['true_class'].value_counts()
        for cls in class_counts.keys():
            c_count = cheating_class_counts.get(cls, 0)
            print(f"  - {cls}: {c_count} images")
            
        print(f"Average Model Cheating Score: {average_cheating_score:.4f} ({average_cheating_score * 100:.1f}%)")
    else:
        print("\nNo matching GAX and Mask files found. Ensure your batch scripts completed successfully.")

if __name__ == "__main__":
    main()