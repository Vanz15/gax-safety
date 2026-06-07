import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import argparse

def get_positive_heatmap(raw_mask):
    """Extracts positive attributions and normalizes them for visual overlay."""
    if raw_mask.shape[-1] in [1, 3]:
        h_sum = np.sum(raw_mask, axis=2)
    else:
        h_sum = raw_mask
        
    positive_attr = np.maximum(h_sum, 0)
    
    # Normalize between 0 and 1 so the colormap scales correctly
    if np.max(positive_attr) > 0:
        positive_attr = positive_attr / np.max(positive_attr)
        
    return positive_attr

def generate_visuals(images_df, prefix_name, orig_dir, mask_dir, gax_dir, output_dir):
    """Generates and saves the 3-panel visual proofs for a given set of images."""
    for index, row in images_df.iterrows():
        img_name = row['image_name']
        true_class = row['true_class']
        score = row['cheating_score']
        
        # Paths
        orig_path = os.path.join(orig_dir, true_class, img_name)
        mask_path = os.path.join(mask_dir, true_class, img_name)
        
        # Check for our 'mult' naming convention
        gax_path = os.path.join(gax_dir, f"op.{img_name}.test.mult.npy")
        if not os.path.exists(gax_path):
            gax_path = os.path.join(gax_dir, f"op.{img_name}.test.sum.npy")
            
        if not all([os.path.exists(orig_path), os.path.exists(mask_path), os.path.exists(gax_path)]):
            print(f"Skipping {img_name} due to missing files.")
            continue
            
        # 2. Load and format images for Matplotlib (RGB format)
        orig_img = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
        orig_img_resized = cv2.resize(orig_img, (224, 224))
        
        mask_img = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
        mask_img_resized = cv2.resize(mask_img, (224, 224))
        
        # 3. Load and normalize the GAX Heatmap
        gax_data = np.load(gax_path)
        heatmap_2d = get_positive_heatmap(gax_data[-1])
        
        # 4. Create the 3-Panel Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Add a professional title with the exact score
        fig.suptitle(f"Shortcut Learning Proof ({prefix_name}) | Cheating Score: {score:.1%} | Class: {true_class}", 
                     fontsize=16, fontweight='bold', y=1.05)
        
        # Panel 1: Original X-Ray
        axes[0].imshow(orig_img_resized)
        axes[0].set_title("Original X-Ray")
        axes[0].axis('off')
        
        # Panel 2: Deep Learning Lung Mask
        axes[1].imshow(mask_img_resized)
        axes[1].set_title("Segmented Lungs (Ground Truth)")
        axes[1].axis('off')
        
        # Panel 3: GAX Heatmap Overlay
        axes[2].imshow(orig_img_resized) # Base image
        
        # We mask the heatmap so that low-value/empty areas become fully transparent.
        # This prevents the whole X-ray from being covered in a dark blue wash.
        heatmap_overlay = np.ma.masked_where(heatmap_2d < 0.15, heatmap_2d) 
        
        # Overlay the hot spots using the 'jet' colormap (Blue = Low, Red = High)
        axes[2].imshow(heatmap_overlay, cmap='jet', alpha=0.55)
        axes[2].set_title("GAX Heatmap Overlay")
        axes[2].axis('off')
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"proof_{prefix_name}_{img_name}")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate visual proofs for shortcut learning")
    parser.add_argument("--csv_path", type=str, default="results/resnet34_v3/cheating_scores.csv", help="Path to the cheating scores CSV")
    parser.add_argument("--gax_dir", type=str, default="results/resnet34_v3/gax_images", help="Directory containing the GAX heatmaps")
    parser.add_argument("--output_dir", type=str, default="results/resnet34_v3/visualizations", help="Directory to save the visualizations")
    args = parser.parse_args()

    # --- Configurations ---
    CSV_PATH = args.csv_path
    ORIGINAL_DIR = "jpeg_dataset/test"
    MASK_DIR = "masked_dataset/test"
    GAX_DIR = args.gax_dir
    OUTPUT_DIR = args.output_dir
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load the scores and sort to find the worst cases of shortcut learning
    if not os.path.exists(CSV_PATH):
        print(f"Error: Could not find {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    # Sort descending so the highest cheating scores are at the top
    df_sorted = df.sort_values(by='cheating_score', ascending=False)
    
    # Grab the top 10 worst offenders and bottom 5 best cases
    top_images = df_sorted.head(10)
    bottom_images = df_sorted.tail(5)
    
    print("Generating visual proofs for the top 10 worst shortcut learning examples...")
    generate_visuals(top_images, "worst", ORIGINAL_DIR, MASK_DIR, GAX_DIR, OUTPUT_DIR)
    
    print("\nGenerating visual proofs for the 5 lowest shortcut learning examples...")
    generate_visuals(bottom_images, "best", ORIGINAL_DIR, MASK_DIR, GAX_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()