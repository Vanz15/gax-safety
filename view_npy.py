import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Put the exact path to ONE of your .npy files here
FILE_PATH = "results/resnet34_v3/gax_images/op.1d1badbd-7789-4923-a387-7ee4e3b18cc7.jpg.test.mult.npy"

def main():
    try:
        print(f"Loading: {FILE_PATH}")
        
        # 1. Load the raw matrix data
        gax_data = np.load(FILE_PATH)
        
        # Print the shape so you know exactly what is inside
        print(f"Full Array Shape: {gax_data.shape} -> (Iterations, Height, Width, Channels)")
        
        # 2. Grab the final iteration (the 100th step we generated)
        final_heatmap = gax_data[-1]
        
        # 3. Sum the color channels to make a 2D map, keeping only the positive attributions
        if final_heatmap.ndim == 3:
            h_sum = np.sum(final_heatmap, axis=2)
        else:
            h_sum = final_heatmap
            
        positive_attr = np.maximum(h_sum, 0)
        
        # 4. Display it!
        plt.figure(figsize=(6, 6))
        
        # We use the 'jet' colormap so high values are red and low values are dark blue
        plt.imshow(positive_attr, cmap='jet')
        plt.colorbar(label="Attribution Strength")
        plt.title("Raw GAX Heatmap")
        plt.axis('off')
        
        # This will pop open a new window on your screen
        plt.show()
        
    except FileNotFoundError:
        print("Error: Could not find the file. Double-check the FILE_PATH string!")

if __name__ == "__main__":
    main()