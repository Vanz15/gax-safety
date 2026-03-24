import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# --- HOTFIX FOR TRANSFORMERS BUG ---
# ==========================================
from transformers.modeling_utils import PreTrainedModel

# Intercept the Hugging Face function that is causing the crash
original_mark = PreTrainedModel.mark_tied_weights_as_initialized

def patched_mark(self, *args, **kwargs):
    # If the model author forgot to add this variable, we inject an empty dictionary for them
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
    return original_mark(self, *args, **kwargs)

# Apply the patch globally
PreTrainedModel.mark_tied_weights_as_initialized = patched_mark
# ==========================================

from transformers import AutoModel

def apply_dl_lung_mask(img_np, mask_tensor):
    """
    Applies the deep learning mask to the original image.
    1 = Right Lung, 2 = Left Lung, 3 = Heart.
    We want to keep 1 and 2, and black out everything else.
    """
    # 1. Convert the tensor output to a single 2D mask array
    mask_2d = mask_tensor.argmax(1).squeeze().cpu().numpy()
    
    # 2. Create a binary mask where ONLY the lungs (1 or 2) are True
    lungs_binary = np.zeros_like(mask_2d, dtype=np.uint8)
    lungs_binary[mask_2d == 1] = 1  # Right lung
    lungs_binary[mask_2d == 2] = 1  # Left lung
    
    # 3. Resize the mask back to the original image dimensions
    original_height, original_width = img_np.shape[:2]
    lungs_resized = cv2.resize(lungs_binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # 4. Expand to 3D so we can multiply it with the RGB image
    mask_3d = np.expand_dims(lungs_resized, axis=2)
    
    # 5. Apply the mask (Lungs keep their color, everything else becomes black)
    masked_img_np = img_np * mask_3d
    
    return Image.fromarray(masked_img_np)

def main():
    INPUT_BASE_DIR = "jpeg_dataset/test"
    OUTPUT_BASE_DIR = "masked_dataset/test"
    
    # --- Load the Deep Learning Segmentation Model ---
    print("Loading Hugging Face CXR Segmentation Model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We use trust_remote_code=True because this model has custom architecture code
    model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
    model = model.eval().to(device)
    
    if not os.path.exists(INPUT_BASE_DIR):
        print(f"Error: Could not find input directory {INPUT_BASE_DIR}")
        return
        
    class_folders = [f for f in os.listdir(INPUT_BASE_DIR) if os.path.isdir(os.path.join(INPUT_BASE_DIR, f))]
    
    for class_name in class_folders:
        input_dir = os.path.join(INPUT_BASE_DIR, class_name)
        output_dir = os.path.join(OUTPUT_BASE_DIR, class_name)
        
        os.makedirs(output_dir, exist_ok=True)
        img_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        print(f"\nProcessing {len(img_files)} images in '{class_name}' folder...")
        
        for img_name in tqdm(img_files, desc=f"Segmenting {class_name}"):
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, img_name)
            
            # Skip if already processed
            if os.path.exists(output_path):
                continue
                
            try:
                # 1. Load the image using OpenCV (0 = Grayscale)
                img_gray = cv2.imread(input_path, 0)
                if img_gray is None: continue
                
                # 2. Preprocess the image
                x = model.preprocess(img_gray)
                
                # --- PROPER TENSOR FORMATTING ---
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                
                # Force tensor to exactly 4D: [Batch, Channel, Height, Width]
                if x.ndim == 2:
                    x = x.unsqueeze(0).unsqueeze(0)  # Turns [H, W] into [1, 1, H, W]
                elif x.ndim == 3:
                    x = x.unsqueeze(0)               # Turns [1, H, W] into [1, 1, H, W]
                    
                # Ensure it strictly has only 1 channel for the Conv2D layer
                if x.shape[1] != 1:
                    x = x[:, 0:1, :, :]
                    
                x = x.to(device)
                # -----------------------------------------
                
                # 3. Run the Deep Learning model to get the mask
                with torch.no_grad():
                    out = model(x)
                
                # 4. Load original color image for the final save
                pil_img = Image.open(input_path).convert("RGB")
                img_color_np = np.array(pil_img)
                
                # 5. Apply the mask to isolate the lungs
                masked_img = apply_dl_lung_mask(img_color_np, out["mask"])
                
                # 6. Save the output
                masked_img.save(output_path)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"\nSegmentation complete! All masked images saved to {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()