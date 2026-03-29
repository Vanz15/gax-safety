import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel


original_mark = PreTrainedModel.mark_tied_weights_as_initialized

def patched_mark(self, *args, **kwargs):
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
    return original_mark(self, *args, **kwargs)

PreTrainedModel.mark_tied_weights_as_initialized = patched_mark


# image mask
def apply_dl_lung_mask(img_np, mask_tensor):
    """
    Takes the model's prediction and uses it like a stencil to cut out 
    only the lungs, turning the background and other organs black.
    Model output codes: 1 = Right Lung, 2 = Left Lung, 3 = Heart.
    """
    # 1. Get the most confident prediction for every single pixel
    mask_2d = mask_tensor.argmax(dim=1).squeeze().cpu().numpy()
    
    # 2. Create a blank (all zeros) stencil
    lungs_binary = np.zeros_like(mask_2d, dtype=np.uint8)
    
    # 3. Mark the stencil with a '1' wherever the model saw either lung
    lungs_binary[(mask_2d == 1) | (mask_2d == 2)] = 1 
    
    # 4. Resize the stencil to match the exact dimensions of the original X-ray
    original_height, original_width = img_np.shape[:2]
    lungs_resized = cv2.resize(lungs_binary, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
    
    # 5. Add a 3rd dimension to the stencil so it can multiply with a 3-channel (RGB) image
    mask_3d = np.expand_dims(lungs_resized, axis=2)
    
    # 6. Multiply the original image by the stencil (Lung pixels * 1 stay the same, Background * 0 turns black)
    masked_img_np = img_np * mask_3d
    
    return Image.fromarray(masked_img_np)


# main function
def main():
    # Directories
    INPUT_BASE_DIR = "jpeg_dataset/test"
    OUTPUT_BASE_DIR = "masked_dataset/test"
    
    # Setup the computation device (GPU if available, otherwise CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Hugging Face CXR Segmentation Model on {device.upper()}...")
    
    # Load the pre-trained chest X-ray segmentation model
    # trust_remote_code=True is required because the author used custom architecture
    model = AutoModel.from_pretrained("ianpan/chest-x-ray-basic", trust_remote_code=True)
    model = model.eval().to(device)
    
    if not os.path.exists(INPUT_BASE_DIR):
        print(f"Error: Could not find input directory '{INPUT_BASE_DIR}'")
        return
        
    # Find all the class folders (e.g., "Normal", "Pneumonia")
    class_folders = [f for f in os.listdir(INPUT_BASE_DIR) if os.path.isdir(os.path.join(INPUT_BASE_DIR, f))]
    
    for class_name in class_folders:
        input_dir = os.path.join(INPUT_BASE_DIR, class_name)
        output_dir = os.path.join(OUTPUT_BASE_DIR, class_name)
        
        os.makedirs(output_dir, exist_ok=True) # Create output folder if it doesn't exist
        img_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        print(f"\nProcessing {len(img_files)} images in '{class_name}' folder...")
        
        # Process each image with a visual progress bar
        for img_name in tqdm(img_files, desc=f"Segmenting {class_name}"):
            input_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, img_name)
            
            # Skip this image if we already processed it previously (allows pausing/resuming)
            if os.path.exists(output_path):
                continue
                
            try:
                # -----------------------------------------
                # STEP A: Data Loading & Preprocessing
                # -----------------------------------------
                # Read image in grayscale (0)
                img_gray = cv2.imread(input_path, 0)
                if img_gray is None: 
                    continue
                
                # Use the model's built-in tool to normalize the image
                x = model.preprocess(img_gray)
                
                # Convert to a PyTorch tensor if it isn't one already
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                
                # -----------------------------------------
                # STEP B: Tensor Reshaping
                # -----------------------------------------
                # The model requires a 4D tensor: [Batch Size, Channels, Height, Width]
                # If the image is 2D [Height, Width], add empty Batch and Channel dimensions
                if x.ndim == 2:
                    x = x.unsqueeze(0).unsqueeze(0)  
                # If the image is 3D [Channel, Height, Width], add an empty Batch dimension
                elif x.ndim == 3:
                    x = x.unsqueeze(0)               
                    
                # Force the image to strictly have 1 channel (Grayscale)
                if x.shape[1] != 1:
                    x = x[:, 0:1, :, :]
                    
                x = x.to(device)
                
                # -----------------------------------------
                # STEP C: Inference & Mask Application
                # -----------------------------------------
                # Run the image through the network to generate the lung predictions
                with torch.no_grad():
                    out = model(x)
                
                # Load the original image in full color so the final output looks natural
                pil_img = Image.open(input_path).convert("RGB")
                img_color_np = np.array(pil_img)
                
                # Apply the stencil mask to the color image
                masked_img = apply_dl_lung_mask(img_color_np, out["mask"])
                
                # Save the successfully isolated lungs to the new folder
                masked_img.save(output_path)
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    print(f"\nSegmentation complete! All masked images saved to {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()