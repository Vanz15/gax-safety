import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import os
from rsna_dataset import JpegRSNADataset, val_transforms, CLASSES, IMAGENET_MEAN, IMAGENET_STD
from deconv import DeconvNet

def load_trained_model(model_path, device):
    print(f"Loading model from {model_path}...")
    model = models.resnet34(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # FIX: Robust cleaning of state dict
    new_state_dict = {}
    for k, v in state_dict.items():
        # Skip non-tensor data
        if not isinstance(v, torch.Tensor):
            continue
        
        name = k.replace("backbone.", "")
        
        # Check for shape mismatch
        if name in model.state_dict():
            if v.shape != model.state_dict()[name].shape:
                print(f"Warning: Skipping {name} due to shape mismatch.")
                continue
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

def denormalize(tensor):
    """Convert normalized tensor back to image for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).detach().numpy()

def process_deconv_map(grads):
    """
    Process the raw gradients for visualization.
    grads: [C, H, W] numpy array
    """
    # 1. Transpose to [H, W, C]
    grads = np.transpose(grads, (1, 2, 0))
    
    # 2. Take absolute value (we care about magnitude of influence)
    grads = np.abs(grads)
    
    # 3. Max across channels to get a 2D heatmap (common for grayscale X-rays)
    # Alternatively, you can keep it RGB if you prefer.
    heatmap = np.max(grads, axis=2)
    
    # 4. Normalize to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-9)
    
    return heatmap

def main():
    root_dir = "jpeg_dataset"
    model_path = os.path.join("checkpoints", "best_resnet34_5.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset & Find a Pneumonia Case
    val_dataset = JpegRSNADataset(root_dir=root_dir, split="val", transform=val_transforms)
    test_dataset = JpegRSNADataset(root_dir=root_dir, split="test", transform=val_transforms)
    
    target_idx = 0
    found = False
    for i in range(len(test_dataset)):
        _, label = test_dataset[i]
        if label == 1: # Pneumonia
            target_idx = i
            found = True
            break
    
    if not found:
        print("No pneumonia samples found, using index 0.")
        target_idx = 0
            
    img_tensor, label = test_dataset[target_idx]
    img_tensor = img_tensor.unsqueeze(0).to(device) # [1, C, H, W]
    
    print(f"Explaining Image Index: {target_idx}, True Label: {CLASSES[label]}")

    # 2. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    model = load_trained_model(model_path, device)

    # 3. Run DeconvNet
    # Get model prediction first
    with torch.no_grad():
        logits = model(img_tensor)
        pred_idx = logits.argmax(dim=1).item()
        print(f"Model Prediction: {CLASSES[pred_idx]}")

    deconv = DeconvNet(model)
    raw_map = deconv.generate(img_tensor, target_class=pred_idx)
    
    # Clean up hooks
    deconv.remove_hooks()

    # 4. Visualize
    original_img = denormalize(img_tensor.squeeze())
    heatmap = process_deconv_map(raw_map)
    
    # --- NEW: Generate Sum and Mult Augmentations for Visualization ---
    # 1. Sum Method (Alpha = 0.25)
    alpha = 0.25
    attr_tensor = torch.from_numpy(raw_map).unsqueeze(0).to(device)
    if attr_tensor.abs().max() > 0:
        attr_tensor = attr_tensor / attr_tensor.abs().max()
    
    sum_tensor = img_tensor + (attr_tensor * alpha)
    sum_img = denormalize(sum_tensor.squeeze())
    
    # 2. Mult Method (Masking)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)
    
    x_denorm = img_tensor * std + mean
    mask = torch.abs(attr_tensor)
    if mask.max() > 0:
        mask = mask / mask.max()
        
    mult_tensor = x_denorm * mask
    # mult_tensor is already denormalized [0,1], convert to numpy directly
    mult_img = mult_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    mult_img = np.clip(mult_img, 0, 1)
    # ------------------------------------------------------------------
    
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 5, 1)
    plt.title("Original Image")
    plt.imshow(original_img)
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title(f"DeconvNet Map\n(Target: {CLASSES[pred_idx]})")
    plt.imshow(heatmap, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title("Hot Overlay")
    plt.imshow(original_img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.title(f"Sum Augmentation\n(Alpha={alpha})")
    plt.imshow(sum_img)
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.title("Mult Augmentation\n(Masked)")
    plt.imshow(mult_img)
    plt.axis('off')
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    save_path = os.path.join("results", f"deconv_heatmap_{target_idx}.png")
    plt.savefig(save_path)
    print(f"Saved heatmap to {save_path}")
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()