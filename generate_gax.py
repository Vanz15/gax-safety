import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from PIL import Image
import random
from tqdm import tqdm
import argparse

# ==========================================
# 1. Models
# ==========================================
class Generator(nn.Module):
    def __init__(self, img_size=(224, 224)):
        super(Generator, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(3,) + img_size) + 1)
        self.b = nn.Parameter(torch.zeros(size=(3,) + img_size) + 0.01)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.W * x + self.b)

def load_trained_resnet(model_path, device):
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('net', checkpoint.get('model', checkpoint.get('state_dict', checkpoint)))
    new_state_dict = {}
    for k, v in state_dict.items():
        if k == 'iter' or not isinstance(v, torch.Tensor): continue
        name = k.replace("backbone.", "")
        if name in model.state_dict() and v.shape == model.state_dict()[name].shape:
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model

# ==========================================
# 2. Batch Optimization Engine
# ==========================================
def run_batch_gax(model_path, output_dir):
    # For reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

   
    # --- Configurations ---
    MODEL_PATH = model_path
    BASE_DATA_DIR = "jpeg_dataset/test" 
    OUTPUT_DIR = output_dir
    
    N_ITER = 150
    LR = 0.1
    IMG_SIZE = (224, 224)
    MAX_IMAGES_PER_CLASS = 100
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Classifier
    print(f"Loading ResNet34 on {device}...")
    model = load_trained_resnet(MODEL_PATH, device)
    
    class_folders = [f for f in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, f))]
        
    for class_name in class_folders:
        data_dir = os.path.join(BASE_DATA_DIR, class_name)
        
        # 0 = Normal, 1 = Pneumonia
        target_label = 1 if class_name.lower() == 'pneumonia' else 0
        
        # Get list of images
        all_img_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        # Apply the limit here
        img_files = all_img_files[:MAX_IMAGES_PER_CLASS]
        print(f"\nProcessing Class: {class_name} | Target Label: {target_label}")
        print(f"Found {len(all_img_files)} images total. Limited to processing: {len(img_files)} images.")
        
        # Process each image with a progress bar
        for img_name in tqdm(img_files, desc=f"Generating GAX ({class_name})"):
            # Define output filenames to match Tjoa's format
            # e.g., op.person1_virus_6.jpeg.test.mult.npy
            base_name = f"op.{img_name}.test.mult"
            mask_save_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
            score_save_path = os.path.join(OUTPUT_DIR, f"{base_name}.COS.npy")
            
            # Skip if already processed (allows you to pause/resume the script)
            if os.path.exists(mask_save_path) and os.path.exists(score_save_path):
                continue
                
            img_path = os.path.join(data_dir, img_name)
            img_pil = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
            x_np = np.array(img_pil).transpose(2, 0, 1) / 255.0
            x = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            
            # Base prediction
            with torch.no_grad():
                base_logits = model(x)
                base_probs = F.softmax(base_logits, dim=1)
                pred_idx = torch.argmax(base_probs, dim=1).item()
                n_class = base_probs.shape[1]
                
            # Initialize fresh Generator for this specific image
            netG = Generator(img_size=IMG_SIZE).to(device)
            optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.9, 0.999))
            
            imgs_history = []
            co_scores_history = []
            epsilon = 1e-4
            similarity_loss_factor = 1.0
            
            # Optimization Loop
            for i in range(N_ITER):
                netG.train()
                optimizerG.zero_grad()
                
                attr_op = netG(x)
                x_aug = x * attr_op 
                
                aug_logits = model(x_aug)
                aug_probs = F.softmax(aug_logits, dim=1)
                
                score_constants = torch.zeros_like(aug_probs) - (1.0 / (n_class - 1))
                score_constants[0, target_label] = 1.0
                
                co_score_tensor = (aug_probs - base_probs) * score_constants
                co_score = torch.sum(co_score_tensor)
                
                sim_loss = similarity_loss_factor / torch.mean((attr_op - x + epsilon)**2 / (x + epsilon))
                loss = -co_score + sim_loss
                
                loss.backward()
                optimizerG.step()
                
                # Save history
                co_scores_history.append(co_score.item())
                netG.eval()
                with torch.no_grad():
                    # Tjoa format: [H, W, C]
                    current_mask = netG(x).cpu().numpy()[0].transpose(1, 2, 0)
                    imgs_history.append(current_mask)
                    
            # Save artifacts to disk
            np.save(mask_save_path, np.array(imgs_history))
            np.save(score_save_path, np.array(co_scores_history))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GAX masks for a given model")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_resnet34_v3.pth", help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/resnet34_v3/gax_images", help="Directory to save the GAX results")
    args = parser.parse_args()
    
    run_batch_gax(args.model_path, args.output_dir)