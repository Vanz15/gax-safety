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
from torch.utils.data import Dataset, DataLoader

# --- PyTorch Speed Optimizations ---
torch.backends.cudnn.benchmark = True

# ==========================================
# 1. Models
# ==========================================
class BatchGenerator(nn.Module):
    """
    Refactored to handle a dynamic batch size. 
    Each image in the batch gets its own independent W and b parameters.
    """
    def __init__(self, batch_size, img_size=(224, 224)):
        super(BatchGenerator, self).__init__()
        self.W = nn.Parameter(torch.zeros(size=(batch_size, 3) + img_size) + 1)
        self.b = nn.Parameter(torch.zeros(size=(batch_size, 3) + img_size) + 0.01)
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
# 2. Dataset Setup
# ==========================================
class XRayDataset(Dataset):
    def __init__(self, img_files, data_dir, img_size=(224, 224)):
        self.img_files = img_files
        self.data_dir = data_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img_pil = Image.open(img_path).convert('RGB').resize(self.img_size)
        x_np = np.array(img_pil).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(x_np).float(), img_name

# ==========================================
# 3. Batch Optimization Engine
# ==========================================
def run_batch_gax(model_path, output_dir, batch_size=16):
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    BASE_DATA_DIR = "jpeg_dataset/test" 
    OUTPUT_DIR = output_dir
    N_ITER = 150
    LR = 0.1
    IMG_SIZE = (224, 224)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading ResNet34 on {device}...")
    model = load_trained_resnet(model_path, device)
    
    class_folders = [f for f in os.listdir(BASE_DATA_DIR) if os.path.isdir(os.path.join(BASE_DATA_DIR, f))]
        
    for class_name in class_folders:
        data_dir = os.path.join(BASE_DATA_DIR, class_name)
        target_label = 1 if class_name.lower() == 'pneumonia' else 0
        
        all_img_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        # Filter out images that have already been processed
        img_files = []
        for f in all_img_files:
            base_name = f"op.{f}.test.mult"
            if not (os.path.exists(os.path.join(OUTPUT_DIR, f"{base_name}.npy")) and 
                    os.path.exists(os.path.join(OUTPUT_DIR, f"{base_name}.COS.npy"))):
                img_files.append(f)
                
        print(f"\nProcessing Class: {class_name} | Target Label: {target_label}")
        print(f"Found {len(all_img_files)} images total. {len(img_files)} remaining to process.")
        
        if len(img_files) == 0:
            continue

        # Initialize DataLoader for fast background image loading
        dataset = XRayDataset(img_files, data_dir, IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        for batch_x, batch_names in tqdm(dataloader, desc=f"Generating Batched GAX ({class_name})"):
            actual_batch_size = batch_x.size(0)
            batch_x = batch_x.to(device, non_blocking=True)
            
            # Base prediction for the whole batch
            with torch.no_grad():
                base_logits = model(batch_x)
                base_probs = F.softmax(base_logits, dim=1)
                n_class = base_probs.shape[1]
                
            # Initialize batch-aware Generator
            netG = BatchGenerator(batch_size=actual_batch_size, img_size=IMG_SIZE).to(device)
            optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(0.9, 0.999))
            
            # Pre-allocate score constants for the whole batch
            score_constants = torch.zeros((actual_batch_size, n_class), device=device) - (1.0 / (n_class - 1))
            score_constants[:, target_label] = 1.0
            
            # Tracking histories
            batch_imgs_history = [[] for _ in range(actual_batch_size)]
            batch_scores_history = [[] for _ in range(actual_batch_size)]
            
            epsilon = 1e-4
            similarity_loss_factor = 1.0
            
            # Optimization Loop for the batch
            for i in range(N_ITER):
                netG.train()
                optimizerG.zero_grad()
                
                attr_op = netG(batch_x)
                x_aug = batch_x * attr_op 
                
                aug_logits = model(x_aug)
                aug_probs = F.softmax(aug_logits, dim=1)
                
                # Calculate scores per image in the batch
                co_score_tensor = (aug_probs - base_probs) * score_constants
                co_scores = torch.sum(co_score_tensor, dim=1) # Shape: [batch_size]
                
                # Calculate similarity loss per image
                sim_losses = similarity_loss_factor / torch.mean((attr_op - batch_x + epsilon)**2 / (batch_x + epsilon), dim=[1,2,3])
                
                # The total loss is the mean of the independent batch losses
                losses = -co_scores + sim_losses
                loss = losses.sum()
                
                loss.backward()
                optimizerG.step()
                
                # Save history efficiently (detached from graph, moved to CPU to save VRAM)
                netG.eval()
                with torch.no_grad():
                    current_masks = netG(batch_x).detach().cpu()
                    current_scores = co_scores.detach().cpu()
                    
                    for b in range(actual_batch_size):
                        batch_scores_history[b].append(current_scores[b].item())
                        # Store as numpy arrays only at the very end to prevent CPU bottlenecks
                        batch_imgs_history[b].append(current_masks[b].numpy().transpose(1, 2, 0))
                        
            # Save batch results to disk
            for b in range(actual_batch_size):
                img_name = batch_names[b]
                base_name = f"op.{img_name}.test.mult"
                mask_save_path = os.path.join(OUTPUT_DIR, f"{base_name}.npy")
                score_save_path = os.path.join(OUTPUT_DIR, f"{base_name}.COS.npy")
                
                np.save(mask_save_path, np.array(batch_imgs_history[b]))
                np.save(score_save_path, np.array(batch_scores_history[b]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GAX masks for a given model")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_resnet34_v3.pth", help="Path to the model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/resnet34_v3/gax_images", help="Directory to save the GAX results")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to process simultaneously")
    args = parser.parse_args()
    
    run_batch_gax(args.model_path, args.output_dir, args.batch_size)