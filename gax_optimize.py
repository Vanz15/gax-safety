import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
from PIL import Image

# 1. --- E. Tjoa's GAX Generator ---
class Generator(nn.Module):
    def __init__(self, img_size):
        super(Generator, self).__init__()
        # Learns a direct affine transformation for every single pixel
        self.W = nn.Parameter(torch.zeros(size=(3,) + img_size) + 1)
        self.b = nn.Parameter(torch.zeros(size=(3,) + img_size) + 0.01)
        self.act = nn.Tanh() # Bounds perturbations between -1 and +1

    def forward(self, x):
        x = self.W * x + self.b
        x = self.act(x)
        return x

# 2. --- Robust Model Loader ---
def load_trained_resnet(model_path, device):
    print(f"Loading ResNet34 from {model_path}...")
    model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found at: {model_path}")
        
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

# 3. --- Image Processing ---
def load_and_preprocess_image(img_path, device, size=(224, 224)):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(size)
    x = np.array(img).transpose(2, 0, 1) # HWC to CHW
    x = x / 255.0 # Normalize to [0, 1]
    x_tensor = torch.from_numpy(x).float().unsqueeze(0).to(device)
    return x_tensor

# 4. --- Interactive Plotter ---
# 4. --- Interactive Plotter with Jet Heatmap Methodology ---
class LocalInteractivePlot(object):
    def __init__(self, data, co_scores, img_tensor):
        self.data = data # Shape: [Steps, H, W, C]
        self.co_scores = co_scores
        # The original image tensor converted to a numpy array for plotting
        self.img = img_tensor.cpu().numpy()[0].transpose(1, 2, 0)
        self.steps = len(data)
        
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # --- LEFT PANEL: Original X-ray ---
        self.ax[0].imshow(self.img)
        self.ax[0].set_title("Original X-ray")
        self.ax[0].axis('off')
        
        # --- RIGHT PANEL: Overlay Methodology ---
        # 1. Draw the base X-ray
        self.ax[1].imshow(self.img)
        
        # 2. Process Step 0 heatmap using the notebook methodology
        initial_heatmap = self.process_heatmap(self.data[0])
        
        # 3. Draw the mask ON TOP with jet colormap and 50% transparency
        self.mask_plot = self.ax[1].imshow(
            initial_heatmap, 
            cmap='jet', 
            alpha=0.5
        )
        
        self.title_text = self.ax[1].set_title(f"Step: 0 | CO Score: {self.co_scores[0]:.4f}")
        self.ax[1].axis('off')
        
        # 4. Add the Colorbar to track Attribution Intensity
        self.cbar = self.fig.colorbar(self.mask_plot, ax=self.ax[1], fraction=0.046, pad=0.04)
        self.cbar.set_label('Attribution intensity')
        
        # --- Setup Interactive Slider ---
        axcolor = 'lightgoldenrodyellow'
        ax_depth = plt.axes([0.2, 0.1, 0.6, 0.03], facecolor=axcolor)
        self.slider = Slider(ax_depth, 'Iteration', 0, self.steps - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.update)
        
    def process_heatmap(self, raw_heatmap):
        """Applies the channel isolation and resizing logic."""
        # Handle channels: take the first channel if it has multiple
        if raw_heatmap.shape[-1] in [1, 3]:
            raw_heatmap = raw_heatmap[..., 0]
            
        # Resize if necessary to match the base image dimensions
        if raw_heatmap.shape != self.img.shape[:2]:
            raw_heatmap = resize(raw_heatmap, self.img.shape[:2], preserve_range=True)
            
        return raw_heatmap

    def update(self, val):
        step = int(val)
        
        # Process the heatmap for the current slider step
        current_heatmap = self.process_heatmap(self.data[step])
        
        # Update the visual data
        self.mask_plot.set_data(current_heatmap)
        
        # Dynamically scale the colorbar so 'jet' colors represent the new min/max values
        self.mask_plot.set_clim(vmin=current_heatmap.min(), vmax=current_heatmap.max())
        
        self.title_text.set_text(f"Step: {step} | CO Score: {self.co_scores[step]:.4f}")
        self.fig.canvas.draw_idle()

# 5. --- The GAX Optimization Engine ---
def run_local_gax(img_path, model_path, target_label_idx, img_size=(224, 224), submethod='mult', n_iter=100, lr=0.01):
    # For reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Models
    model = load_trained_resnet(model_path, device)
    netG = Generator(img_size=img_size).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Load Data
    x = load_and_preprocess_image(img_path, device, size=img_size)
    
    # Get base prediction
    with torch.no_grad():
        base_logits = model(x)
        base_probs = F.softmax(base_logits, dim=1)
        pred_idx = torch.argmax(base_probs, dim=1).item()
        n_class = base_probs.shape[1]
        
    print(f"Initial Prediction: {pred_idx} | Target Optimization Label: {target_label_idx}")
    
    # History tracking
    imgs_history = []
    co_scores_history = []
    epsilon = 1e-4
    similarity_loss_factor = 1.0 # Lambda parameter
    
    print("Starting Generator Optimization...")
    for i in range(n_iter):
        netG.train()
        optimizerG.zero_grad()
        
        # 1. Generator creates perturbation matrix
        attr_op = netG(x)
        
        # 2. Apply perturbation
        if submethod == 'sum':
            x_aug = x + attr_op
        elif submethod == 'mult':
            x_aug = x * attr_op
        else:
            raise ValueError("submethod must be 'sum' or 'mult'")
            
        # 3. Model evaluates augmented image
        aug_logits = model(x_aug)
        aug_probs = F.softmax(aug_logits, dim=1)
        
        # 4. Compute CO Score
        score_constants = torch.zeros_like(aug_probs) - (1.0 / (n_class - 1))
        score_constants[0, target_label_idx] = 1.0
        
        co_score_tensor = (aug_probs - base_probs) * score_constants
        co_score = torch.sum(co_score_tensor)
        
        # 5. Calculate Loss 
        sim_loss = similarity_loss_factor / torch.mean((attr_op - x + epsilon)**2 / (x + epsilon))
        loss = -co_score + sim_loss
        
        # 6. Backpropagate and update GENERATOR parameters (W and b)
        loss.backward()
        optimizerG.step()
        
        # 7. Record data for visualization
        co_scores_history.append(co_score.item())
        
        # Save state every iteration for smooth slider visualization
        netG.eval()
        with torch.no_grad():
            current_mask = netG(x).cpu().numpy()[0].transpose(1, 2, 0)
            imgs_history.append(current_mask)
            
        if i % 10 == 0 or i == n_iter - 1:
            print(f"Iter {i:3d}/{n_iter} | Loss: {loss.item():.4f} | CO Score: {co_score.item():.4f}")

    print("Optimization Complete. Launching Interactive Plot...")
    
    # Launch GUI
    data_np = np.array(imgs_history)
    plotter = LocalInteractivePlot(data_np, co_scores_history, x)
    plt.show()

if __name__ == "__main__":
    # --- Execute Local GAX ---
    LOCAL_MODEL_PATH = "checkpoints/best_resnet34_v3.pth" 
    LOCAL_IMAGE_PATH = "jpeg_dataset/test/PNEUMONIA/0ae5dcc7-197d-4d59-991a-3b93f3a1e760.jpg" # Replace with valid local path
    
    # 0 = Normal, 1 = Pneumonia
    run_local_gax(
        img_path=LOCAL_IMAGE_PATH,
        model_path=LOCAL_MODEL_PATH,
        target_label_idx=1,   
        img_size=(224, 224),
        submethod='mult',
        n_iter=100,
        lr=0.01
    )