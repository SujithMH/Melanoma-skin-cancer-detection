import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.dataset import MelanomaDataset
from src.models.hybrid import HybridDeepNet

# Wrapper to isolate the classification head for Grad-CAM
class HybridClassificationWrapper(torch.nn.Module):
    def __init__(self, hybrid_model):
        super().__init__()
        self.hybrid_model = hybrid_model
        
    def forward(self, x):
        cls_out, _ = self.hybrid_model(x)
        return cls_out

def denormalize(tensor):
    """Reverses ImageNet normalization for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return img

def run_gradcam(device, num_images=5):
    print("\n--- Generating Grad-CAM Heatmaps ---")
    
    # 1. Load the optimized configuration
    config = {
        'fc_hidden': 448,
        'fusion_type': 'concat',
        'dropout': 0.357
    }
    
    model = HybridDeepNet(
        fusion_type=config['fusion_type'],
        fc_hidden=config['fc_hidden'],
        dropout=config['dropout']
    ).to(device)
    
    # 2. Load the optimized weights
    try:
        model.load_state_dict(torch.load('best_optimized_hybrid.pth', map_location=device))
        print("Successfully loaded best_optimized_hybrid.pth")
    except FileNotFoundError:
        print("Error: 'best_optimized_hybrid.pth' not found. Ensure Phase F saved it.")
        return
        
    model.eval()
    wrapped_model = HybridClassificationWrapper(model).to(device)
    wrapped_model.eval()
    
    # 3. Target the final convolutional layer of EfficientNet-B1
    target_layers = [model.cnn.conv_head]
    
    # 4. Load a few validation images (shuffle to get a random mix)
    val_dataset = MelanomaDataset('splits/val.csv', 'data/cache', is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=num_images, shuffle=True)
    
    images, cls_labels, _ = next(iter(val_loader))
    images = images.to(device)
    
    # 5. Initialize Grad-CAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    # Generate heatmaps
    grayscale_cams = cam(input_tensor=images, targets=None) # targets=None uses the highest scoring class
    
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    
    # 6. Plotting
    fig, axes = plt.subplots(num_images, 2, figsize=(8, 4 * num_images))
    if num_images == 1:
        axes = [axes]
        
    for i in range(num_images):
        # Prepare original image
        rgb_img = denormalize(images[i])
        
        # Prepare heatmap
        grayscale_cam = grayscale_cams[i, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        # Get predictions to label the plots
        with torch.no_grad():
            preds = wrapped_model(images[i].unsqueeze(0))
            pred_idx = torch.argmax(preds, dim=1).item()
            true_idx = cls_labels[i].item()
            
        pred_label = class_names[pred_idx]
        true_label = class_names[true_idx]
        
        # Plot Original
        axes[i][0].imshow(rgb_img)
        axes[i][0].set_title(f"Original\nTrue: {true_label}")
        axes[i][0].axis('off')
        
        # Plot Grad-CAM
        axes[i][1].imshow(cam_image)
        axes[i][1].set_title(f"Grad-CAM\nPred: {pred_label}")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig('gradcam_results.png', dpi=300, bbox_inches='tight')
    print("Saved visualization to 'gradcam_results.png'")
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_gradcam(device, num_images=5)