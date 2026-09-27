import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ecosort import models
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def load_explainer_model(model_path, device):
    """Loads the model once and unfreezes it for gradient math."""
    model = models.build_EfficientNet() 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Unfreeze all parameters so Grad-CAM can calculate the math
    for param in model.parameters():
        param.requires_grad = True
        
    return model

def generate_heatmap(image_path, model_path, device):
    fresh_model = models.build_EfficientNet() 
    fresh_model.load_state_dict(torch.load(model_path, map_location=device))
    fresh_model = fresh_model.to(device)
    fresh_model.eval()

    # 2. Unfreeze weights for the math
    for param in fresh_model.parameters():
        param.requires_grad = True

    target_layers = [fresh_model.features[-1]]

    raw_img = Image.open(image_path).convert('RGB')
    img_resized = raw_img.resize((224, 224))
    rgb_img = np.float32(img_resized) / 255 

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # 3. Generate heatmap
    with GradCAM(model=fresh_model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
        
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return rgb_img, visualization