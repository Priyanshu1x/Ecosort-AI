import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ecosort import models

def generate_heatmap(image_path, model_path):
    # 1. Set up device and load the winning EfficientNet model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.build_EfficientNet() 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Target the final convolutional layer of EfficientNet
    target_layers = [model.features[-1]]

    # 3. Load and prepare the image for both visualization and the model
    raw_img = Image.open(image_path).convert('RGB')
    img_resized = raw_img.resize((224, 224))
    
    # Scale pixels to 0-1 range for the heatmap overlay function
    rgb_img = np.float32(img_resized) / 255 
    
    # Standard transforms exactly as your DataLoader used them
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_resized).unsqueeze(0).to(device)

    # 4. Initialize Grad-CAM and generate the grayscale heatmap
    cam = GradCAM(model=model, target_layers=target_layers)
    # Setting targets=None automatically tells Grad-CAM to highlight what caused its #1 prediction
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # 5. Overlay the heatmap onto the color image
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Plot the before and after
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title("AI Attention (Grad-CAM)")
    plt.axis('off')
    
    plt.show()

# You can test it by running this file directly
# if __name__ == "__main__":
#     # Point this to one actual image in your Colab dataset folder
#     test_image = "/content/dataset/ewaste/some_image_name.jpg" 
#     best_model = "/content/drive/MyDrive/Ecosort/checkpoints/efficientnet.pth"
    
    # Uncomment this when running in Colab:
    # generate_heatmap(test_image, best_model)