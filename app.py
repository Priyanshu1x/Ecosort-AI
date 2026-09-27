import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from ecosort import models, explainability

# --- 1. Page Setup ---
st.set_page_config(page_title="Ecosort AI", layout="wide")
st.title("♻️ Ecosort AI: Waste Classifier")
st.write("Upload a photo or use your webcam to classify waste and see what the AI focuses on.")

# --- 2. Load Model (Cached so it only loads once) ---
@st.cache_resource
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Update this path to where your model is saved locally!
    model_path = "checkpoints/efficientnet.pth" 
    model = explainability.load_explainer_model(model_path, device)
    return model, device

model, device = load_system()
class_names = ['Dry', 'Wet', 'Recyclable', 'ewaste']

# --- 3. Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. UI: File Uploader & Webcam ---
input_method = st.radio("Choose Input Method:", ("Upload Image", "Webcam"))

img_file = None
if input_method == "Upload Image":
    img_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
else:
    img_file = st.camera_input("Take a picture")

# --- 5. Prediction & Explainability Logic ---
if img_file is not None:
    col1, col2 = st.columns(2)
    
    # Load and show original image
    image = Image.open(img_file).convert('RGB')
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    # Make Prediction
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Get top 3 predictions
    top_prob, top_catid = torch.topk(probabilities, 3)
    
    st.subheader(f"Prediction: **{class_names[top_catid[0]]}**")
    st.write(f"Confidence: {top_prob[0].item() * 100:.2f}%")
    
    st.write("**Top Alternatives:**")
    for i in range(1, 3):
        st.write(f"- {class_names[top_catid[i]]}: {top_prob[i].item() * 100:.2f}%")

    # Generate and display Grad-CAM
    with st.spinner('Generating AI Explainability Map...'):
        # Save temp image for Grad-CAM function
        temp_path = "temp_img.jpg"
        image.save(temp_path)
        
        _, heatmap_img = explainability.generate_heatmap(temp_path, "checkpoints/efficientnet.pth", device)
        
        with col2:
            st.subheader("AI Attention (Grad-CAM)")
            st.image(heatmap_img, use_container_width=True)