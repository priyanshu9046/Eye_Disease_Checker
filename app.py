import streamlit as st
import torch
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
import torch.nn as nn
from PIL import Image

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(
    page_title="Eye Disease Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Title & Instructions
# -------------------------------
st.title("Eye Disease Classification")
st.write("Upload an image of an eye, and the model will predict the disease.")

# Two columns: instructions left, image & prediction right
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Instructions:")
    st.markdown("""
    1. Click the 'Browse files' button to upload a JPG or PNG image of an eye.  
    2. Ensure the image is clear and well-lit.  
    3. Wait a few seconds for the prediction to appear.  
    4. The predicted disease will be displayed neatly.  
    """)

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Load Model with Caching
# -------------------------------
@st.cache_resource
def load_model(model_path="best_model.pth", num_classes=3):
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# -------------------------------
# File Uploader & Prediction
# -------------------------------
with col2:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        input_tensor = preprocess_image(image)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            class_idx = pred.item()

        classes = ["Cataract", "Normal Eyes", "Uveitis"]
        st.markdown(
            f"<h2>Predicted Disease: {classes[class_idx]}</h2>",
            unsafe_allow_html=True
        )
