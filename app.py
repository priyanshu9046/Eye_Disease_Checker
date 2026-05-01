import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

# ─────────────────────────────────────────────────
# Streamlit Page Config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Eye Disease AI Classifier",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# Custom CSS – Premium Dark UI
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    border-right: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

/* Cards */
.info-card {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}

/* Disease tag badge */
.disease-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 10px;
}
.badge-cataract  { background: rgba(255,160,50,0.2);  color: #ffb347; border: 1px solid #ffb347; }
.badge-normal    { background: rgba(50,200,100,0.2);  color: #4cde7a; border: 1px solid #4cde7a; }
.badge-uveitis   { background: rgba(220,70,70,0.2);   color: #ff6b6b; border: 1px solid #ff6b6b; }

/* Confidence bar wrapper */
.conf-bar-wrapper {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    overflow: hidden;
    height: 10px;
    margin-top: 4px;
    margin-bottom: 14px;
}
.conf-bar-fill {
    height: 10px;
    border-radius: 8px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    transition: width 0.6s ease;
}

/* Section headers */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 8px;
}

/* Disclaimer box */
.disclaimer {
    background: rgba(255,200,50,0.08);
    border-left: 4px solid #facc15;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #fde68a;
    margin-top: 16px;
}

/* ── Eliminate black top gap (keep sidebar toggle alive) ── */
[data-testid="stHeader"] {
    background: transparent !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    border-bottom: none !important;
}

/* Keep the sidebar collapse/expand toggle always visible */
[data-testid="collapsedControl"] {
    display: flex !important;
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 9999 !important;
    background: rgba(167, 139, 250, 0.2) !important;
    border: 1px solid rgba(167, 139, 250, 0.4) !important;
    border-radius: 8px !important;
    padding: 4px !important;
    backdrop-filter: blur(8px) !important;
    cursor: pointer !important;
    transition: background 0.2s ease !important;
}
[data-testid="collapsedControl"]:hover {
    background: rgba(167, 139, 250, 0.4) !important;
}

/* Remove ALL top padding/margin from the main container */
[data-testid="stAppViewContainer"] > .main > .block-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding-top: 0 !important;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(120deg, rgba(167,139,250,0.15) 0%, rgba(96,165,250,0.12) 100%);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 20px;
    padding: 32px 40px 28px 40px;
    margin-bottom: 28px;
    margin-top: 12px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    border-radius: 20px 20px 0 0;
}
.hero-tag {
    display: inline-block;
    background: rgba(167,139,250,0.2);
    border: 1px solid rgba(167,139,250,0.4);
    color: #c4b5fd;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 14px;
    border-radius: 50px;
    margin-bottom: 14px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #e0d7ff, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px 0;
    line-height: 1.15;
    letter-spacing: -0.5px;
}
.hero-subtitle {
    font-size: 0.97rem;
    color: #9ca3af;
    margin: 0;
    font-weight: 400;
}
.hero-dots {
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.08;
    user-select: none;
    pointer-events: none;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────
CLASSES = ["Cataract", "Normal Eyes", "Uveitis"]
BADGE_CLASSES = ["badge-cataract", "badge-normal", "badge-uveitis"]

DISEASE_INFO = {
    "Cataract": {
        "emoji": "🌫️",
        "description": (
            "A <strong>cataract</strong> is a clouding of the eye&#39;s natural lens, leading to decreased vision. "
            "It is one of the most common causes of vision loss worldwide and is highly treatable "
            "with surgery. Early signs include blurry or foggy vision, sensitivity to light, and "
            "faded colours."
        ),
        "treatment": "Surgical removal and replacement with an artificial intraocular lens (IOL).",
    },
    "Normal Eyes": {
        "emoji": "✅",
        "description": (
            "The eye appears <strong>healthy</strong> with no detectable signs of the conditions in this classifier. "
            "Continue regular eye check-ups and protect your eyes from UV radiation."
        ),
        "treatment": "No immediate treatment required. Maintain routine eye care.",
    },
    "Uveitis": {
        "emoji": "🔴",
        "description": (
            "<strong>Uveitis</strong> is inflammation of the uvea — the middle layer of the eye. It can cause "
            "redness, pain, blurred vision, and sensitivity to light. If left untreated it may lead "
            "to vision-threatening complications such as glaucoma, cataracts, or retinal damage."
        ),
        "treatment": "Anti-inflammatory eye drops, corticosteroids, or immunosuppressants under medical supervision.",
    },
}

# ─────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────
# Image Preprocessing
# ─────────────────────────────────────────────────
def preprocess_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor


# ─────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────
def generate_gradcam(model, input_tensor, class_idx, original_image):
    """
    Generates a Grad-CAM heatmap overlaid on the original image.

    Args:
        model         : Loaded PyTorch model (eval mode).
        input_tensor  : Preprocessed image tensor (1, C, H, W), requires_grad=True.
        class_idx     : Predicted class index.
        original_image: PIL Image (RGB) – used for overlay size.

    Returns:
        PIL Image of the heatmap overlaid on the original image.
    """
    activations = {}
    gradients   = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    # Register hooks on last conv block
    target_layer = model.layer4[-1]
    fwd_handle   = target_layer.register_forward_hook(forward_hook)
    bwd_handle   = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass (with gradients)
    model.zero_grad()
    outputs = model(input_tensor)

    # Backward pass on predicted class score
    score = outputs[0, class_idx]
    score.backward()

    # Remove hooks
    fwd_handle.remove()
    bwd_handle.remove()

    # Compute Grad-CAM weights
    grads = gradients["value"]          # (1, C, H, W)
    acts  = activations["value"]        # (1, C, H, W)
    weights = grads.mean(dim=(2, 3), keepdim=True)   # global average pooling

    # Weighted sum of activations
    cam = (weights * acts).sum(dim=1, keepdim=True)  # (1, 1, H, W)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize to [0, 1]
    if cam.max() > cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros_like(cam)

    # Resize to original image size
    orig_w, orig_h = original_image.size
    cam_resized = cv2.resize(cam, (orig_w, orig_h))

    # Apply colour map
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    orig_np = np.array(original_image.resize((orig_w, orig_h)))
    overlay = cv2.addWeighted(orig_np, 0.55, heatmap, 0.45, 0)

    return Image.fromarray(overlay)


# ─────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-dots">👁️</div>
    <div class="hero-tag">⚕️ AI-Powered Diagnostics</div>
    <h1 class="hero-title">OcuScan AI</h1>
    <p class="hero-subtitle">
        Eye Disease Classification &amp; Explainability &nbsp;·&nbsp;
        ResNet-18 &nbsp;·&nbsp; Grad-CAM &nbsp;·&nbsp; Instant Predictions
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────
# Sidebar – Instructions & About
# ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:16px 0 8px 0;">'
        '<span style="font-size:2rem;">👁️</span><br>'
        '<span style="font-size:1.3rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#60a5fa);'
        '-webkit-background-clip:text;-webkit-text-fill-color:transparent;">OcuScan AI</span><br>'
        '<span style="font-size:0.72rem;color:#6b7280;letter-spacing:1px;">EYE DISEASE CLASSIFIER</span>'
        '</div>',
        unsafe_allow_html=True
    )
    st.divider()
    st.markdown("## 📋 How to Use")
    st.markdown("""
1. Upload a **clear, well-lit** eye image (JPG / PNG).
2. Wait a moment for the model to analyse it.
3. Review the **prediction**, **confidence**, and the **Grad-CAM heatmap** showing which region the model focused on.
    """)
    st.divider()
    st.markdown("## 🧠 Detectable Conditions")
    st.markdown("""
| Condition | Description |
|-----------|-------------|
| 🌫️ Cataract | Clouding of the lens |
| ✅ Normal | Healthy eye |
| 🔴 Uveitis | Uveal inflammation |
    """)
    st.divider()
    st.markdown("## ⚙️ Model Info")
    st.markdown(f"""
- **Architecture:** ResNet-18  
- **Classes:** {len(CLASSES)}  
- **Device:** `{device}`
    """)

# ─────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<p class="section-header">📤 Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG eye image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

# ─────────────────────────────────────────────────
# Prediction & Grad-CAM (only when image uploaded)
# ─────────────────────────────────────────────────
if uploaded_file:
    with st.spinner("🔍 Analysing image…"):

        # Preprocess
        input_tensor = preprocess_image(image)
        input_tensor.requires_grad = True   # needed for Grad-CAM

        # Forward pass – keep gradients
        outputs = model(input_tensor)

        # Softmax → probabilities
        probs      = torch.softmax(outputs, dim=1)[0]
        class_idx  = torch.argmax(probs).item()
        confidence = probs[class_idx].item() * 100

        predicted_class = CLASSES[class_idx]
        badge_cls       = BADGE_CLASSES[class_idx]

        # Grad-CAM heatmap
        gradcam_image = generate_gradcam(model, input_tensor, class_idx, image)

    # ── Results Column ───────────────────────────
    with col_results:

        # ── Prediction ──────────────────────────
        st.markdown('<p class="section-header">🎯 Prediction</p>', unsafe_allow_html=True)
        st.markdown(
            f'<span class="disease-badge {badge_cls}">'
            f'{DISEASE_INFO[predicted_class]["emoji"]} {predicted_class}'
            f'</span>',
            unsafe_allow_html=True
        )

        # ── Confidence ──────────────────────────
        st.markdown('<p class="section-header">📊 Confidence</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="conf-bar-wrapper">'
            f'<div class="conf-bar-fill" style="width:{confidence:.1f}%"></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(f"**{confidence:.1f}%** confidence")

        # ── All Class Probabilities ──────────────
        st.markdown('<p class="section-header">📈 All Probabilities</p>', unsafe_allow_html=True)
        for i, cls in enumerate(CLASSES):
            p = probs[i].item() * 100
            st.markdown(
                f'<div style="margin-bottom:6px;">'
                f'<span style="min-width:110px;display:inline-block;font-size:0.9rem;">{cls}</span>'
                f'<span style="font-size:0.9rem;color:#a78bfa;font-weight:600;">{p:.1f}%</span>'
                f'</div>'
                f'<div class="conf-bar-wrapper">'
                f'<div class="conf-bar-fill" style="width:{p:.1f}%"></div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Grad-CAM Full Width ──────────────────────
    st.divider()
    col_cam, col_info = st.columns([1, 1], gap="large")

    with col_cam:
        st.markdown('<p class="section-header">🔥 Model Focus (Grad-CAM)</p>', unsafe_allow_html=True)
        st.image(
            gradcam_image,
            caption="Red/warm regions show where the model focused most",
            use_container_width=True
        )

    with col_info:
        # ── Disease Info ────────────────────────
        info = DISEASE_INFO[predicted_class]
        st.markdown('<p class="section-header">🩺 Disease Information</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="info-card">'
            f'<h4 style="margin-top:0;">{info["emoji"]} {predicted_class}</h4>'
            f'{info["description"]}<br><br>'
            f'<strong>💊 Common Treatment:</strong> {info["treatment"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # ── Medical Disclaimer ──────────────────
        st.markdown(
            '<div class="disclaimer">'
            '⚠️ <strong>Medical Disclaimer:</strong> This tool is for <em>educational purposes only</em> '
            'and is <strong>not</strong> a replacement for professional medical diagnosis. '
            'Always consult a qualified ophthalmologist for clinical advice.'
            '</div>',
            unsafe_allow_html=True
        )
