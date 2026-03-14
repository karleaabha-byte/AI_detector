import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------
# Load model
# ------------------------
model = load_model("my_model.h5")
IMG_SIZE = (32, 32)  # CIFAR input size

# ------------------------
# Preprocess any image to CIFAR-style
# ------------------------
def preprocess_to_cifar(img):
    # Ensure RGB
    img = img.convert("RGB")
    
    # Center crop to square
    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize to 32x32 (CIFAR)
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize to [0,1]
    img_array = np.array(img) / 255.0
    
    # Optional: normalize using CIFAR-10 mean/std (uncomment if needed)
    # mean = np.array([0.4914, 0.4822, 0.4465])
    # std  = np.array([0.2470, 0.2435, 0.2616])
    # img_array = (img_array - mean) / std
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ------------------------
# Streamlit UI
# ------------------------
st.title("AI vs Real Image Detector (CIFAR-style)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file)
    
    # Display original image
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess
    img_input = preprocess_to_cifar(img)
    
    # Predict
    prediction = float(model.predict(img_input)[0][0])
    
    # Threshold for real vs AI
    threshold = 0.5
    if prediction > threshold:
        label = "REAL IMAGE"
        confidence = prediction
    else:
        label = "AI GENERATED"
        confidence = 1 - prediction
    
    # Display result
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence:.4f}")
