import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------
# Load model
# ------------------------
model = load_model("my_model.h5")
IMG_SIZE = (32, 32)  # Model input size

# ------------------------
# CIFAR-style preprocessing
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
    
    # Resize to 32x32
    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Expand dimensions for batch
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# ------------------------
# Streamlit UI
# ------------------------
st.title("AI vs Real Image Detector (CIFAKE style)")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_input = preprocess_to_cifar(img)

    # Predict
    prediction = float(model.predict(img_input)[0][0])

    # Threshold (you can adjust if needed)
    threshold = 0.5
    if prediction > threshold:
        label = "REAL IMAGE"
        confidence = prediction
    else:
        label = "AI GENERATED"
        confidence = 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence:.4f}")
