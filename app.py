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
# Preprocess image
# ------------------------
def preprocess(img):
    # Resize image to 32x32
    img = img.resize(IMG_SIZE)
    # Convert to numpy array and normalize
    img = np.array(img) / 255.0
    # Ensure it has 3 channels
    if img.shape[-1] != 3:
        img = np.stack((img,) * 3, axis=-1)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# ------------------------
# Streamlit UI
# ------------------------
st.title("AI vs Real Image Detector")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Resize and preprocess
    img_input = preprocess(img)

    # Predict
    prediction = float(model.predict(img_input)[0][0])

    if prediction > 0.5:
        label = "REAL IMAGE"
        confidence = prediction
    else:
        label = "AI GENERATED"
        confidence = 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence:.4f}")
