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
# Preprocess image safely
# ------------------------
def preprocess(img):
    # Ensure RGB
    img = img.convert("RGB")
    
    # Resize while maintaining aspect ratio
    img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
    
    # Create new 32x32 black background and paste resized image in center
    new_img = Image.new("RGB", IMG_SIZE, (0, 0, 0))
    x_offset = (IMG_SIZE[0] - img.size[0]) // 2
    y_offset = (IMG_SIZE[1] - img.size[1]) // 2
    new_img.paste(img, (x_offset, y_offset))
    
    # Convert to numpy array and normalize
    img_array = np.array(new_img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------
# Streamlit UI
# ------------------------
st.title("AI vs Real Image Detector")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img_input = preprocess(img)

    # Predict
    prediction = float(model.predict(img_input)[0][0])

    # Optional: adjust threshold if needed
    threshold = 0.5
    if prediction > threshold:
        label = "REAL IMAGE"
        confidence = prediction
    else:
        label = "AI GENERATED"
        confidence = 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence:.4f}")
