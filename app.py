pip install tensorflow
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
# ------------------------
# Load model
# ------------------------
model = load_model("my_model.h5")
IMG_SIZE = (32, 32)

# ------------------------
# Preprocess image
# ------------------------
def preprocess(img):
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
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

    img_input = preprocess(img)

    prediction = model.predict(img_input)[0][0]

    if prediction > 0.5:
        label = "REAL IMAGE"
    else:
        label = "AI GENERATED"

    confidence = prediction if label == "REAL IMAGE" else 1 - prediction

    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence Score: {confidence:.4f}")
