import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from PIL import Image
from scipy.stats import norm

# ------------------------
# LOAD MODEL
# ------------------------
@st.cache_resource
def load_cnn():
    return load_model("my_model.h5")

model = load_cnn()
IMG_SIZE = (32, 32)

# ------------------------
# PREPROCESS
# ------------------------
def preprocess_to_cifar(img):
    img = img.convert("RGB")

    w, h = img.size
    m = min(w, h)
    img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    img = img.resize(IMG_SIZE)

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="AI vs Real Detector", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
    }
    .metric-title {
        font-size: 18px;
        opacity: 0.8;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AI vs Real Image Detector")

tab1, tab2 = st.tabs(["Prediction", "Model Statistics"])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        img = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded Image")

        img_input = preprocess_to_cifar(img)
        prediction = float(model.predict(img_input)[0][0])

        if prediction > 0.5:
            label = "REAL IMAGE"
            confidence = prediction
        else:
            label = "AI GENERATED"
            confidence = 1 - prediction

        with col2:
            st.subheader("Prediction")
            st.metric("Result", label)
            st.metric("Confidence", f"{confidence:.4f}")

# =====================================================
# TAB 2
# =====================================================
with tab2:

    # ---------------- CONFUSION MATRIX ----------------
    TN, FP, FN, TP = 8568, 1432, 366, 9634
    total = TN + FP + FN + TP

    # ---------------- METRICS ----------------
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    specificity = TN / (TN + FP)
    error = (FP + FN) / total

    # ---------------- TOP DASHBOARD ----------------
    st.markdown("## 📊 Model Performance Dashboard")

    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)

    def card(title, value):
        return f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """

    col1.markdown(card("Accuracy", f"{accuracy:.4f}"), unsafe_allow_html=True)
    col2.markdown(card("Precision", f"{precision:.4f}"), unsafe_allow_html=True)
    col3.markdown(card("Recall", f"{recall:.4f}"), unsafe_allow_html=True)

    col4.markdown(card("F1 Score", f"{f1:.4f}"), unsafe_allow_html=True)
    col5.markdown(card("Specificity", f"{specificity:.4f}"), unsafe_allow_html=True)
    col6.markdown(card("Error Rate", f"{error:.4f}"), unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm = np.array([[TN, FP], [FN, TP]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["REAL","AI"],
                yticklabels=["REAL","AI"], ax=ax)

    st.pyplot(fig)

    # ---------------- MLE ----------------
    st.subheader("Maximum Likelihood Estimation")

    correct = TP + TN
    p_hat = correct / total

    st.latex(r"\hat{p} = \frac{\text{correct}}{n}")
    st.latex(rf"\hat{{p}} = \frac{{{correct}}}{{{total}}}")

    st.success(f"MLE Accuracy = {p_hat:.4f}")

    # ---------------- HYPOTHESIS TEST ----------------
    st.subheader("Hypothesis Testing")

    errors = FP + FN
    e_hat = errors / total
    e0 = 0.5

    z = (e_hat - e0) / np.sqrt((e0*(1-e0))/total)

    st.latex(r"Z = \frac{\hat{e} - e_0}{\sqrt{\frac{e_0(1-e_0)}{n}}}")
    st.write(f"Z = {z:.2f}")

    if abs(z) > 1.96:
        st.success("Model is significantly better than random")
    else:
        st.warning("No significant difference")
