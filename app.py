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
# PREPROCESS FUNCTION
# ------------------------
def preprocess_to_cifar(img):
    img = img.convert("RGB")

    width, height = img.size
    min_dim = min(width, height)
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))

    img = img.resize(IMG_SIZE, Image.Resampling.LANCZOS)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="AI vs Real Detector", layout="wide")
st.title("AI vs Real Image Detector")

tab1, tab2 = st.tabs(["Prediction", "Model Statistics"])

# =====================================================
# TAB 1 — IMAGE PREDICTION
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

        threshold = 0.5
        if prediction > threshold:
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
# TAB 2 — MODEL STATISTICS WITH CALCULATIONS
# =====================================================
with tab2:

    st.header("Model Evaluation (With Calculations)")

    # ----------------------------
    # CONFUSION MATRIX (GIVEN)
    # ----------------------------
    TN = 8568
    FP = 1432
    FN = 366
    TP = 9634

    cm = np.array([[TN, FP],
                   [FN, TP]])

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    sns.heatmap(cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=["REAL","AI"],
                yticklabels=["REAL","AI"],
                ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    st.pyplot(fig)

    # ----------------------------
    # METRICS
    # ----------------------------
    st.subheader("Metrics with Formula")

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    st.write(f"Accuracy = (TP + TN) / Total = ({TP} + {TN}) / {total} = {accuracy:.4f}")
    st.write(f"Precision = TP / (TP + FP) = {TP} / ({TP} + {FP}) = {precision:.4f}")
    st.write(f"Recall = TP / (TP + FN) = {TP} / ({TP} + {FN}) = {recall:.4f}")
    st.write(f"F1 Score = 2PR/(P+R) = {f1:.4f}")

    # ----------------------------
    # MLE
    # ----------------------------
    st.subheader("Maximum Likelihood Estimation (MLE)")

    correct = TP + TN
    n = total

    p_hat = correct / n

    st.write("Formula:")
    st.latex(r"\hat{p} = \frac{\text{correct}}{n}")

    st.write("Substitution:")
    st.latex(rf"\hat{{p}} = \frac{{{correct}}}{{{n}}}")

    st.success(f"MLE Estimate of Accuracy = {p_hat:.4f}")

    # ----------------------------
    # HYPOTHESIS TEST
    # ----------------------------
    st.subheader("Hypothesis Testing")

    errors = FP + FN
    e_hat = errors / n

    e0 = 0.5

    z = (e_hat - e0) / np.sqrt((e0 * (1 - e0)) / n)

    alpha = 0.05
    z_critical = norm.ppf(1 - alpha/2)
    p_value = 2 * (1 - norm.cdf(abs(z)))

    st.write("Hypotheses:")
    st.latex(r"H_0: e = 0.5")
    st.latex(r"H_1: e \neq 0.5")

    st.write("Z-test formula:")
    st.latex(r"Z = \frac{\hat{e} - e_0}{\sqrt{\frac{e_0(1-e_0)}{n}}}")

    st.write("Substitution:")
    st.latex(rf"\hat{{e}} = \frac{{{errors}}}{{{n}}} = {e_hat:.4f}")
    st.latex(rf"Z = \frac{{{e_hat:.4f} - 0.5}}{{\sqrt{{\frac{{0.5(1-0.5)}}{{{n}}}}}}}")

    st.write(f"Z = {z:.4f}")
    st.write(f"Z critical = ±{z_critical:.4f}")
    st.write(f"P-value = {p_value:.6f}")

    if abs(z) > z_critical:
        if z < 0:
            st.success("Reject H0 → Model is significantly better than random guessing")
        else:
            st.error("Reject H0 → Model is worse than random guessing")
    else:
        st.warning("Fail to reject H0 → No significant difference")
