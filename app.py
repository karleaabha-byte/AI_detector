import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from PIL import Image

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report, roc_curve, auc
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
# TAB 1 — PREDICTION
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
# TAB 2 — STATISTICS
# =====================================================
with tab2:

    st.header("Model Evaluation")

    # LOAD CSV
    real_df = pd.read_csv("predictions_real.csv")
    ai_df = pd.read_csv("predictions_ai.csv")

    real_df["true_label"] = 0
    ai_df["true_label"] = 1

    df = pd.concat([real_df, ai_df], ignore_index=True)

    df["prediction"] = df["prediction"].map({
        "REAL":0,
        "AI":1
    })

    y_true = df["true_label"]
    y_pred = df["prediction"]

    # ---------------- METRICS ----------------
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Accuracy", round(accuracy,3))
    col2.metric("Precision", round(precision,3))
    col3.metric("Recall", round(recall,3))
    col4.metric("F1 Score", round(f1,3))

    # ---------------- CONFUSION MATRIX ----------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true,y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["REAL","AI"],
                yticklabels=["REAL","AI"], ax=ax)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    st.pyplot(fig)

    # ---------------- MLE ----------------
    st.subheader("Maximum Likelihood Estimation")

    n = len(y_true)
    correct = (y_true == y_pred).sum()
    p_hat = correct / n

    st.write("Total Samples:", n)
    st.write("Correct Predictions:", correct)
    st.success(f"MLE Accuracy = {p_hat:.4f}")

    # ---------------- HYPOTHESIS TEST ----------------
    st.subheader("Hypothesis Testing")

    errors = (y_true != y_pred).sum()
    e_hat = errors / n

    e0 = 0.5
    z = (e_hat - e0) / np.sqrt((e0*(1-e0))/n)

    alpha = 0.05
    z_critical = norm.ppf(1-alpha/2)
    p_value = 2*(1-norm.cdf(abs(z)))

    st.write("Error Rate:", e_hat)
    st.write("Z:", z)
    st.write("P-value:", p_value)

    if abs(z) > z_critical:
        if z < 0:
            st.success("Model significantly better than random")
        else:
            st.error("Model worse than random")
    else:
        st.warning("No significant difference")

    # ---------------- CLASSIFICATION REPORT ----------------
    st.subheader("Classification Report")
    st.text(classification_report(y_true,y_pred, target_names=["REAL","AI"]))

    # ---------------- ROC ----------------
    st.subheader("ROC Curve")

    y_prob = df["probability"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr,tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr,tpr,label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0,1],[0,1],'--')

    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend()

    st.pyplot(fig2)
