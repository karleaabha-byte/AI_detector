import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from PIL import Image
from scipy.stats import norm
from sklearn.metrics import classification_report, roc_curve, auc

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn():
    return load_model("my_model.h5")

model = load_cnn()
IMG_SIZE = (32, 32)

# ---------------- PREPROCESS ----------------
def preprocess_to_cifar(img):
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w-m)//2, (h-m)//2, (w+m)//2, (h+m)//2))
    img = img.resize(IMG_SIZE)

    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- UI ----------------
st.set_page_config(page_title="AI Detector", layout="wide")

st.markdown("<h1 style='text-align:center;'>👾 AI vs Real Image Detector</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction", "Model Statistics"])

# =====================================================
# TAB 1
# =====================================================
with tab1:
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if file:
        img = Image.open(file)
        st.image(img)

        x = preprocess_to_cifar(img)
        pred = float(model.predict(x)[0][0])

        label = "REAL IMAGE" if pred > 0.5 else "AI GENERATED"
        conf = pred if pred > 0.5 else 1-pred

        st.metric("Prediction", label)
        st.metric("Confidence", f"{conf:.4f}")

# =====================================================
# TAB 2
# =====================================================
with tab2:

    TN, FP, FN, TP = 8568, 1432, 366, 9634
    total = TN + FP + FN + TP

    # -------- METRICS --------
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    specificity = TN / (TN + FP)
    error = (FP + FN) / total

    st.subheader("📊 Performance Dashboard")

    st.write(f"Accuracy = {accuracy:.4f}")
    st.write(f"Precision = {precision:.4f}")
    st.write(f"Recall = {recall:.4f}")
    st.write(f"F1 Score = {f1:.4f}")
    st.write(f"Specificity = {specificity:.4f}")
    st.write(f"Error Rate = {error:.4f}")

    # -------- FORMULAS --------
    st.subheader("📐 Metric Formulas")

    st.latex(r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}")
    st.latex(rf"= \frac{{{TP}+{TN}}}{{{total}}} = {accuracy:.4f}")

    st.latex(r"Precision = \frac{TP}{TP + FP}")
    st.latex(rf"= \frac{{{TP}}}{{{TP}+{FP}}} = {precision:.4f}")

    st.latex(r"Recall = \frac{TP}{TP + FN}")
    st.latex(rf"= \frac{{{TP}}}{{{TP}+{FN}}} = {recall:.4f}")

    st.latex(r"F1 = \frac{2PR}{P+R}")
    st.latex(rf"= {f1:.4f}")

    # -------- CONFUSION MATRIX --------
    st.subheader("Confusion Matrix")
    cm = np.array([[TN, FP],[FN, TP]])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    st.pyplot(fig)

    # -------- LOAD CSV --------
    real_df = pd.read_csv("predictions_real.csv")
    ai_df = pd.read_csv("predictions_ai.csv")

    real_df["true"] = 0
    ai_df["true"] = 1

    df = pd.concat([real_df, ai_df])
    df["pred"] = df["prediction"].map({"REAL":0,"AI":1})

    y_true = df["true"]
    y_pred = df["pred"]

    # -------- CLASSIFICATION REPORT --------
    st.subheader("Classification Report")
    st.text(classification_report(y_true,y_pred,target_names=["REAL","AI"]))

    # -------- ROC --------
    st.subheader("ROC Curve")

    y_prob = df["probability"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax2.plot([0,1],[0,1],'--')
    ax2.legend()
    st.pyplot(fig2)

    # -------- MLE --------
    st.subheader("MLE")

    n = len(y_true)
    correct = (y_true == y_pred).sum()
    p_hat = correct / n

    st.latex(r"\hat{p} = \frac{\text{correct}}{n}")
    st.latex(rf"\hat{{p}} = \frac{{{correct}}}{{{n}}} = {p_hat:.4f}")

    # -------- HYPOTHESIS TEST --------
    st.subheader("Hypothesis Test")

    errors = (y_true != y_pred).sum()
    e_hat = errors / n
    e0 = 0.5

    z = (e_hat - e0) / np.sqrt((e0*(1-e0))/n)
    z_critical = norm.ppf(1-0.05/2)
    p_value = 2*(1-norm.cdf(abs(z)))

    st.latex(r"Z = \frac{\hat{e} - e_0}{\sqrt{\frac{e_0(1-e_0)}{n}}}")
    st.latex(rf"= \frac{{{e_hat:.4f}-0.5}}{{\sqrt{{\frac{{0.5(1-0.5)}}{{{n}}}}}}}")

    st.write(f"Z = {z:.2f}")
    st.write(f"P-value = {p_value:.6f}")

    if abs(z)>z_critical:
        st.success("Reject H0 → Model better than random")
    else:
        st.warning("Fail to reject H0")

    # -------- CLASS-WISE ACCURACY --------
    st.subheader("Class-wise Accuracy")

    real_acc = (real_df["prediction"]=="REAL").mean()
    ai_acc = (ai_df["prediction"]=="AI").mean()

    st.write(f"Real Accuracy = {real_acc*100:.2f}%")
    st.write(f"AI Accuracy = {ai_acc*100:.2f}%")

    # -------- 2-PROPORTION TEST --------
    st.subheader("Misclassification Comparison")

    n_real = len(real_df)
    n_ai = len(ai_df)

    x_real = (real_df["prediction"]=="AI").sum()
    x_ai = (ai_df["prediction"]=="REAL").sum()

    p_real = x_real/n_real
    p_ai = x_ai/n_ai

    p_pool = (x_real+x_ai)/(n_real+n_ai)
    SE = np.sqrt(p_pool*(1-p_pool)*(1/n_real + 1/n_ai))

    Z = (p_ai - p_real)/SE
    p_val = 2*(1-norm.cdf(abs(Z)))

    st.latex(r"Z = \frac{p_1 - p_2}{SE}")
    st.write(f"Z = {Z:.3f}")
    st.write(f"P-value = {p_val:.6f}")

    if abs(Z)>1.96:
        st.success("Significant difference between AI & Real errors")
    else:
        st.warning("No significant difference")
