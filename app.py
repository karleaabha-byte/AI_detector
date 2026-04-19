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

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Detector", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0b1220;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 15px rgba(0, 0, 255, 0.3);
}

.metric-title {
    font-size: 16px;
    opacity: 0.85;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
}

/* Section headers */
.section {
    color: #3b82f6;
    font-size: 22px;
    margin-top: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

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

    # -------- CONFUSION MATRIX VALUES (MATCH IMAGE) --------
    TP = 8568
    FN = 1432
    FP = 366
    TN = 9634

    total = TP + TN + FP + FN

    # -------- METRICS --------
    accuracy = (TP + TN) / total
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    specificity = TN / (TN + FP)
    error = (FP + FN) / total

    st.markdown("<div class='section'>📊 Performance Dashboard</div>", unsafe_allow_html=True)

    def card(t,v):
        return f"<div class='metric-card'><div class='metric-title'>{t}</div><div class='metric-value'>{v}</div></div>"

    c1,c2,c3 = st.columns(3)
    c4,c5,c6 = st.columns(3)

    c1.markdown(card("Accuracy",f"{accuracy:.4f}"),True)
    c2.markdown(card("Precision",f"{precision:.4f}"),True)
    c3.markdown(card("Recall",f"{recall:.4f}"),True)
    c4.markdown(card("F1 Score",f"{f1:.4f}"),True)
    c5.markdown(card("Specificity",f"{specificity:.4f}"),True)
    c6.markdown(card("Error Rate",f"{error:.4f}"),True)

    # -------- CONFUSION MATRIX (FIXED ORIENTATION) --------
    st.markdown("<div class='section'>Confusion Matrix</div>", unsafe_allow_html=True)

    # IMPORTANT: matches your image layout
    cm = np.array([
        [TP, FN],   # Actual AI (1)
        [FP, TN]    # Actual Real (0)
    ])

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        linewidths=1,
        linecolor='white',
        xticklabels=["Predicted AI (1)", "Predicted Real (0)"],
        yticklabels=["Actual AI (1)", "Actual Real (0)"],
        ax=ax
    )

    ax.set_title("Confusion Matrix")
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
    st.markdown("<div class='section'>Classification Report</div>", unsafe_allow_html=True)

    report_dict = classification_report(y_true, y_pred, target_names=["REAL","AI"], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.dataframe(report_df.style.background_gradient(cmap="Blues"))

    # -------- ROC CURVE --------
    st.markdown("<div class='section'>ROC Curve</div>", unsafe_allow_html=True)

    y_prob = df["probability"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", color="#2563eb")
    ax2.plot([0,1],[0,1],'--', color="gray")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("ROC Curve")
    ax2.legend()

    st.pyplot(fig2)

    # -------- MLE --------
    st.markdown("<div class='section'>MLE (Maximum Likelihood Estimation)</div>", unsafe_allow_html=True)
    
    n = len(y_true)
    correct = (y_true == y_pred).sum()
    p_hat = correct / n
    
    st.latex(r"\hat{p} = \frac{\text{correct predictions}}{\text{total samples}}")
    st.latex(rf"\hat{{p}} = \frac{{{correct}}}{{{n}}}")
    st.latex(rf"\hat{{p}} = {p_hat:.4f}")
    
    st.success(f"Final MLE Accuracy = {p_hat:.4f} ({p_hat*100:.2f}%)")

    # -------- HYPOTHESIS TEST --------
    st.markdown("<div class='section'>Hypothesis Testing</div>", unsafe_allow_html=True)

    errors = (y_true != y_pred).sum()
    e_hat = errors / n
    e0 = 0.5

    z = (e_hat - e0) / np.sqrt((e0*(1-e0))/n)
    z_critical = norm.ppf(1-0.05/2)
    p_value = 2*(1-norm.cdf(abs(z)))

    st.write(f"Z Statistic: {z:.3f}")
    st.write(f"Z Critical: ±{z_critical:.3f}")
    st.write(f"P-value: {p_value:.6f}")

    if abs(z)>z_critical:
        st.success("Reject H0 → Model better than random")
    else:
        st.warning("Fail to reject H0")

    # -------- CLASS-WISE ACCURACY --------
    st.markdown("<div class='section'>Class-wise Accuracy</div>", unsafe_allow_html=True)

    real_acc = (real_df["prediction"]=="REAL").mean()
    ai_acc = (ai_df["prediction"]=="AI").mean()

    st.write(f"Real Accuracy: {real_acc*100:.2f}%")
    st.write(f"AI Accuracy: {ai_acc*100:.2f}%")
