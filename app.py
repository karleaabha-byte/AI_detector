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
body {background-color:#0b1220;}

.metric-card {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    color: white;
    box-shadow: 0px 4px 15px rgba(0,0,255,0.3);
}

.metric-title {font-size:16px;}
.metric-value {font-size:26px; font-weight:bold;}

.section {
    color:#3b82f6;
    font-size:22px;
    margin-top:20px;
}

.formula-box {
    background:#111827;
    padding:20px;
    border-radius:12px;
    text-align:center;
    margin:15px 0;
}

.info-box {
    background:#0f172a;
    padding:12px;
    border-left:4px solid #3b82f6;
    border-radius:8px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>👾 AI vs Real Image Detector</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Prediction", "Model Statistics"])

# ================= TAB 1 =================
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

# ================= TAB 2 =================
with tab2:

    TN, FP, FN, TP = 8568, 1432, 366, 9634
    total = TN + FP + FN + TP

    accuracy = (TP + TN) / total
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    st.markdown("<div class='section'>📊 Performance Dashboard</div>", True)

    def card(t,v):
        return f"<div class='metric-card'><div class='metric-title'>{t}</div><div class='metric-value'>{v}</div></div>"

    c1,c2,c3,c4 = st.columns(4)
    c1.markdown(card("Accuracy",f"{accuracy:.4f}"),True)
    c2.markdown(card("Precision",f"{precision:.4f}"),True)
    c3.markdown(card("Recall",f"{recall:.4f}"),True)
    c4.markdown(card("F1 Score",f"{f1:.4f}"),True)

    # -------- CONFUSION MATRIX --------
    st.markdown("<div class='section'>Confusion Matrix</div>", True)
    cm = np.array([[TN, FP],[FN, TP]])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
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
    st.markdown("<div class='section'>Classification Report</div>", True)
    report = pd.DataFrame(classification_report(y_true,y_pred,output_dict=True)).transpose()
    st.dataframe(report.style.background_gradient(cmap="Blues"))

    # -------- ROC --------
    st.markdown("<div class='section'>ROC Curve</div>", True)
    y_prob = df["probability"]

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr,tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
    ax2.plot([0,1],[0,1],'--')
    ax2.legend()
    st.pyplot(fig2)

    # -------- MLE --------
    st.markdown("<div class='section'>MLE</div>", True)

    n = len(y_true)
    correct = (y_true == y_pred).sum()
    p_hat = correct / n

    st.markdown("<div class='info-box'>Estimate of accuracy using MLE</div>", True)

    st.markdown("<div class='formula-box'>", True)
    st.latex(r"\hat{p} = \frac{\text{correct}}{n}")
    st.latex(rf"\hat{{p}} = \frac{{{correct}}}{{{n}}} = {p_hat:.4f}")
    st.markdown("</div>", True)

    # -------- HYPOTHESIS --------
    st.markdown("<div class='section'>Hypothesis Test</div>", True)

    errors = (y_true != y_pred).sum()
    e_hat = errors / n

    z = (e_hat-0.5)/np.sqrt((0.5*0.5)/n)
    p_val = 2*(1-norm.cdf(abs(z)))

    st.markdown("<div class='formula-box'>", True)
    st.latex(r"H_0: e=0.5 \quad H_1:e\neq0.5")
    st.latex(rf"Z = {z:.3f}")
    st.markdown("</div>", True)

    st.write(f"Error Rate: {e_hat*100:.2f}%")
    st.write(f"P-value: {p_val:.6f}")

    # -------- MISCLASSIFICATION --------
    st.markdown("<div class='section'>Misclassification Test</div>", True)

    n_real = len(real_df)
    n_ai = len(ai_df)

    x_real = (real_df["prediction"]=="AI").sum()
    x_ai = (ai_df["prediction"]=="REAL").sum()

    p_real = x_real/n_real
    p_ai = x_ai/n_ai

    p_pool = (x_real+x_ai)/(n_real+n_ai)
    SE = np.sqrt(p_pool*(1-p_pool)*(1/n_real+1/n_ai))

    Z = (p_ai-p_real)/SE
    p_val = 2*(1-norm.cdf(abs(Z)))

    st.markdown("<div class='formula-box'>", True)
    st.latex(rf"Z = {Z:.3f}")
    st.markdown("</div>", True)

    st.write(f"Real Misclassification: {p_real*100:.2f}%")
    st.write(f"AI Misclassification: {p_ai*100:.2f}%")
