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

    # -------- CONFUSION MATRIX VALUES --------
    TN, FP, FN, TP = 8568, 1432, 366, 9634
    total = TN + FP + FN + TP

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

    # -------- CONFUSION MATRIX --------
    st.markdown("<div class='section'>Confusion Matrix</div>", unsafe_allow_html=True)

    cm = np.array([[TN, FP],[FN, TP]])

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        linewidths=1,
        linecolor='white',
        xticklabels=["REAL","AI"],
        yticklabels=["REAL","AI"],
        ax=ax
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
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

    st.latex(r"\hat{p} = \frac{\text{correct}}{n}")
    st.latex(rf"\hat{{p}} = \frac{{{correct}}}{{{n}}} = {p_hat:.4f}")

    # -------- HYPOTHESIS TEST --------
   st.markdown("<div class='section'>Hypothesis Testing</div>", unsafe_allow_html=True)

# counts
n = len(y_true)
errors = (y_true != y_pred).sum()
correct = n - errors

# rates
e_hat = errors / n
accuracy_hat = correct / n
e0 = 0.5

# z test
z = (e_hat - e0) / np.sqrt((e0*(1-e0))/n)
z_critical = norm.ppf(1-0.05/2)
p_value = 2*(1-norm.cdf(abs(z)))

# display
st.write(f"Total Samples: {n}")
st.write(f"Correct Predictions: {correct}")
st.write(f"Misclassified Samples: {errors}")

st.write(f"Observed Error Rate (ê): {e_hat:.4f} ({e_hat*100:.2f}%)")
st.write(f"Observed Accuracy: {accuracy_hat:.4f} ({accuracy_hat*100:.2f}%)")

st.latex(r"H_0: e = 0.5 \quad vs \quad H_1: e \neq 0.5")
st.latex(r"Z = \frac{\hat{e} - e_0}{\sqrt{\frac{e_0(1-e_0)}{n}}}")

st.write(f"Z Statistic: {z:.3f}")
st.write(f"Z Critical: ±{z_critical:.3f}")
st.write(f"P-value: {p_value:.6f}")

if abs(z)>z_critical:
    if z < 0:
        st.success("Reject H0 → Model error is significantly LOWER than random guessing")
    else:
        st.error("Reject H0 → Model error is WORSE than random guessing")
else:
    st.warning("Fail to reject H0 → No significant difference")

    # -------- CLASS-WISE ACCURACY --------
    st.markdown("<div class='section'>Class-wise Accuracy</div>", unsafe_allow_html=True)

    real_acc = (real_df["prediction"]=="REAL").mean()
    ai_acc = (ai_df["prediction"]=="AI").mean()

    st.write(f"Real Accuracy: {real_acc*100:.2f}%")
    st.write(f"AI Accuracy: {ai_acc*100:.2f}%")

    # -------- 2-PROPORTION TEST --------
    st.markdown("<div class='section'>Misclassification Comparison Test</div>", unsafe_allow_html=True)

# sizes
n_real = len(real_df)
n_ai = len(ai_df)

# misclassified counts
x_real = (real_df["prediction"]=="AI").sum()   # real → AI
x_ai = (ai_df["prediction"]=="REAL").sum()     # AI → real

# rates
p_real = x_real / n_real
p_ai = x_ai / n_ai

# pooled
p_pool = (x_real + x_ai) / (n_real + n_ai)
SE = np.sqrt(p_pool*(1-p_pool)*(1/n_real + 1/n_ai))

Z = (p_ai - p_real)/SE
p_val = 2*(1-norm.cdf(abs(Z)))

# display counts
st.write("🔹 Real Images:")
st.write(f"Total Real Images: {n_real}")
st.write(f"Misclassified as AI: {x_real}")

st.write("🔹 AI Images:")
st.write(f"Total AI Images: {n_ai}")
st.write(f"Misclassified as Real: {x_ai}")

# display rates
st.write(f"Real Misclassification Rate: {p_real:.4f} ({p_real*100:.2f}%)")
st.write(f"AI Misclassification Rate: {p_ai:.4f} ({p_ai*100:.2f}%)")

# stats
st.latex(r"H_0: p_{real} = p_{ai}")
st.latex(r"H_1: p_{real} \neq p_{ai}")
st.latex(r"Z = \frac{p_{ai} - p_{real}}{SE}")

st.write(f"Z Statistic: {Z:.3f}")
st.write(f"P-value: {p_val:.6f}")

# decision
if abs(Z) > 1.96:
    st.success("Reject H0 → Significant difference in misclassification rates")
else:
    st.warning("Fail to reject H0 → No significant difference")
