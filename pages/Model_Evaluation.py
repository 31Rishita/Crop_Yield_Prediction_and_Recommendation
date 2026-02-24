import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# PAGE CONFIG
st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("Model Evaluation Dashboard")
st.markdown(
    "Performance analysis of **Yield Prediction** and "
    "**Crop Recommendation** models"
)

# LOAD EVALUATION DATA
@st.cache_resource
def load_eval_data():
    yield_eval = pickle.load(
        open("models/yield/yield_eval.pkl", "rb")
    )
    reco_eval = pickle.load(
        open("models/reco/reco_eval.pkl", "rb")
    )
    return yield_eval, reco_eval


yield_eval, reco_eval = load_eval_data()

# LAYOUT
col1, col2 = st.columns(2)

# ========= LEFT: YIELD =========
with col1:
    st.subheader("🌾 Yield Prediction Performance")

    # Actual vs Predicted
    st.markdown("**Actual vs Predicted Yield**")
    fig1, ax1 = plt.subplots()
    ax1.scatter(
        yield_eval["y_test_real"],
        yield_eval["y_pred_real"],
        alpha=0.5
    )
    ax1.set_xlabel("Actual Yield (kg/ha)")
    ax1.set_ylabel("Predicted Yield (kg/ha)")
    ax1.set_title("Actual vs Predicted Yield")
    ax1.grid(True)
    st.pyplot(fig1)

    # Residuals
    residuals = (
        yield_eval["y_test_real"]
        - yield_eval["y_pred_real"]
    )

    st.markdown("**Residual Distribution**")
    fig2, ax2 = plt.subplots()
    ax2.hist(residuals, bins=30)
    ax2.set_xlabel("Residual (kg/ha)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Yield Residuals")
    ax2.grid(True)
    st.pyplot(fig2)

# ========= RIGHT: RECO =========
with col2:
    st.subheader("🌱 Crop Recommendation Performance")

    # Accuracy curve
    st.markdown("**Training vs Validation Accuracy**")
    fig3, ax3 = plt.subplots()
    ax3.plot(reco_eval["train_acc"], label="Training Accuracy")
    ax3.plot(reco_eval["val_acc"], label="Validation Accuracy")
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy")
    ax3.set_title("Accuracy Curve")
    ax3.legend()
    ax3.grid(True)
    st.pyplot(fig3)

    # Confusion matrix
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(
        reco_eval["y_true"],
        reco_eval["y_pred"]
    )

    fig4, ax4 = plt.subplots()
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(ax=ax4)
    ax4.set_title("Crop Recommendation Confusion Matrix")
    st.pyplot(fig4)

# FOOTER
st.markdown("---")
st.caption(
    "Model Evaluation | Crop Yield Prediction and Crop Recommendation"
)
