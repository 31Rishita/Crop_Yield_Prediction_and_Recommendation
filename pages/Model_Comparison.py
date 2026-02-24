import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# PAGE CONFIG
st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Yield Prediction Model Comparison")
st.markdown(
    "Comparison between **Random Forest** and **BiLSTM** models"
)

# LOAD EVALUATION FILES
@st.cache_resource
def load_eval():
    rf_path = "models/yield/yield_rf_eval.pkl"
    lstm_path = "models/yield/yield_eval.pkl"

    if not os.path.exists(rf_path):
        st.error("❌ Random Forest evaluation file not found")
        st.stop()

    if not os.path.exists(lstm_path):
        st.error("❌ BiLSTM evaluation file not found")
        st.stop()

    rf = pickle.load(open(rf_path, "rb"))
    lstm = pickle.load(open(lstm_path, "rb"))

    return rf, lstm

rf_eval, lstm_eval = load_eval()

# EXTRACT DATA
rf_y_true = np.array(rf_eval["y_test_real"]).flatten()
rf_y_pred = np.array(rf_eval["y_pred_real"]).flatten()

lstm_y_true = np.array(lstm_eval["y_test_real"]).flatten()
lstm_y_pred = np.array(lstm_eval["y_pred_real"]).flatten()

# METRICS FUNCTION
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


rf_mae, rf_rmse, rf_r2 = metrics(rf_y_true, rf_y_pred)
lstm_mae, lstm_rmse, lstm_r2 = metrics(lstm_y_true, lstm_y_pred)

# METRICS DISPLAY
st.subheader("📌 Quantitative Comparison")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🌳 Random Forest")
    st.metric("MAE (kg/ha)", f"{rf_mae:.2f}")
    st.metric("RMSE (kg/ha)", f"{rf_rmse:.2f}")
    st.metric("R²", f"{rf_r2:.4f}")

with col2:
    st.markdown("### 🤖 BiLSTM (District-aware)")
    st.metric("MAE (kg/ha)", f"{lstm_mae:.2f}")
    st.metric("RMSE (kg/ha)", f"{lstm_rmse:.2f}")
    st.metric("R²", f"{lstm_r2:.4f}")

st.markdown("---")

# GRAPHICAL COMPARISON
st.subheader("📈 Actual vs Predicted Yield")

g1, g2 = st.columns(2)

# -------- Random Forest Plot --------
with g1:
    st.markdown("### 🌳 Random Forest")
    fig1, ax1 = plt.subplots()
    ax1.scatter(rf_y_true, rf_y_pred, alpha=0.5)
    ax1.plot(
        [rf_y_true.min(), rf_y_true.max()],
        [rf_y_true.min(), rf_y_true.max()],
        linestyle="--"
    )
    ax1.set_xlabel("Actual Yield (kg/ha)")
    ax1.set_ylabel("Predicted Yield (kg/ha)")
    ax1.set_title("Random Forest: Actual vs Predicted")
    ax1.grid(True)
    st.pyplot(fig1)

# -------- BiLSTM Plot --------
with g2:
    st.markdown("### 🤖 BiLSTM")
    fig2, ax2 = plt.subplots()
    ax2.scatter(lstm_y_true, lstm_y_pred, alpha=0.5, color="green")
    ax2.plot(
        [lstm_y_true.min(), lstm_y_true.max()],
        [lstm_y_true.min(), lstm_y_true.max()],
        linestyle="--",
        color="black"
    )
    ax2.set_xlabel("Actual Yield (kg/ha)")
    ax2.set_ylabel("Predicted Yield (kg/ha)")
    ax2.set_title("BiLSTM: Actual vs Predicted")
    ax2.grid(True)
    st.pyplot(fig2)

st.markdown("---")

# FINAL CONCLUSION
if lstm_rmse < rf_rmse:
    st.success(
        "✅ **BiLSTM outperforms Random Forest** by capturing temporal and district-level patterns."
    )
else:
    st.warning(
        "⚠️ Random Forest performs competitively, but lacks temporal modeling capability."
    )
