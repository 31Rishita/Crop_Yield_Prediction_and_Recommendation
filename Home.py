import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from cache.cache_store import save_cache

# PAGE CONFIG
st.set_page_config(
    page_title="Smart Crop Advisory System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌾Crop Yield Prediction and Crop Recommendation")
st.markdown(
    " **Crop Yield Prediction & Crop Recommendation** "
    "for Andhra Pradesh and Telangana"
)

# STATIC DROPDOWNS
CROPS = ["Rice", "Maize", "Chickpea", "Cotton"]

AP_DISTRICTS = [
    "Srikakulam", "Visakhapatnam", "East Godavari", "West Godavari",
    "Krishna", "Guntur", "S.P.S. Nellore", "Kurnool",
    "Ananthapur", "Kadapa YSR", "Chittoor"
]

TG_DISTRICTS = [
    "Hyderabad", "Nizamabad", "Medak", "Mahabubnagar",
    "Nalgonda", "Warangal", "Khammam",
    "Karimnagar", "Adilabad"
]

# CACHE: LOAD MODELS & ARTIFACTS
@st.cache_resource
def load_yield_model():
    model = tf.keras.models.load_model(
        "models/yield/bilstm_attention_yield_model.keras"
    )
    x_scaler = pickle.load(open("models/yield/x_scaler.pkl", "rb"))
    y_scaler = pickle.load(open("models/yield/y_scaler.pkl", "rb"))
    district_mean = pickle.load(
        open("models/yield/district_yield_mean.pkl", "rb")
    )
    return model, x_scaler, y_scaler, district_mean


@st.cache_resource
def load_reco_model():
    model = tf.keras.models.load_model(
        "models/reco/crop_reco_bilstm_model.h5"
    )
    scaler = pickle.load(open("models/reco/reco_scaler.pkl", "rb"))
    encoder = pickle.load(open("models/reco/crop_encoder.pkl", "rb"))
    return model, scaler, encoder


yield_model, yield_x_scaler, yield_y_scaler, district_yield_mean = load_yield_model()
reco_model, reco_scaler, reco_encoder = load_reco_model()

# SIDEBAR INPUTS
st.sidebar.header("🌍 Location")

state = st.sidebar.selectbox(
    "State",
    ["Andhra Pradesh", "Telangana"]
)

district = st.sidebar.selectbox(
    "District",
    AP_DISTRICTS if state == "Andhra Pradesh" else TG_DISTRICTS
)

crop = st.sidebar.selectbox(
    "Crop",
    CROPS
)

year = st.sidebar.number_input(
    "Year",
    min_value=2000,
    max_value=2030,
    value=2024
)

st.sidebar.header("🌦 Climate & Soil Inputs")

area = st.sidebar.number_input("Area (ha)", 0.1, 1000.0, 1.0)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 3000.0, 800.0)
temperature = st.sidebar.number_input("Temperature (°C)", 10.0, 45.0, 30.0)
humidity = st.sidebar.number_input("Humidity (%)", 10.0, 100.0, 60.0)
ph = st.sidebar.number_input("Soil pH", 3.0, 9.0, 6.5)

n_req = st.sidebar.number_input("Nitrogen (kg/ha)", 0.0, 300.0, 120.0)
p_req = st.sidebar.number_input("Phosphorus (kg/ha)", 0.0, 200.0, 60.0)
k_req = st.sidebar.number_input("Potassium (kg/ha)", 0.0, 200.0, 40.0)

wind = st.sidebar.number_input("Wind Speed (m/s)", 0.0, 20.0, 3.0)
solar = st.sidebar.number_input(
    "Solar Radiation (MJ/m²/day)", 0.0, 30.0, 18.0
)

# ===================== RECURSIVE YIELD PREDICTION =====================
def predict_yield_recursive(
    base_inputs,
    district_name,
    start_year,
    target_year
):
    TIME_STEPS = 7

    current_yield = district_yield_mean.get(
        district_name,
        np.mean(list(district_yield_mean.values()))
    )

    district_mean = current_yield

    for _ in range(start_year + 1, target_year + 1):

        full_input = base_inputs + [current_yield, district_mean]

        X_single = np.array(full_input).reshape(1, 1, -1)
        X_seq = np.repeat(X_single, TIME_STEPS, axis=1)

        X_scaled = yield_x_scaler.transform(
            X_seq.reshape(-1, X_seq.shape[2])
        ).reshape(X_seq.shape)

        y_scaled = yield_model.predict(X_scaled, verbose=0)
        y_pred = np.expm1(
            yield_y_scaler.inverse_transform(y_scaled)
        )[0][0]

        current_yield = float(y_pred)

    return current_yield
# ====================================================================


@st.cache_data(show_spinner=False)
def recommend_crop_cached(base_inputs):
    SEQUENCE_LENGTH = 6

    X_single = np.array(base_inputs).reshape(1, 1, -1)
    X_seq = np.repeat(X_single, SEQUENCE_LENGTH, axis=1)

    X_scaled = reco_scaler.transform(
        X_seq.reshape(-1, X_seq.shape[2])
    ).reshape(X_seq.shape)

    probs = reco_model.predict(X_scaled)[0]
    top3_idx = np.argsort(-probs)[:3]
    crops = reco_encoder.inverse_transform(top3_idx)

    return crops.tolist()

# ===================== BUTTON =====================
if st.button("🚀 Generate Recommendation"):
    st.subheader("📊 Prediction Results")

    base_features = [
        area,
        n_req,
        p_req,
        k_req,
        temperature,
        humidity,
        ph,
        rainfall,
        wind,
        solar
    ]

    DATASET_LAST_YEAR = 2000

    predicted_yield = predict_yield_recursive(
        base_features,
        district_name=district,
        start_year=DATASET_LAST_YEAR,
        target_year=year
    )

    top_crops = recommend_crop_cached(base_features)

    input_data = {
        "state": state,
        "district": district,
        "crop": crop,
        "year": year,
        "area": area,
        "rainfall": rainfall,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "nitrogen": n_req,
        "phosphorus": p_req,
        "potassium": k_req,
        "wind_speed": wind,
        "solar_radiation": solar
    }

    save_cache(
        input_data=input_data,
        yield_pred=predicted_yield,
        crops=top_crops
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌾 Yield Prediction")
        st.metric("Predicted Yield", f"{predicted_yield:.2f} kg/ha")

    with col2:
        st.markdown("### 🌱 Recommended Crops (Top-3)")
        for i, c in enumerate(top_crops, 1):
            st.write(f"**{i}. {c}**")

    st.success("Recursive year-wise prediction completed & cached 🚀")
