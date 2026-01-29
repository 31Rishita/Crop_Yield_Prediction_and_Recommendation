import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Smart Crop Advisory System",
    page_icon="🌱",
    layout="wide"
)

st.title("🌾 Smart Crop Advisory System")
st.markdown(
    "AI-powered **Crop Recommendation & Yield Prediction** "
    "for Andhra Pradesh and Telangana"
)

# ---------------------------------
# STATIC DROPDOWNS
# ---------------------------------
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

# ---------------------------------
# CACHE: LOAD MODELS & ARTIFACTS
# ---------------------------------
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

# ---------------------------------
# SIDEBAR INPUTS
# ---------------------------------
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
    min_value=2020,
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

# ---------------------------------
# CACHE-ENABLED PREDICTION FUNCTIONS
# ---------------------------------
@st.cache_data(show_spinner=False)
def predict_yield_cached(base_inputs, district_name, crop_name, year):
    TIME_STEPS = 7  # must match training

    # Get district mean (log scale)
    district_mean = district_yield_mean.get(
        district_name,
        np.mean(list(district_yield_mean.values()))
    )

    # Approximate previous year yield
    yield_lag1 = district_mean

    # Final feature vector (12 features)
    full_input = base_inputs + [yield_lag1, district_mean]

    # 🔁 Repeat for time steps
    X_single = np.array(full_input).reshape(1, 1, -1)
    X_seq = np.repeat(X_single, TIME_STEPS, axis=1)

    # Scale
    X_scaled = yield_x_scaler.transform(
        X_seq.reshape(-1, X_seq.shape[2])
    ).reshape(X_seq.shape)

    # Predict
    y_scaled = yield_model.predict(X_scaled)
    y = np.expm1(yield_y_scaler.inverse_transform(y_scaled))

    return float(y[0][0])



@st.cache_data(show_spinner=False)
def recommend_crop_cached(base_inputs):
    SEQUENCE_LENGTH = 6  # must match training

    # Build single-step input
    X_single = np.array(base_inputs).reshape(1, 1, -1)

    # 🔁 Repeat to required sequence length
    X_seq = np.repeat(X_single, SEQUENCE_LENGTH, axis=1)

    # Scale properly
    X_scaled = reco_scaler.transform(
        X_seq.reshape(-1, X_seq.shape[2])
    ).reshape(X_seq.shape)

    # Predict
    probs = reco_model.predict(X_scaled)[0]
    top3_idx = np.argsort(-probs)[:3]
    crops = reco_encoder.inverse_transform(top3_idx)

    return crops.tolist()

# ---------------------------------
# PREDICTION BUTTON
# ---------------------------------
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

    predicted_yield = predict_yield_cached(
        base_features,
        district,
        crop,
        year
    )

    top_crops = recommend_crop_cached(base_features)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌾 Yield Prediction")
        st.write(f"**Crop:** {crop}")
        st.write(f"**District:** {district}")
        st.metric(
            "Predicted Yield",
            f"{predicted_yield:.2f} kg/ha"
        )

    with col2:
        st.markdown("### 🌱 Recommended Crops (Top-3)")
        for i, c in enumerate(top_crops, 1):
            st.write(f"**{i}. {c}**")

    st.success("Prediction generated using cached intelligence 🚀")
