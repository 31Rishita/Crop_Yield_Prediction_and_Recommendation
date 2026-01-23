from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ----------------------------
# LOAD MODELS
# ----------------------------
yield_model = load_model("models/yield/bilstm_attention_yield_model.keras")

x_scaler = pickle.load(open("models/yield/x_scaler.pkl", "rb"))
y_scaler = pickle.load(open("models/yield/y_scaler.pkl", "rb"))
feature_cols = pickle.load(open("models/yield/feature_cols.pkl", "rb"))

reco_model = pickle.load(open("models/reco/crop_reco_model.pkl", "rb"))
reco_scaler = pickle.load(open("models/reco/reco_scaler.pkl", "rb"))
crop_encoder = pickle.load(open("models/reco/crop_encoder.pkl", "rb"))

# ❌ fields to remove ONLY from recommendation UI
REMOVED_RECO_FIELDS = ["production", "fertilizer", "pesticide", "yield"]

# ----------------------------
# DASHBOARD (SINGLE PAGE)
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    yield_prediction = None
    recommendations = None
    state_error = None

    # ================= YIELD PREDICTION =================
    if "predict_yield" in request.form:
        data = {}

        for col in feature_cols:
            if col == "Production":
                data[col] = 0.0  # prevent data leakage
            else:
                data[col] = float(request.form.get(col, 0))

        X = pd.DataFrame([data])[feature_cols]

        # BiLSTM needs time steps
        X_seq = np.repeat(X.values[np.newaxis, :, :], 7, axis=1)
        X_seq = x_scaler.transform(
            X_seq.reshape(-1, X_seq.shape[2])
        ).reshape(X_seq.shape)

        y_pred = yield_model.predict(X_seq)
        yield_prediction = np.expm1(
            y_scaler.inverse_transform(y_pred)
        )[0][0]

    # ================= CROP RECOMMENDATION =================
    if "recommend_crop" in request.form:
        state = request.form.get("state")
        season = request.form.get("season")

        # Restrict states
        if state not in ["Telangana", "Andhra Pradesh"]:
            state_error = "Crop recommendation is available only for Telangana and Andhra Pradesh."
        else:
            reco_features = reco_scaler.feature_names_in_
            data = {}

            for col in reco_features:

                # One-hot encode state
                if col.startswith("state_"):
                    data[col] = 1.0 if col == f"state_{state}" else 0.0

                # One-hot encode season
                elif col.startswith("season_"):
                    data[col] = 1.0 if col == f"season_{season}" else 0.0

                # Removed fields → default 0
                elif col.lower() in REMOVED_RECO_FIELDS:
                    data[col] = 0.0

                # Numeric inputs from form
                else:
                    data[col] = float(request.form.get(col, 0))

            X = pd.DataFrame([data])
            X = reco_scaler.transform(X)

            proba = reco_model.predict_proba(X)[0]
            top3 = np.argsort(-proba)[:3]

            recommendations = []
            for idx in top3:
                crop = crop_encoder.inverse_transform([idx])[0]
                confidence = round(proba[idx] * 100, 2)
                recommendations.append((crop, confidence))

    return render_template(
        "index.html",
        yield_features=feature_cols,
        reco_features=reco_scaler.feature_names_in_,
        yield_prediction=yield_prediction,
        recommendations=recommendations,
        state_error=state_error
    )

if __name__ == "__main__":
    app.run(debug=True)
