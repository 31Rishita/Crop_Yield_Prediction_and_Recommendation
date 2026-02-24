import pandas as pd
import numpy as np
import os, pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def prepare_reco_data(
    csv_path,
    sequence_length=6
):
    df = pd.read_csv(csv_path)
    df = df.dropna()
    df = df[df["State Name"].isin(["Andhra Pradesh", "Telangana"])]

    df = df.sort_values(
        by=["State Name", "Dist Name", "Crop", "Year"]
    )

    # 🎯 Target
    crop_encoder = LabelEncoder()
    df["crop_encoded"] = crop_encoder.fit_transform(df["Crop"])

    feature_cols = [
        "Area_ha",
        "N_req_kg_per_ha",
        "P_req_kg_per_ha",
        "K_req_kg_per_ha",
        "Temperature_C",
        "Humidity_%",
        "pH",
        "Rainfall_mm",
        "Wind_Speed_m_s",
        "Solar_Radiation_MJ_m2_day"
    ]

    X_raw = df[feature_cols].values
    y_raw = df["crop_encoded"].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    os.makedirs("models/reco", exist_ok=True)
    pickle.dump(scaler, open("models/reco/reco_scaler.pkl", "wb"))
    pickle.dump(crop_encoder, open("models/reco/crop_encoder.pkl", "wb"))

    # 🔁 Build sequences
    X_seq, y_seq = [], []

    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y_raw[i])

    return np.array(X_seq), np.array(y_seq)
