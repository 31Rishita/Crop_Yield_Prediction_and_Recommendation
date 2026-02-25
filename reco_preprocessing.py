import pandas as pd
import numpy as np
import os, pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def prepare_reco_data(
    csv_path,
    sequence_length=6
):
    print("📂 Loading dataset...")
    df = pd.read_csv(csv_path)

    print("Initial Dataset Shape:", df.shape)

    # -----------------------------
    # Store original stats
    # -----------------------------
    original_rows = df.shape[0]
    original_missing = df.isnull().sum().sum()

    # -----------------------------
    # Cleaning
    # -----------------------------
    print("\n🧹 Dropping missing values...")
    df = df.dropna()
    print("Shape after dropping NA:", df.shape)

    print("\n🔎 Filtering for Andhra Pradesh and Telangana...")
    df = df[df["State Name"].isin(["Andhra Pradesh", "Telangana"])]
    print("Shape after state filtering:", df.shape)

    # -----------------------------
    # Sorting
    # -----------------------------
    print("\n🔄 Sorting dataset by State, District, Crop, Year...")
    df = df.sort_values(
        by=["State Name", "Dist Name", "Crop", "Year"]
    )

    # 🎯 Target
    print("\n🎯 Encoding Crop as Target Variable...")
    crop_encoder = LabelEncoder()
    df["crop_encoded"] = crop_encoder.fit_transform(df["Crop"])

    print("Total Unique Crops:", len(crop_encoder.classes_))
    print("Encoded Crop Classes:", list(crop_encoder.classes_))

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

    print("\n📈 Selected Feature Columns:")
    print(feature_cols)

    X_raw = df[feature_cols].values
    y_raw = df["crop_encoded"].values

    # -----------------------------
    # Feature Scaling
    # -----------------------------
    print("\n⚖ Applying MinMax Scaling to Features...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    print("Feature Scaling Completed.")

    # -----------------------------
    # Save Artifacts
    # -----------------------------
    print("\n💾 Saving Scaler and Crop Encoder...")
    os.makedirs("models/reco", exist_ok=True)
    pickle.dump(scaler, open("models/reco/reco_scaler.pkl", "wb"))
    pickle.dump(crop_encoder, open("models/reco/crop_encoder.pkl", "wb"))
    print("Artifacts Saved Successfully.")

    # 🔁 Build sequences
    print("\n🔁 Creating Sequences for Recommendation Model...")
    X_seq, y_seq = [], []

    for i in range(sequence_length, len(X_scaled)):
        X_seq.append(X_scaled[i-sequence_length:i])
        y_seq.append(y_raw[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    print("Total Sequences Created:", len(X_seq))

    print("\n📐 Final Output Shapes:")
    print("X_seq Shape:", X_seq.shape)
    print("y_seq Shape:", y_seq.shape)
    print("Time Steps:", sequence_length)
    print("Features per Time Step:", len(feature_cols))

    print("\n✅ Recommendation Data Preparation Completed Successfully!")

    return X_seq, y_seq


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    csv_file_path = r"dataset/Custom_Crops_yield_Historical_Dataset.csv"

    X, y = prepare_reco_data(
        csv_path=csv_file_path,
        sequence_length=6
    )

    print("\n🎯 Returned Values:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)