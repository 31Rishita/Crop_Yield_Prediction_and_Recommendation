import pandas as pd
import numpy as np
import pickle


def prepare_bilstm_data(csv_path, time_steps=7, save_artifacts=False):
    df = pd.read_csv(csv_path)

    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("%", "percent")
    )

    df = df[df["State_Name"].isin(["Andhra Pradesh", "Telangana"])]
    df = df.dropna()
    df = df[df["Yield_kg_per_ha"] > 0]

    low = df["Yield_kg_per_ha"].quantile(0.01)
    high = df["Yield_kg_per_ha"].quantile(0.99)
    df = df[(df["Yield_kg_per_ha"] >= low) & (df["Yield_kg_per_ha"] <= high)]

    numeric_features = [
        "Area_ha",
        "N_req_kg_per_ha",
        "P_req_kg_per_ha",
        "K_req_kg_per_ha",
        "Temperature_C",
        "Humidity_percent",
        "pH",
        "Rainfall_mm",
        "Wind_Speed_m_s",
        "Solar_Radiation_MJ_m2_day"
    ]

    # 🔥 Save district mean yield (for frontend use)
    district_mean_dict = (
        df.groupby("Dist_Name")["Yield_kg_per_ha"]
        .mean()
        .apply(np.log1p)
        .to_dict()
    )

    if save_artifacts:
        pickle.dump(
            district_mean_dict,
            open("models/yield/district_yield_mean.pkl", "wb")
        )

    X_seq, y_seq = [], []
    feature_cols = None

    for _, g in df.groupby(["Crop", "Dist_Name"]):
        g = g.sort_values("Year")

        g["Yield_log"] = np.log1p(g["Yield_kg_per_ha"])
        g["Yield_lag1"] = g["Yield_log"].shift(1)
        g["District_Yield_Mean"] = g["Dist_Name"].map(district_mean_dict)

        g = g.dropna()

        y = g["Yield_log"].values.reshape(-1, 1)

        X_num = g[
            numeric_features + ["Yield_lag1", "District_Yield_Mean"]
        ]

        X_cat = pd.get_dummies(g["Dist_Name"], drop_first=True)
        X = pd.concat([X_num, X_cat], axis=1)

        if feature_cols is None:
            feature_cols = X.columns
        else:
            X = X.reindex(columns=feature_cols, fill_value=0)

        for i in range(len(X) - time_steps):
            X_seq.append(X.iloc[i:i+time_steps].values)
            y_seq.append(y[i+time_steps])

    return np.array(X_seq), np.array(y_seq).flatten(), feature_cols
