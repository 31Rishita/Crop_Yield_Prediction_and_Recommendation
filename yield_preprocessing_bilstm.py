import pandas as pd
import numpy as np
import pickle


def prepare_bilstm_data(csv_path, time_steps=7, save_artifacts=False):
    
    print("📂 Loading dataset...")
    original_df = pd.read_csv(csv_path)
    df = original_df.copy()

    print("Initial Dataset Shape:", df.shape)

    # -------------------------------
    # STORE ORIGINAL STATS
    # -------------------------------
    original_rows = df.shape[0]
    original_missing = df.isnull().sum().sum()

    # Clean column names
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(" ", "_")
        .str.replace("%", "percent")
    )

    # Filter states
    df = df[df["State_Name"].isin(["Andhra Pradesh", "Telangana"])]

    # Drop missing values
    df = df.dropna()

    # Remove zero/negative yield
    df = df[df["Yield_kg_per_ha"] > 0]

    # Remove outliers
    low = df["Yield_kg_per_ha"].quantile(0.01)
    high = df["Yield_kg_per_ha"].quantile(0.99)
    df = df[(df["Yield_kg_per_ha"] >= low) & (df["Yield_kg_per_ha"] <= high)]

    # -------------------------------
    # STORE CLEANED STATS
    # -------------------------------
    cleaned_rows = df.shape[0]
    cleaned_missing = df.isnull().sum().sum()

        # -------------------------------
    # 📊 HISTOGRAM BEFORE & AFTER LOG TRANSFORMATION
    # -------------------------------
    import matplotlib.pyplot as plt

    # Before Log Transformation
    plt.figure()
    plt.hist(df["Yield_kg_per_ha"], bins=30)
    plt.title("Yield Distribution (Before Log Transformation)")
    plt.xlabel("Yield (kg/ha)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Apply Log Transformation (Temporary for Visualization)
    yield_log_temp = np.log1p(df["Yield_kg_per_ha"])

    # After Log Transformation
    plt.figure()
    plt.hist(yield_log_temp, bins=30)
    plt.title("Yield Distribution (After Log Transformation)")
    plt.xlabel("Log(Yield)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # -------------------------------
    # 📊 COMPARISON TABLE
    # -------------------------------
    comparison_table = pd.DataFrame({
        "Metric": [
            "Total Rows",
            "Total Columns",
            "Missing Values",
            "Min Yield (kg/ha)",
            "Max Yield (kg/ha)"
        ],
        "Original Dataset": [
            original_rows,
            original_df.shape[1],
            original_missing,
            original_df["Yield_kg_per_ha"].min(),
            original_df["Yield_kg_per_ha"].max()
        ],
        "After Preprocessing": [
            cleaned_rows,
            df.shape[1],
            cleaned_missing,
            df["Yield_kg_per_ha"].min(),
            df["Yield_kg_per_ha"].max()
        ]
    })

    print("\n📊 DATASET COMPARISON TABLE")
    print(comparison_table.to_string(index=False))

    print("\n🔎 Sample Original Data (First 5 Rows)")
    print(original_df.head().to_string(index=False))

    print("\n🔎 Sample After Preprocessing (First 5 Rows)")
    print(df.head().to_string(index=False))

    # Continue your sequence logic (unchanged)
    X_seq, y_seq = [], []
    feature_cols = None

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

    district_mean_dict = (
        df.groupby("Dist_Name")["Yield_kg_per_ha"]
        .mean()
        .apply(np.log1p)
        .to_dict()
    )

    for (crop, dist), g in df.groupby(["Crop", "Dist_Name"]):
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

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq).flatten()

    return X_seq, y_seq, feature_cols


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":

    csv_file_path = r"dataset/Custom_Crops_yield_Historical_Dataset.csv"

    X, y, features = prepare_bilstm_data(
        csv_path=csv_file_path,
        time_steps=7,
        save_artifacts=False
    )

    print("\n🎯 Returned Values:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Number of features:", len(features))