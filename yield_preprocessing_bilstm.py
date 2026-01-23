import pandas as pd
import numpy as np

def prepare_bilstm_data(csv_path, time_steps=7):
    df = pd.read_csv(csv_path)

    df = df.dropna()
    df = df[df["Yield"] > 0]

    lower = df["Yield"].quantile(0.01)
    upper = df["Yield"].quantile(0.99)
    df = df[(df["Yield"] >= lower) & (df["Yield"] <= upper)]

    X_seq, y_seq = [], []
    feature_cols = None

    for _, g in df.groupby(["Crop", "State"]):
        g = g.sort_values("Crop_Year")

        g["Yield"] = np.log1p(g["Yield"])
        y = g["Yield"].values.reshape(-1, 1)

        X = g.drop(columns=["Yield", "Crop_Year"])
        X = pd.get_dummies(X, drop_first=True)

        if feature_cols is None:
            feature_cols = X.columns
        else:
            X = X.reindex(columns=feature_cols, fill_value=0)

        for i in range(len(X) - time_steps):
            X_seq.append(X.iloc[i:i+time_steps].values)
            y_seq.append(y[i + time_steps])

    return np.array(X_seq), np.array(y_seq).flatten(), feature_cols
