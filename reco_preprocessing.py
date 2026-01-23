import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle, os

def prepare_reco_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna()

    y_enc = LabelEncoder()
    y = y_enc.fit_transform(df["crop"])

    X = df.drop(columns=["crop"])
    X = pd.get_dummies(X, drop_first=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    os.makedirs("models/reco", exist_ok=True)
    pickle.dump(scaler, open("models/reco/reco_scaler.pkl", "wb"))
    pickle.dump(y_enc, open("models/reco/crop_encoder.pkl", "wb"))

    return X, y
