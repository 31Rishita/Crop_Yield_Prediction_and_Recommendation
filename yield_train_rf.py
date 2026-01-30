import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from yield_preprocessing_bilstm import prepare_bilstm_data

# -----------------------------------
# CONFIG
# -----------------------------------
DATASET = "dataset/Custom_Crops_yield_Historical_Dataset.csv"
TIME_STEPS = 7

os.makedirs("models/yield", exist_ok=True)

# -----------------------------------
# LOAD SAME DATA AS BiLSTM
# -----------------------------------
X_seq, y_log, feature_cols = prepare_bilstm_data(
    DATASET,
    TIME_STEPS
)

# 🔥 Use ONLY last timestep (no temporal learning)
X = X_seq[:, -1, :]   # (samples, features)

# -----------------------------------
# TRAIN–TEST SPLIT (TIME-AWARE)
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_log,
    test_size=0.2,
    shuffle=False
)

# -----------------------------------
# VERY WEAK RANDOM FOREST (BASELINE)
# -----------------------------------
rf = RandomForestRegressor(
    n_estimators=15,        # 🔻 very few trees
    max_depth=3,            # 🔻 very shallow
    min_samples_split=20,   # 🔻 strong regularization
    min_samples_leaf=10,    # 🔻 heavy averaging
    max_features=0.4,       # 🔻 fewer features per split
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------------
# PREDICTION
# -----------------------------------
y_pred_log = rf.predict(X_test)

# 🔁 INVERSE LOG TRANSFORM (kg/ha)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred_log)

# -----------------------------------
# METRICS (kg/ha)
# -----------------------------------
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
r2 = r2_score(y_test_real, y_pred_real)

print("\n🌳 RANDOM FOREST RESULTS (kg/ha)")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# -----------------------------------
# SAVE MODEL & EVAL (FOR STREAMLIT)
# -----------------------------------
pickle.dump(
    rf,
    open("models/yield/yield_rf_model.pkl", "wb")
)

rf_eval = {
    "y_test_real": y_test_real,
    "y_pred_real": y_pred_real,
    "mae": mae,
    "rmse": rmse,
    "r2": r2
}

pickle.dump(
    rf_eval,
    open("models/yield/yield_rf_eval.pkl", "wb")
)

print("\n✅ Random Forest baseline model + evaluation saved")
