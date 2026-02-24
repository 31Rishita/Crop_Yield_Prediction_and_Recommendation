import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.layers import (
    Input,
    LSTM,
    Bidirectional,
    Dense,
    Dropout,
    Attention,
    GlobalAveragePooling1D,
    BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from yield_preprocessing_bilstm import prepare_bilstm_data

DATASET = "dataset/Custom_Crops_yield_Historical_Dataset.csv"
TIME_STEPS = 7

X, y, feature_cols = prepare_bilstm_data(
    DATASET,
    TIME_STEPS,
    save_artifacts=True
)


# TRAIN / TEST SPLIT (NO SHUFFLE)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# SCALE DATA (GLOBAL)
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[2])
).reshape(X_train.shape)

X_test = x_scaler.transform(
    X_test.reshape(-1, X_test.shape[2])
).reshape(X_test.shape)

y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_test = y_scaler.transform(y_test.reshape(-1, 1))

# MODEL (BiLSTM + Attention)
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

att = Attention()([x, x])
x = GlobalAveragePooling1D()(att)

x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(1, name="yield_output")(x)

model = Model(inputs, output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="log_cosh"
)

model.summary()

# TRAIN
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

model.fit(
    X_train,
    y_train,
    epochs=120,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# EVALUATION (REAL SCALE)
y_pred = model.predict(X_test)

y_test_real = np.expm1(y_scaler.inverse_transform(y_test))
y_pred_real = np.expm1(y_scaler.inverse_transform(y_pred))

MAE = mean_absolute_error(y_test_real, y_pred_real)
RMSE = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
R2 = r2_score(y_test_real, y_pred_real)
MAPE = np.mean(
    np.abs((y_test_real - y_pred_real) / y_test_real)
) * 100

print("\n🌾 FINAL YIELD METRICS")
print(f"MAE  : {MAE:.2f} kg/ha")
print(f"RMSE : {RMSE:.2f} kg/ha")
print(f"R²   : {R2:.4f}")
print(f"MAPE : {MAPE:.2f}%")

os.makedirs("models/yield", exist_ok=True)

model.save("models/yield/bilstm_attention_yield_model.keras")
pickle.dump(x_scaler, open("models/yield/x_scaler.pkl", "wb"))
pickle.dump(y_scaler, open("models/yield/y_scaler.pkl", "wb"))
pickle.dump(feature_cols, open("models/yield/feature_cols.pkl", "wb"))

print("✅ District-aware optimized Yield Model saved")

import matplotlib.pyplot as plt
# PLOT 1: Actual vs Predicted Yield
plt.figure()
plt.scatter(y_test_real, y_pred_real, alpha=0.6)
plt.xlabel("Actual Yield (kg/ha)")
plt.ylabel("Predicted Yield (kg/ha)")
plt.title("Yield Prediction: Actual vs Predicted")
plt.grid(True)
plt.show()

# PLOT 2: Residual Distribution
residuals = y_test_real - y_pred_real

plt.figure()
plt.hist(residuals, bins=30)
plt.xlabel("Residual (kg/ha)")
plt.ylabel("Frequency")
plt.title("Yield Prediction Residual Distribution")
plt.grid(True)
plt.show()

import pickle

eval_data = {
    "y_test_real": y_test_real.flatten(),
    "y_pred_real": y_pred_real.flatten()
}

pickle.dump(
    eval_data,
    open("models/yield/yield_eval.pkl", "wb")
)

from cache.cache_store import load_cache
import pandas as pd

cache_data = load_cache()

if len(cache_data) > 0:
    df_cache = pd.DataFrame([c["input_features"] for c in cache_data])
    df_cache["yield"] = [c["predicted_yield"] for c in cache_data]

    # Now merge with original dataset
    df_full = pd.concat([original_df, df_cache], ignore_index=True)
