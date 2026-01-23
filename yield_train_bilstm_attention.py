import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    Attention, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from yield_preprocessing_bilstm import prepare_bilstm_data

# -----------------------------
# LOAD DATA
# -----------------------------
DATASET = "dataset/crop_yield.csv"
TIME_STEPS = 7

X, y, feature_cols = prepare_bilstm_data(DATASET, TIME_STEPS)

# -----------------------------
# SPLIT (NO SHUFFLING)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# -----------------------------
# SCALE (TRAIN ONLY)
# -----------------------------
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_train = x_scaler.fit_transform(
    X_train.reshape(-1, X_train.shape[2])
).reshape(X_train.shape)

X_test = x_scaler.transform(
    X_test.reshape(-1, X_test.shape[2])
).reshape(X_test.shape)

y_train = y_scaler.fit_transform(y_train.reshape(-1,1))
y_test = y_scaler.transform(y_test.reshape(-1,1))

# -----------------------------
# MODEL
# -----------------------------
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
x = Dropout(0.3)(x)

x = Bidirectional(LSTM(64, return_sequences=True))(x)
x = Dropout(0.3)(x)

att = Attention()([x, x])
x = GlobalAveragePooling1D()(att)

output = Dense(1)(x)

model = Model(inputs, output)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="huber"
)

# -----------------------------
# TRAIN
# -----------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

lr_sched = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_sched],
    verbose=1
)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

y_test_real = np.expm1(y_scaler.inverse_transform(y_test))
y_pred_real = np.expm1(y_scaler.inverse_transform(y_pred))

MAE = mean_absolute_error(y_test_real, y_pred_real)
RMSE = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
R2 = r2_score(y_test_real, y_pred_real)
MAPE = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

print("\n🌾 FINAL YIELD METRICS")
print(f"MAE  : {MAE:.4f}")
print(f"RMSE : {RMSE:.4f}")
print(f"R²   : {R2:.4f}")
print(f"MAPE : {MAPE:.2f}%")

# -----------------------------
# SAVE
# -----------------------------
os.makedirs("models/yield", exist_ok=True)

model.save("models/yield/bilstm_attention_yield_model.keras")
pickle.dump(x_scaler, open("models/yield/x_scaler.pkl", "wb"))
pickle.dump(y_scaler, open("models/yield/y_scaler.pkl", "wb"))
pickle.dump(feature_cols, open("models/yield/feature_cols.pkl", "wb"))

print("✅ Yield model saved")
