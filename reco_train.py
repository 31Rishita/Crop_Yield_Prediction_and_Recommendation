from reco_preprocessing import prepare_reco_data
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Bidirectional,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

DATASET = "dataset\Custom_Crops_yield_Historical_Dataset.csv"
SEQUENCE_LENGTH = 6
# 📦 Load data
X, y = prepare_reco_data(DATASET, SEQUENCE_LENGTH)

num_classes = len(np.unique(y))
y_cat = to_categorical(y, num_classes)

# ⛔ Time-series split (NO shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_cat,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

model = Sequential([
    Bidirectional(
        LSTM(128, return_sequences=True),
        input_shape=X_train.shape[1:]
    ),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=7,
    restore_best_weights=True
)

lr_reduce = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1
)

# 🚀 Train
history = model.fit(
    X_train,
    y_train,
    epochs=60,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, lr_reduce],
    verbose=1
)

# 📊 Evaluation
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# 🔥 TOP-2 Accuracy
top2 = np.argsort(-y_pred_prob, axis=1)[:, :2]
top2_acc = np.mean([y_true[i] in top2[i] for i in range(len(y_true))])
print("\n🔥 TOP-2 ACCURACY:", top2_acc)

print("\n📋 Classification Report\n")
print(classification_report(y_true, y_pred))

# 💾 Save model
os.makedirs("models/reco", exist_ok=True)
model.save("models/reco/crop_reco_bilstm_model.h5")

print("✅ BiLSTM Crop Recommendation Model Saved")


import matplotlib.pyplot as plt

# PLOT 3: Training vs Validation Accuracy
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Crop Recommendation Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Crop Recommendation Confusion Matrix")
plt.show()

import pickle
eval_data = {
    "y_true": y_true,
    "y_pred": y_pred,
    "train_acc": history.history["accuracy"],
    "val_acc": history.history["val_accuracy"]
}

pickle.dump(
    eval_data,
    open("models/reco/reco_eval.pkl", "wb")
)

from cache.cache_store import load_cache
import pandas as pd

cache_data = load_cache()
original_df = pd.read_csv(DATASET)
if len(cache_data) > 0:
    df_cache = pd.DataFrame([c["input_features"] for c in cache_data])
    df_cache["yield"] = [c["predicted_yield"] for c in cache_data]

    # Now merge with original dataset
    df_full = pd.concat([original_df, df_cache], ignore_index=True)
