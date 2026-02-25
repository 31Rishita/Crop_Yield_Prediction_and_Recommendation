from reco_preprocessing import prepare_reco_data
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

DATASET = "dataset\Custom_Crops_yield_Historical_Dataset.csv"
SEQUENCE_LENGTH = 6

# 📦 Load data (same as BiLSTM)
X, y = prepare_reco_data(DATASET, SEQUENCE_LENGTH)

# 🔄 Flatten sequences for Random Forest
# (RF cannot take 3D input like LSTM)
X = X.reshape(X.shape[0], -1)

# ⛔ Time-series split (NO shuffle)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=False
)

# 🌲 Random Forest Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=48,
    n_jobs=-1
)

# 🚀 Train
rf_model.fit(X_train, y_train)

# 📊 Evaluation
y_pred = rf_model.predict(X_test)
y_pred_prob = rf_model.predict_proba(X_test)

# 🔥 TOP-2 Accuracy
top2 = np.argsort(-y_pred_prob, axis=1)[:, :2]
top2_acc = np.mean([y_test[i] in top2[i] for i in range(len(y_test))])
print("\n🔥 TOP-2 ACCURACY (Random Forest):", top2_acc)

print("\n📋 Classification Report (Random Forest)\n")
print(classification_report(y_test, y_pred))

# 💾 Save model
os.makedirs("models/reco", exist_ok=True)
pickle.dump(
    rf_model,
    open("models/reco/crop_reco_random_forest.pkl", "wb")
)

print("✅ Random Forest Crop Recommendation Model Saved")

# 📦 Save evaluation
eval_data = {
    "y_true": y_test,
    "y_pred": y_pred
}

pickle.dump(
    eval_data,
    open("models/reco/reco_rf_eval.pkl", "wb")
)