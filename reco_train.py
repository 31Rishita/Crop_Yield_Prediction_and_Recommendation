from reco_preprocessing import prepare_reco_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pickle, os

DATASET = "dataset/crop_data.csv"

X, y = prepare_reco_data(DATASET)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=800,
    max_depth=25,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\n🌱 TOP-1 ACCURACY:", accuracy_score(y_test, y_pred))

proba = model.predict_proba(X_test)
top3 = np.argsort(-proba, axis=1)[:, :3]
top3_acc = np.mean([y_test[i] in top3[i] for i in range(len(y_test))])
print("🔥 TOP-3 ACCURACY:", top3_acc)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

os.makedirs("models/reco", exist_ok=True)
pickle.dump(model, open("models/reco/crop_reco_model.pkl", "wb"))

print("✅ Recommendation model saved")
