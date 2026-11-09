import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score

DATA = Path("data/Crop_recommendation.csv")
PKL  = Path("crop_model.pkl")

df = pd.read_csv(DATA)
features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X, y = df[features].values, df["label"].values

model = pickle.load(open(PKL, "rb"))
classes = list(model.classes_)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)
y_pred = np.array(classes)[proba.argmax(axis=1)]

print(f"Accuracy     : {accuracy_score(y, y_pred):.4f}")
print(f"Top-3 accuracy: {top_k_accuracy_score(y, proba, k=3, labels=classes):.4f}")
print()
print(classification_report(y, y_pred, zero_division=0))
