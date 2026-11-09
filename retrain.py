import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report

DATA  = Path("data/Crop_recommendation.csv")
NEW   = Path("crop_model_new.pkl")
META  = Path("model_metadata.json")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET   = "label"

def load():
    df = pd.read_csv(DATA)
    # ADDED sorted() HERE to fix the top_k_accuracy_score error
    return df[FEATURES].values, df[TARGET].values, sorted(df[TARGET].unique().tolist())

def build_improved():
    return RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=2,
        min_samples_split=4, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=42,
    )

def cv_report(name, model, X, y, classes):
    print(f"\n=== {name} ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
    print(f"5-fold accuracy : {acc.mean():.4f} +/- {acc.std():.4f}")
    proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)
    top3 = top_k_accuracy_score(y, proba, k=3, labels=classes)
    print(f"Top-3 accuracy  : {top3:.4f}")
    max_proba = proba.max(axis=1)
    print(f"Max-proba dist  : mean={max_proba.mean():.3f}  >0.95 share={(max_proba>0.95).mean():.2%}")
    return {
        "cv_accuracy_mean": float(acc.mean()), "cv_accuracy_std": float(acc.std()),
        "top3_accuracy": float(top3), "max_proba_mean": float(max_proba.mean()),
        "high_conf_share": float((max_proba > 0.95).mean())
    }

if __name__ == "__main__":
    X, y, classes = load()
    print(f"Loaded {len(X)} samples, {len(classes)} classes")
    base_meta = cv_report("Baseline", RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42), X, y, classes)
    imp_meta = cv_report("Improved", build_improved(), X, y, classes)
    print("\n=== Fitting calibrated model ===")
    calibrated = CalibratedClassifierCV(estimator=build_improved(), method="sigmoid", cv=5, n_jobs=-1)
    calibrated.fit(X, y)
    
    sample = np.array([[90, 42, 43, 20.9, 82.0, 6.5, 202.9]])
    proba  = calibrated.predict_proba(sample)[0]
    top3   = np.argsort(proba)[::-1][:3]
    print("Sample probabilities for rice-like input:")
    for i in top3:
        print(f"  {calibrated.classes_[i]:12s}  {proba[i]:.4f}")
        
    with open(NEW, "wb") as f:
        pickle.dump(calibrated, f)
    print(f"\nWrote {NEW}")
    metadata = {"n_samples": int(len(X)), "n_classes": int(len(classes)), "classes": classes, "features": FEATURES, "baseline": base_meta, "improved": imp_meta}
    META.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {META}")
