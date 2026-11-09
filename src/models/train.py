import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data
try:
    df = pd.read_csv("data/raw/Crop_recommendation.csv")
except FileNotFoundError:
    print("❌ Dataset not found at data/raw/Crop_recommendation.csv")
    print("📂 Please make sure the CSV file is in the right location.")
    exit(1)

X = df.drop('label', axis=1)
y = df['label']

# Encode labels (convert crop names to numbers)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("📊 Training models...\n")

# 1. Random Forest
print("🌳 Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)
print(f"✅ Random Forest Accuracy: {rf_acc:.4f}")

# 2. XGBoost
print("⚡ Training XGBoost...")
xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print(f"✅ XGBoost Accuracy: {xgb_acc:.4f}")

# Save the best model
if xgb_acc >= rf_acc:
    best_model = xgb
    print("🏆 XGBoost saved as the best model!")
else:
    best_model = rf
    print("🏆 Random Forest saved as the best model!")

# Save model and encoder
joblib.dump(best_model, "models/best_crop_model.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("✅ Model and encoder saved successfully!")
print(f"📁 Files saved in: {os.path.abspath('models/')}")