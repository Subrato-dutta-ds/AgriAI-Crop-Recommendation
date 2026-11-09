import pickle
import numpy as np
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass

MODEL_PATH = "crop_model.pkl"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

@dataclass
class PredictionResult:
    crop: str
    confidence: float
    top3: list
    shap_contributions: dict

class CropRecommender:
    def __init__(self, model_path: str = MODEL_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        # Handle CalibratedClassifierCV wrapper hiding attributes
        if hasattr(self.model, "feature_names_in_"):
            self.features = list(self.model.feature_names_in_)
        else:
            self.features = FEATURES
            
        self.classes = list(self.model.classes_)
        
        self._explainer = None
        try:
            import shap
            # Explain the underlying Random Forest inside the calibration wrapper
            if hasattr(self.model, "calibrated_classifiers_"):
                base_rf = self.model.calibrated_classifiers_[0].estimator
                self._explainer = shap.TreeExplainer(base_rf)
            else:
                self._explainer = shap.TreeExplainer(self.model)
        except Exception:
            self._explainer = None

    def _to_array(self, payload: dict) -> np.ndarray:
        return np.array([[payload[f] for f in self.features]], dtype=float)

    def predict(self, payload: dict) -> PredictionResult:
        x = self._to_array(payload)
        proba = self.model.predict_proba(x)[0]
        order = np.argsort(proba)[::-1]
        top3 = [(self.classes[i], float(proba[i])) for i in order[:3]]
        contribs = self._explain(x, order[0])
        return PredictionResult(
            crop=top3[0][0], confidence=top3[0][1], top3=top3, shap_contributions=contribs,
        )

    def _explain(self, x: np.ndarray, class_idx: int) -> dict:
        if self._explainer is not None:
            try:
                sv = self._explainer.shap_values(x)
                if isinstance(sv, list):
                    arr = np.asarray(sv[class_idx])[0]
                else:
                    arr = np.asarray(sv)
                    if arr.ndim == 3:
                        arr = arr[0, :, class_idx]
                    elif arr.ndim == 2:
                        arr = arr[0]
                return {f: float(v) for f, v in zip(self.features, arr)}
            except Exception:
                pass
        
        # Fallback
        try:
            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
            elif hasattr(self.model, "calibrated_classifiers_"):
                imp = np.mean([clf.estimator.feature_importances_ for clf in self.model.calibrated_classifiers_], axis=0)
            else:
                imp = np.ones(len(self.features)) / len(self.features)
        except Exception:
            imp = np.ones(len(self.features)) / len(self.features)
            
        mean = np.array([50, 50, 50, 25, 65, 6.5, 100])
        std  = np.array([40, 40, 40,  8, 20, 1.5,  80])
        dev = (x[0] - mean) / std
        return {f: float(imp[i] * dev[i]) for i, f in enumerate(self.features)}

SOIL_RANGES = {"N": (0, 140, 40, 90), "P": (0, 145, 30, 80), "K": (0, 205, 30, 100), "ph": (3.5, 10, 6.0, 7.5)}

def analyze_soil(payload: dict) -> dict:
    report = {}
    for key, (lo, hi, ok_lo, ok_hi) in SOIL_RANGES.items():
        v = payload[key]
        status = "low" if v < ok_lo else "high" if v > ok_hi else "optimal"
        report[key] = {"value": v, "status": status, "range": (lo, hi)}
    ok = sum(1 for r in report.values() if r["status"] == "optimal")
    report["score"] = round(ok / len(SOIL_RANGES) * 5, 1)
    return report

FERTILIZER_MAP = {
    "N": {"low": "Apply Nitrogen-rich fertilizer.", "high": "Reduce nitrogen input."},
    "P": {"low": "Apply Phosphorus fertilizer.", "high": "Avoid additional P."},
    "K": {"low": "Apply Potassium fertilizer.", "high": "Reduce K application."},
}

def recommend_fertilizer(soil_report: dict) -> dict:
    actions = []
    for key in ("N", "P", "K"):
        st = soil_report[key]["status"]
        if st in FERTILIZER_MAP[key]:
            actions.append(FERTILIZER_MAP[key][st])
    return {"actions": actions}

CROP_WATER_MM = {
    "rice": (1100, 1800), "wheat": (450, 650), "maize": (500, 800), "cotton": (700, 1200),
    "chickpea": (300, 450), "lentil": (250, 400), "kidneybeans": (350, 500),
    "mungbean": (300, 450), "blackgram": (300, 450), "mothbeans": (250, 400),
    "pigeonpeas": (400, 600), "coffee": (1000, 1500), "coconut": (1200, 2000),
    "banana": (1200, 2000), "mango": (900, 1500), "grapes": (500, 800),
    "apple": (600, 1000), "orange": (900, 1200), "papaya": (1200, 1800),
    "pomegranate": (500, 800), "watermelon": (400, 600), "muskmelon": (400, 600), "jute": (500, 800),
}

def recommend_irrigation(crop: str, payload: dict) -> dict:
    lo, hi = CROP_WATER_MM.get(crop.lower(), (500, 800))
    season_rain = float(payload["rainfall"])
    deficit = max(0.0, lo - season_rain)
    if deficit < 50: level, freq = "Low", "Rainfall alone is sufficient."
    elif deficit < 300: level, freq = "Medium", "Supplement every 4-5 days."
    elif deficit < 700: level, freq = "High", "Irrigate every 2-3 days."
    else: level, freq = "Very High", "Drip/sprinkler daily."
    return {"deficit_mm": round(deficit, 1), "irrigation_level": level, "suggested_frequency": freq}

def fetch_weather(city: str, api_key: str) -> dict:
    base = "https://api.openweathermap.org/data/2.5/weather"
    q = urllib.parse.urlencode({"q": city, "appid": api_key, "units": "metric"})
    with urllib.request.urlopen(f"{base}?{q}", timeout=10) as resp:
        data = json.loads(resp.read().decode())
    rain_1h = data.get("rain", {}).get("1h", 0)
    return {"temperature": data["main"]["temp"], "humidity": data["main"]["humidity"], "rainfall": rain_1h * 24, "description": data["weather"][0]["description"], "city": data.get("name", city)}
