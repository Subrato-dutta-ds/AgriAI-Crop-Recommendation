# agri_core.py
"""
AgriAI core ML layer.
- Wraps the trained RandomForestClassifier
- Top-3 predictions from predict_proba
- SHAP explanations (falls back to feature importance if SHAP fails)
- Soil health analysis
- Fertilizer recommendation
- Irrigation recommendation
- Live weather auto-fill
"""
import pickle
import numpy as np
import urllib.request
import urllib.parse
import json
from dataclasses import dataclass

MODEL_PATH = "crop_model.pkl"

# ---- Prediction -------------------------------------------------------

@dataclass
class PredictionResult:
    crop: str
    confidence: float
    top3: list                 # [(crop, conf), ...]
    shap_contributions: dict   # feature -> signed contribution for top class


class CropRecommender:
    def __init__(self, model_path: str = MODEL_PATH):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.features = list(self.model.feature_names_in_)
        self.classes = list(self.model.classes_)
        self._explainer = None
        try:
            import shap
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
            crop=top3[0][0],
            confidence=top3[0][1],
            top3=top3,
            shap_contributions=contribs,
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
        # Fallback: feature_importances_ × standardized deviation
        imp = self.model.feature_importances_
        mean = np.array([50, 50, 50, 25, 65, 6.5, 100])
        std  = np.array([40, 40, 40,  8, 20, 1.5,  80])
        dev = (x[0] - mean) / std
        return {f: float(imp[i] * dev[i]) for i, f in enumerate(self.features)}

    def feature_importances(self) -> dict:
        return {f: float(v) for f, v in zip(self.features, self.model.feature_importances_)}


# ---- Soil health ------------------------------------------------------

# (min_plausible, max_plausible, ok_low, ok_high) — dataset-typical ranges
SOIL_RANGES = {
    "N":  (0,   140, 40,  90),
    "P":  (0,   145, 30,  80),
    "K":  (0,   205, 30, 100),
    "ph": (3.5,  10,  6.0, 7.5),
}


def analyze_soil(payload: dict) -> dict:
    report = {}
    for key, (lo, hi, ok_lo, ok_hi) in SOIL_RANGES.items():
        v = payload[key]
        status = "low" if v < ok_lo else "high" if v > ok_hi else "optimal"
        report[key] = {"value": v, "status": status, "range": (lo, hi)}
    ok = sum(1 for r in report.values() if r["status"] == "optimal")
    report["score"] = round(ok / len(SOIL_RANGES) * 5, 1)
    return report


# ---- Fertilizer -------------------------------------------------------

FERTILIZER_MAP = {
    "N": {"low":  "Apply Nitrogen-rich fertilizer (Urea / Ammonium Sulphate).",
          "high": "Reduce nitrogen input; consider legume rotation."},
    "P": {"low":  "Apply Phosphorus fertilizer (DAP / SSP).",
          "high": "Avoid additional P; confirm with soil test."},
    "K": {"low":  "Apply Potassium fertilizer (MOP / SOP).",
          "high": "Reduce K application; watch for leaching."},
}
NPK_HINT = {
    ("low","low","low"): "Balanced NPK 10-10-10",
    ("low","low","high"): "High-N-P, low-K blend (e.g. 20-20-0)",
    ("low","high","low"): "High-N, low-P, high-K (e.g. 20-0-20)",
    ("high","low","low"): "Low-N, high-P, high-K (e.g. 5-20-20)",
    ("optimal","optimal","optimal"): "Maintenance dose — no change needed",
}


def recommend_fertilizer(soil_report: dict) -> dict:
    actions, statuses = [], []
    for key in ("N", "P", "K"):
        st = soil_report[key]["status"]
        statuses.append(st)
        if st in FERTILIZER_MAP[key]:
            actions.append(FERTILIZER_MAP[key][st])
    ratio = NPK_HINT.get(tuple(statuses), "Follow local soil-test guidance.")
    return {"actions": actions, "ratio_hint": ratio}


# ---- Irrigation -------------------------------------------------------

# Crop seasonal water requirement in mm (approximate)
CROP_WATER_MM = {
    "rice": (1100, 1800), "wheat": (450, 650), "maize": (500, 800),
    "cotton": (700, 1200), "sugarcane": (1500, 2500),
    "chickpea": (300, 450), "lentil": (250, 400), "kidneybeans": (350, 500),
    "mungbean": (300, 450), "blackgram": (300, 450), "mothbeans": (250, 400),
    "pigeonpeas": (400, 600), "coffee": (1000, 1500), "coconut": (1200, 2000),
    "banana": (1200, 2000), "mango": (900, 1500), "grapes": (500, 800),
    "apple": (600, 1000), "orange": (900, 1200), "papaya": (1200, 1800),
    "pomegranate": (500, 800), "watermelon": (400, 600), "muskmelon": (400, 600),
    "jute": (500, 800),
}


def recommend_irrigation(crop: str, payload: dict) -> dict:
    lo, hi = CROP_WATER_MM.get(crop.lower(), (500, 800))
    season_rain = float(payload["rainfall"])
    deficit = max(0.0, lo - season_rain)
    if deficit < 50:
        level, freq = "Low",    "Rainfall alone is likely sufficient."
    elif deficit < 300:
        level, freq = "Medium", "Supplement every 4–5 days."
    elif deficit < 700:
        level, freq = "High",   "Irrigate every 2–3 days."
    else:
        level, freq = "Very High", "Drip/sprinkler daily; consider alternate crop."
    return {
        "crop_water_need_mm": (lo, hi),
        "expected_rainfall_mm": season_rain,
        "deficit_mm": round(deficit, 1),
        "irrigation_level": level,
        "suggested_frequency": freq,
    }


# ---- Weather ---------------------------------------------------------

def fetch_weather(city: str, api_key: str) -> dict:
    base = "https://api.openweathermap.org/data/2.5/weather"
    q = urllib.parse.urlencode({"q": city, "appid": api_key, "units": "metric"})
    with urllib.request.urlopen(f"{base}?{q}", timeout=10) as resp:
        data = json.loads(resp.read().decode())
    rain_1h = data.get("rain", {}).get("1h", 0)
    return {
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "rainfall": rain_1h * 24,            # rough daily mm
        "description": data["weather"][0]["description"],
        "city": data.get("name", city),
    }


# ---- Smoke test -------------------------------------------------------

if __name__ == "__main__":
    rec = CropRecommender()
    sample = {"N": 90, "P": 42, "K": 43, "temperature": 20.9,
              "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
    r = rec.predict(sample)
    print("Top-3:", [(c, round(p, 4)) for c, p in r.top3])
    print("SHAP:", {k: round(v, 3) for k, v in r.shap_contributions.items()})
    soil = analyze_soil(sample)
    print("Soil:", soil)
    print("Fertilizer:", recommend_fertilizer(soil))
    print("Irrigation:", recommend_irrigation(r.crop, sample))