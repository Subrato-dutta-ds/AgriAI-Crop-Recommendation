# disease_vision.py
"""
Real disease classification using MobileNetV2 fine-tuned on PlantVillage.
Drop-in replacement for the random-choice placeholder.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

MODEL_PATH = "disease_model.keras"   # produced by train_disease.py
IMG_SIZE = (224, 224)

# PlantVillage class list (38). Order must match training.
LABELS = [
    "Apple___Apple_scab", "Apple___Black_rot",
    "Apple___Cedar_apple_rust", "Apple___healthy",
    "Corn___Cercospora_leaf_spot", "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight", "Corn___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight", "Grape___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Mosaic_virus", "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
    # add the remaining PlantVillage classes as you expand the training set
]

TREATMENTS = {
    "Tomato___Early_blight": [
        "Remove and destroy infected leaves.",
        "Improve airflow by staking / pruning.",
        "Avoid overhead irrigation; water at the base.",
        "Apply a locally approved fungicide if symptoms spread.",
    ],
    "Tomato___Late_blight": [
        "Isolate affected plants immediately.",
        "Apply copper-based fungicide per local guidance.",
        "Reduce leaf wetness; irrigate early morning.",
    ],
    "Potato___Late_blight": [
        "Destroy infected haulms.",
        "Apply protective fungicide before forecasted rain.",
    ],
    "Grape___Black_rot": [
        "Remove mummified berries and canes.",
        "Apply recommended fungicide at bud break.",
    ],
    # add per-class treatments progressively
}


def _load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH)


_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = _load_model()
    return _MODEL


def check_quality(img: Image.Image) -> tuple[bool, str]:
    """Return (is_ok, reason)."""
    gray = np.array(img.convert("L"), dtype=float)
    if gray.shape[0] < 64 or gray.shape[1] < 64:
        return False, "Image too small — please upload at least 256×256."
    blur_score = ndimage.laplace(gray).var()
    brightness = gray.mean()
    rgb = np.array(img.convert("RGB"), dtype=float)
    green_dominance = (
        rgb[..., 1].mean() > 0.85 * max(rgb[..., 0].mean(), rgb[..., 2].mean())
    )
    if blur_score < 30:
        return False, f"Image is blurry (sharpness={blur_score:.1f}). Retake steadily."
    if brightness < 40:
        return False, "Image too dark — retake in better light."
    if brightness > 235:
        return False, "Image overexposed — reduce glare."
    if not green_dominance:
        return False, "Leaf doesn't appear dominant. Crop tighter to the leaf."
    return True, "ok"


def predict_disease(img: Image.Image) -> dict:
    ok, reason = check_quality(img)
    if not ok:
        return {"ok": False, "reason": reason}

    model = _get_model()
    arr = np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.expand_dims(arr, 0)

    proba = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(proba))
    label = LABELS[idx]

    plant, _, disease = label.partition("___")
    disease = disease.replace("_", " ")
    healthy = disease.lower() == "healthy"

    return {
        "ok": True,
        "plant": plant,
        "disease": disease,
        "healthy": healthy,
        "confidence": float(proba[idx]),
        "treatment": [] if healthy else TREATMENTS.get(label, ["Consult local agronomist."]),
        "top3": [(LABELS[i], float(proba[i])) for i in np.argsort(proba)[-3:][::-1]],
    }