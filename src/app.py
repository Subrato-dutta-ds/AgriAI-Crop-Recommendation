# app.py
import os
import pandas as pd
import streamlit as st
from PIL import Image

from agri_core import (
    CropRecommender, analyze_soil, recommend_fertilizer,
    recommend_irrigation, fetch_weather,
)
from disease_vision import predict_disease

st.set_page_config(page_title="AgriAI", page_icon="🌱", layout="wide")
st.title("🌱 AgriAI — Intelligent Agricultural Decision Support")

@st.cache_resource
def get_rec():
    return CropRecommender()

rec = get_rec()

# ---------- Sidebar: inputs ----------
st.sidebar.header("Farm inputs")

if "use_weather" not in st.session_state:
    st.session_state.use_weather = False

api_key = os.getenv("OWM_API_KEY", "")
city = st.sidebar.text_input("Location (for weather)", value="Jaipur")
if st.sidebar.button("Use current weather") and api_key:
    try:
        w = fetch_weather(city, api_key)
        st.session_state.temperature = w["temperature"]
        st.session_state.humidity = w["humidity"]
        st.session_state.rainfall = w["rainfall"]
        st.sidebar.success(f"{w['city']}: {w['description']}")
    except Exception as e:
        st.sidebar.error(f"Weather error: {e}")

N = st.sidebar.number_input("Nitrogen (N)",  0.0, 140.0, st.session_state.get("N", 90.0))
P = st.sidebar.number_input("Phosphorus (P)",0.0, 145.0, st.session_state.get("P", 42.0))
K = st.sidebar.number_input("Potassium (K)", 0.0, 205.0, st.session_state.get("K", 43.0))
temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, st.session_state.get("temperature", 25.0))
hum  = st.sidebar.number_input("Humidity (%)",     0.0, 100.0, st.session_state.get("humidity", 70.0))
ph   = st.sidebar.number_input("Soil pH",          0.0, 14.0,  st.session_state.get("ph", 6.5))
rain = st.sidebar.number_input("Rainfall (mm, season)", 0.0, 400.0, st.session_state.get("rainfall", 120.0))

payload = {"N": N, "P": P, "K": K, "temperature": temp,
           "humidity": hum, "ph": ph, "rainfall": rain}

# ---------- Tabs ----------
tab_crop, tab_soil, tab_disease = st.tabs(["🌾 Crop", "🧪 Soil & Water", "📷 Disease"])

with tab_crop:
    r = rec.predict(payload)

    c1, c2, c3 = st.columns([2, 2, 3])
    with c1:
        st.metric("Recommended crop", r.crop.title(),
                  f"{r.confidence*100:.1f}% confidence")
    with c2:
        st.markdown("**Top 3**")
        st.table(pd.DataFrame(
            [{"crop": c.title(), "confidence": f"{p*100:.2f}%"} for c, p in r.top3]))
    with c3:
        st.markdown("**Why this crop (SHAP)**")
        contribs = sorted(r.shap_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        st.bar_chart(pd.DataFrame(
            {"feature": [k for k, _ in contribs],
             "contribution": [v for _, v in contribs]}).set_index("feature"))

with tab_soil:
    soil = analyze_soil(payload)
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Soil health")
        st.write(f"Score: **{soil['score']} / 5**")
        for k in ("N", "P", "K", "ph"):
            icon = {"optimal": "🟢", "low": "🔴", "high": "🟡"}[soil[k]["status"]]
            st.write(f"{icon} **{k}** = {soil[k]['value']} → {soil[k]['status']}")
    with col_b:
        st.subheader("Fertilizer")
        fert = recommend_fertilizer(soil)
        st.write(f"**Suggested ratio:** {fert['ratio_hint']}")
        for a in fert["actions"]:
            st.write(f"- {a}")

    st.subheader("Irrigation")
    irr = recommend_irrigation(r.crop, payload)
    st.write(f"**Water need:** {irr['crop_water_need_mm'][0]}–{irr['crop_water_need_mm'][1]} mm")
    st.write(f"**Expected rainfall:** {irr['expected_rainfall_mm']} mm")
    st.write(f"**Deficit:** {irr['deficit_mm']} mm → "
             f"**{irr['irrigation_level']}** ({irr['suggested_frequency']})")

with tab_disease:
    st.write("Upload a clear leaf photo. The model will reject blurry, dark "
             "or non-leaf images before predicting.")
    up = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)
        st.image(img, caption="Uploaded", width=300)
        if os.path.exists("disease_model.keras"):
            res = predict_disease(img)
            if not res["ok"]:
                st.warning(res["reason"])
            else:
                st.success(f"**{res['plant']} — {res['disease']}**  "
                           f"({res['confidence']*100:.1f}%)")
                if res["healthy"]:
                    st.info("Leaf looks healthy. No treatment needed.")
                else:
                    for step in res["treatment"]:
                        st.write(f"- {step}")
        else:
            st.info("No `disease_model.keras` found. Run `train_disease.py` first.")