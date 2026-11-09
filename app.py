import os
import json
import pandas as pd
import streamlit as st
import altair as alt
from pathlib import Path
from PIL import Image
from agri_core import CropRecommender, analyze_soil, recommend_fertilizer, recommend_irrigation, fetch_weather
from disease_vision import predict_disease

st.set_page_config(page_title="AgriAI | Hackathon", page_icon="🌱", layout="wide")

# Custom CSS for Hackathon Glow-up
st.markdown("""
<style>
    .hero { background: linear-gradient(90deg, #059669 0%, #10b981 100%); padding: 2rem; border-radius: 12px; color: white; text-align: center; margin-bottom: 2rem; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3); }
    .hero h1 { color: white; margin: 0; }
    div[data-testid="stMetric"] { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2); }
    div[data-testid="stMetricLabel"] { color: #94a3b8; }
    div[data-testid="stMetricValue"] { color: #10b981; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>🌱 AgriAI Platform</h1>
    <p style="opacity: 0.9;">Intelligent Agricultural Decision Support System</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def get_rec(): return CropRecommender()
rec = get_rec()

# --- SIDEBAR ---
st.sidebar.header("🚜 Farm Inputs")
api_key = os.getenv("OWM_API_KEY", "")
city = st.sidebar.text_input("Location", value="Jaipur")
if st.sidebar.button("📡 Fetch Live Weather", width='stretch') and api_key:
    try:
        w = fetch_weather(city, api_key)
        st.session_state.temperature = w["temperature"]
        st.session_state.humidity = w["humidity"]
        st.session_state.rainfall = w["rainfall"]
        st.sidebar.success(f"✅ Loaded {w['city']} weather")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

st.sidebar.markdown("---")
N = st.sidebar.slider("Nitrogen (N)", 0.0, 140.0, st.session_state.get("N", 90.0))
P = st.sidebar.slider("Phosphorus (P)", 0.0, 145.0, st.session_state.get("P", 42.0))
K = st.sidebar.slider("Potassium (K)", 0.0, 205.0, st.session_state.get("K", 43.0))
temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, st.session_state.get("temperature", 25.0))
hum  = st.sidebar.slider("Humidity (%)", 0.0, 100.0, st.session_state.get("humidity", 70.0))
rain = st.sidebar.slider("Rainfall (mm)", 0.0, 400.0, st.session_state.get("rainfall", 120.0))
ph   = st.sidebar.slider("pH Level", 0.0, 14.0, st.session_state.get("ph", 6.5))

payload = {"N": N, "P": P, "K": K, "temperature": temp, "humidity": hum, "ph": ph, "rainfall": rain}

# --- MAIN TABS ---
tab_crop, tab_soil, tab_disease = st.tabs(["🌾 Crop Advisory", "🧪 Soil & Water", "🔬 Disease Scan"])

with tab_crop:
    r = rec.predict(payload)
    st.subheader("Prediction Results")
    
    cols = st.columns(3)
    medals = ["🥇 Best Match", "🥈 Alternative", "🥉 Alternative"]
    for idx, col in enumerate(cols):
        if idx < len(r.top3):
            crop, conf = r.top3[idx]
            col.metric(medals[idx], crop.title(), f"{conf*100:.1f}%")

    st.markdown("---")
    st.subheader("🧠 Explainable AI (SHAP)")
    st.caption("Feature impact on this recommendation. Green pushes UP, Red pushes DOWN.")
    
    contribs = sorted(r.shap_contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
    df_shap = pd.DataFrame(contribs, columns=["Feature", "Impact"])
    
    chart = alt.Chart(df_shap).mark_bar(cornerRadius=5).encode(
        x=alt.X('Impact:Q', title='Impact Score'),
        y=alt.Y('Feature:N', sort='-x', title=''),
        color=alt.condition(alt.datum.Impact > 0, alt.value('#10b981'), alt.value('#ef4444'))
    ).properties(height=300)
    st.altair_chart(chart, width='stretch')

with tab_soil:
    soil = analyze_soil(payload)
    st.subheader(f"Overall Soil Health: {soil['score']} / 5.0")
    st.progress(soil['score'] / 5.0)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Nutrient Status")
        for k in ("N", "P", "K"):
            status = soil[k]["status"]
            icon = {"optimal": "🟢", "low": "🔴", "high": "🟡"}[status]
            st.info(f"{icon} **{k}** = {soil[k]['value']} ({status.title()})")
    with col_b:
        st.markdown("### Fertilizer Advice")
        fert = recommend_fertilizer(soil)
        if not fert["actions"]: st.success("✅ Soil is perfectly balanced.")
        else:
            for a in fert["actions"]: st.warning(f"💡 {a}")

    st.markdown("---")
    st.subheader("💧 Irrigation Recommendation")
    irr = recommend_irrigation(r.crop, payload)
    st.metric("Water Deficit", f"{irr['deficit_mm']} mm", irr['irrigation_level'])

with tab_disease:
    st.subheader("Leaf Disease Detection")
    up = st.file_uploader("Upload a leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up:
        col1, col2 = st.columns([1, 2])
        with col1:
            img = Image.open(up)
            st.image(img, caption="Uploaded Image", width='stretch')
        with col2:
            with st.spinner("Scanning leaf..."):
                res = predict_disease(img)
                if not res["ok"]: st.error(f"Quality Check Failed: {res['reason']}")
                else:
                    st.success(f"**Detected:** {res['plant']} - {res['disease']}")
                    st.metric("Confidence", f"{res['confidence']*100:.1f}%")
                    if res["healthy"]: st.balloons(); st.info("🎉 Leaf is healthy!")
                    else:
                        st.markdown("### 🩺 Treatment Plan")
                        for step in res["treatment"]: st.write(f"- {step}")