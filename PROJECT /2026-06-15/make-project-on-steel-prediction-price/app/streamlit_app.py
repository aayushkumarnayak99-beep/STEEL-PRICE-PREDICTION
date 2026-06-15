import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.steel_price_prediction.train_model import FEATURES, METRICS_PATH, MODEL_PATH


st.set_page_config(page_title="Steel Price Predictor", page_icon="chart_with_upwards_trend", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("Steel Price Predictor")
st.caption("Predict estimated steel price in INR per metric ton from key commodity and market signals.")

if not MODEL_PATH.exists():
    st.error("Model file is missing. Run `python3 -m src.steel_price_prediction.train_model` first.")
    st.stop()

left, right = st.columns([1, 1])

with left:
    st.subheader("Market Inputs")
    iron_ore_price = st.slider("Iron ore price (USD/ton)", 60.0, 190.0, 122.0, 1.0)
    coal_price = st.slider("Coal price (USD/ton)", 80.0, 320.0, 188.0, 1.0)
    energy_cost = st.slider("Energy cost (USD/kWh)", 0.06, 0.20, 0.12, 0.005)
    usd_inr = st.slider("USD/INR exchange rate", 68.0, 92.0, 83.4, 0.1)

with right:
    st.subheader("Demand and Supply")
    demand_index = st.slider("Demand index", 70.0, 135.0, 104.0, 1.0)
    production_volume = st.slider("Production volume (million ton)", 6.0, 13.5, 9.8, 0.1)
    inventory_level = st.slider("Inventory level (million ton)", 2.5, 8.0, 5.1, 0.1)
    season = st.selectbox("Season", ["winter", "summer", "monsoon", "festival"], index=2)

input_values = {
    "iron_ore_price_usd_ton": iron_ore_price,
    "coal_price_usd_ton": coal_price,
    "energy_cost_usd_kwh": energy_cost,
    "usd_inr_rate": usd_inr,
    "demand_index": demand_index,
    "production_volume_million_ton": production_volume,
    "inventory_level_million_ton": inventory_level,
    "season": season,
}

model = load_model()
prediction = float(model.predict(pd.DataFrame([input_values], columns=FEATURES))[0])

st.divider()
metric_left, metric_right = st.columns([1, 2])
metric_left.metric("Predicted Price", f"INR {prediction:,.0f} / ton")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    metric_right.write("Model metrics")
    metric_right.json(metrics)

with st.expander("Input row"):
    st.dataframe(pd.DataFrame([input_values]), use_container_width=True)
