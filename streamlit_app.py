import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/demand_forecasting_model.pkl")
scaler = joblib.load("models/dai_scaler.pkl")

st.title("📈 Digital Health Demand Predictor")

st.write("Enter digital readiness metrics to predict future demand:")

business = st.slider("Business Sub-index", 0.0, 1.0, 0.6)
people = st.slider("People Sub-index", 0.0, 1.0, 0.6)
government = st.slider("Government Sub-index", 0.0, 1.0, 0.6)
adoption_growth = st.slider("Recent Adoption Growth", 0.0, 0.5, 0.1)

data = pd.DataFrame([{
    'Digital Adoption Index': (business + people + government) / 3,
    'DAI Business Sub-index': business,
    'DAI People Sub-index': people,
    'DAI Government Sub-index': government,
    'Adoption_Growth': adoption_growth
}])

scaled = scaler.transform(data)
pred = model.predict(scaled)[0]
st.success(f"📊 **Predicted Demand Level:** {'High' if pred == 1 else 'Low'}")
