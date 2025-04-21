# app.py
from fastapi import FastAPI
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load("models/demand_forecasting_model.pkl")
scaler = joblib.load("models/dai_scaler.pkl")

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Digital Health Demand Prediction API"}


@app.post("/predict/")
def predict(data: dict):
    # Extract input features
    input_data = pd.DataFrame([{
        'Digital Adoption Index': data['digital_adoption_index'],
        'DAI Business Sub-index': data['business'],
        'DAI People Sub-index': data['people'],
        'DAI Government Sub-index': data['government'],
        'Adoption_Growth': data['adoption_growth']
    }])

    # Scale the input data
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]
    return {"predicted_demand": "High" if prediction == 1 else "Low"}