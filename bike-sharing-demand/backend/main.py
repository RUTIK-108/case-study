
from pathlib import Path
from datetime import date

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import feature_engineering


# File paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "bike_demand_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.json"


# Load the trained model
model = joblib.load(MODEL_PATH)

# Load the feature names used during training
import json

with open(FEATURE_PATH, "r") as file:
    feature_names = json.load(file)


app = FastAPI(
    title="Bike Sharing Demand Prediction",
    description="Predict hourly bike rental demand."
)


# Input data validation
class BikeDemandInput(BaseModel):

    dteday: date

    season: int = Field(ge=1, le=4)
    yr: int = Field(ge=0, le=1)
    mnth: int = Field(ge=1, le=12)
    hr: int = Field(ge=0, le=23)

    holiday: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    workingday: int = Field(ge=0, le=1)

    weathersit: int = Field(ge=1, le=4)

    temp: float = Field(ge=0, le=1)
    atemp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Bike Sharing Demand Prediction API is running"
    }


# Health check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prepare input for prediction
def prepare_input(data):

    input_data = pd.DataFrame([
        data.model_dump()
    ])

    # Apply the same feature engineering used during training
    processed_data = feature_engineering(input_data)

    # Keep the same feature order as the trained model
    processed_data = processed_data.reindex(
        columns=feature_names
    )

    return processed_data


# Prediction endpoint
@app.post("/predict")
def predict_demand(data: BikeDemandInput):

    try:
        input_features = prepare_input(data)

        prediction = model.predict(input_features)[0]

        prediction = max(0, float(prediction))

        return {
            "predicted_demand": round(prediction, 2),
            "unit": "bikes"
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
