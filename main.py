from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

# Paths (files already inside Models folder)
MODEL_PATH = "Models/voting_classifier.joblib"
SCALER_PATH = "Models/scaler.joblib"

# Check if files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

# Load model and scaler
voting_clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Feature names (same order!)
feature_names = [
    'Active Min', 'Fwd PSH Flags', 'SYN Flag Count', 'Flow Packets/s',
    'Fwd Packets/s', 'Active Mean', 'Active Std', 'Flow IAT Min',
    'Bwd IAT Total', 'URG Flag Count', 'Bwd IAT Std', 'FIN Flag Count',
    'Min Packet Length', 'Down/Up Ratio', 'Total Length of Fwd Packets',
    'Subflow Fwd Bytes', 'PSH Flag Count', 'Bwd IAT Max'
]

# Input schema
class InputData(BaseModel):
    features: list[float]

@app.get("/")
def home():
    return {"status": "API running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    try:
        if len(data.features) != len(feature_names):
            return {"error": f"Expected {len(feature_names)} features"}

        df = pd.DataFrame([data.features], columns=feature_names)
        scaled = scaler.transform(df)

        pred = voting_clf.predict(scaled)[0]
        confidence = voting_clf.predict_proba(scaled)[0]

        return {
            "prediction": int(pred),
            "confidence": float(max(confidence))
        }

    except Exception as e:
        return {"error": str(e)}