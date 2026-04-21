from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import gdown

MODEL_PATH = "Models/voting_classifier.sav"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    os.makedirs("Models", exist_ok=True)
    url = "https://drive.google.com/uc?id=1S0SAu6Pj2nbb2euwUN5yuK1nlRihhsMW"
    print("Downloading model...")
    gdown.download(url, MODEL_PATH, quiet=False)
app = FastAPI()

# Load model
voting_clf = joblib.load("Models/voting_classifier.sav")
scaler = joblib.load("Models/scaler.sav")

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
    return {"status": "API running"}

@app.post("/predict")
def predict(data: InputData):
    try:
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