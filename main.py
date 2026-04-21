from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

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