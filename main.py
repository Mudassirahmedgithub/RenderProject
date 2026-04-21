from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from threading import Lock

app = FastAPI()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Paths
MODEL_PATH = "Models/voting_classifier.joblib"
SCALER_PATH = "Models/scaler.joblib"

# Check files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

# Lazy-loaded globals
voting_clf = None
scaler = None
model_lock = Lock()

# Feature names
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

# Lazy loader
def load_models():
    global voting_clf, scaler

    if voting_clf is None or scaler is None:
        with model_lock:
            if voting_clf is None:
                voting_clf = joblib.load(MODEL_PATH)
            if scaler is None:
                scaler = joblib.load(SCALER_PATH)

@app.get("/")
def home():
    return {"status": "API running 🚀"}

@app.post("/predict")
def predict(data: InputData):
    try:
        load_models()

        if len(data.features) != len(feature_names):
            return {"error": f"Expected {len(feature_names)} features"}

        df = pd.DataFrame(
            [data.features],
            columns=feature_names,
            dtype="float32"
        )

        scaled = scaler.transform(df)

        pred = voting_clf.predict(scaled)[0]
        confidence = voting_clf.predict_proba(scaled)[0]

        return {
            "prediction": int(pred),
            "confidence": float(max(confidence))
        }

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, workers=1)