# ⬇️ PASTE FASTAPI ENDPOINT CODE HERE
import os
import json
import time
import logging
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="SMS Spam Prediction API", version="1.0.0")

MODEL_PATH = Path("models/best_model.joblib")
if not MODEL_PATH.exists():
    raise FileNotFoundError("Run 'uv run python src/train.py' first to generate models/best_model.joblib")

model = joblib.load(MODEL_PATH)
logging.basicConfig(filename="logs/predictions.log", level=logging.INFO, format="%(message)s")

class MessageRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict(req: MessageRequest):
    start = time.time()
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty")
        
    try:
        pred = model.predict([req.message])[0]
        prob = model.predict_proba([req.message])[0]
        confidence = float(prob[pred])
        label = "spam" if pred == 1 else "ham"
        
        latency = round(time.time() - start, 4)
        log_entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "prediction": label,
            "confidence": confidence,
            "latency_s": latency,
            "status": 200
        }
        logging.info(json.dumps(log_entry))
        
        return {"prediction": label, "confidence": confidence, "latency_s": latency}
    except Exception as e:
        logging.error(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "error": str(e), "status": 500}))
        raise HTTPException(500, "Prediction failed. Check server logs.")