from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from models.ml_model import predict_ml
from models.dl_model import predict_dl
from models.qml_model import predict_qml
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QuantumMed AI API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    # 13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    features: List[float]

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: PatientData):
    if len(data.features) != 13:
        raise HTTPException(status_code=400, detail="Must provide exactly 13 features")
    
    try:
        ml_pred, ml_conf = predict_ml(data.features)
        dl_pred, dl_conf = predict_dl(data.features)
        qml_pred, qml_conf = predict_qml(data.features)
        
        return {
            "ml": {"prediction": ml_pred, "confidence": ml_conf},
            "dl": {"prediction": dl_pred, "confidence": dl_conf},
            "qml": {"prediction": qml_pred, "confidence": qml_conf}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
