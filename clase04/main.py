from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Predictor de enfermedad cardíaca")
modelo = joblib.load("model.joblib")


class Paciente(BaseModel):
    age: int
    resting_bp: int
    cholesterol: int
    max_heart_rate: int
    exercise_angina: int  # 0 = no, 1 = sí
    st_depression: float


@app.get("/health")
def health():
    return {"mensaje": "API funcionando"}

@app.get("/", include_in_schema=False)
def inicio():
    return FileResponse("index.html")


@app.post("/predict")
def predict(paciente: Paciente):
    datos = pd.DataFrame([paciente.model_dump()])
    prob = modelo.predict_proba(datos)[0][1]
    return {
        "heart_disease": int(prob >= 0.5),
        "probabilidad": round(float(prob), 3),
    }