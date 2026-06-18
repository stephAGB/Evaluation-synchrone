import json
import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


API_TOKEN = "churn-demo-token"
MODEL_PATH = Path("artifacts/model.pkl")
FEATURE_COLUMNS_PATH = Path("artifacts/feature_columns.json")

metrics = {
    "n_predictions": 0,
    "n_errors": 0,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

model = None
feature_columns = []

try:
    logger.info("Chargement du modèle et des features depuis %s et %s...", MODEL_PATH, FEATURE_COLUMNS_PATH)
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    logger.info("Chargement des artefacts réussi (Modèle chargé et %d colonnes chargées)", len(feature_columns))
except Exception as e:
    logger.error("Erreur au chargement des artefacts : %s", e)


class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]


app = FastAPI(title="Churn Prediction API", version="1.0")


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    status = "healthy" if model is not None else "unhealthy"
    logger.info("Appel route health (/health) - status=%s, model_loaded=%s", status, model is not None)
    return {"status": status, "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    logger.info("Appel route metrics (/metrics) - metrics=%s", metrics)
    return metrics


@app.post("/predict")
def predict(payload: CustomerInput):
    start_time = time.time()
    logger.info("Requête de prédiction reçue avec payload : %s", payload.model_dump())

    if model is None:
        metrics["n_errors"] += 1
        logger.error("Échec de la prédiction : le modèle n'est pas chargé (None)")
        raise HTTPException(status_code=500, detail="Modèle indisponible")

    try:
        df = pd.DataFrame([payload.model_dump()])
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(df_aligned)[0]
        confidence = model.predict_proba(df_aligned)[0].max()

        metrics["n_predictions"] += 1
        latency = time.time() - start_time
        logger.info(
            "Prédiction réussie : prediction=%s, label=%s, latence=%.4fs",
            prediction,
            "churn" if prediction == 1 else "no_churn",
            latency,
        )

        return {
            "prediction": int(prediction),
            "label": "churn" if prediction == 1 else "no_churn",
            "confidence": float(confidence),
        }
    except Exception as e:
        metrics["n_errors"] += 1
        logger.error("Erreur pendant la prédiction : %s", e)
        raise HTTPException(status_code=500, detail=str(e))
