# --- IMPORTS ---
import json
import logging
import time
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# --- CONFIGURATION & CONSTANTES ---
API_TOKEN = "churn-demo-token"
MODEL_PATH = Path("artifacts/model.pkl")
FEATURE_COLUMNS_PATH = Path("artifacts/feature_columns.json")

# Métriques globales de l'API stockées en mémoire
metrics = {
    "n_predictions": 0,
    "n_errors": 0,
}

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

model = None
feature_columns = []

# --- CHARGEMENT DU MODÈLE ET DES FEATURES ---
try:
    logger.info("Chargement du modèle et des features depuis %s et %s...", MODEL_PATH, FEATURE_COLUMNS_PATH)
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    logger.info("Chargement des artefacts réussi (Modèle chargé et %d colonnes chargées)", len(feature_columns))
except Exception as e:
    logger.error("Erreur au chargement des artefacts : %s", e)


# --- MODÈLES DE DONNÉES (SCHÉMAS PYDANTIC) ---

class CustomerInput(BaseModel):
    """Schéma de validation pour les données d'un client individuel."""
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]


class BatchInput(BaseModel):
    """Schéma de validation pour une requête de prédiction par lot (batch)."""
    inputs: list[dict]


# --- INITIALISATION DE L'API ---
app = FastAPI(title="Churn Prediction API", version="1.0")


# --- ENDPOINTS GET ---

@app.get("/")
def root():
    """Route racine pour vérifier si l'API répond."""
    return {"status": "ok"}


@app.get("/health")
def health():
    """Route de santé (Liveness/Readiness probe) indiquant l'état de chargement du modèle."""
    status = "healthy" if model is not None else "unhealthy"
    logger.info("Appel route health (/health) - status=%s, model_loaded=%s", status, model is not None)
    return {"status": status, "model_loaded": model is not None}


@app.get("/metrics")
def get_metrics():
    """Route exposant les statistiques de performance de l'API."""
    logger.info("Appel route metrics (/metrics) - metrics=%s", metrics)
    return metrics


# --- ENDPOINTS POST (INFÉRENCE) ---

@app.post("/predict")
def predict(payload: CustomerInput):
    """Effectue une prédiction de churn pour un client unique."""
    start_time = time.time()
    logger.info("Requête de prédiction reçue avec payload : %s", payload.model_dump())

    # Vérification de la disponibilité du modèle
    if model is None:
        metrics["n_errors"] += 1
        logger.error("Échec de la prédiction : le modèle n'est pas chargé (None)")
        raise HTTPException(status_code=500, detail="Modèle indisponible")

    try:
        # Transformation du payload Pydantic en DataFrame pandas
        df = pd.DataFrame([payload.model_dump()])
        # Encodage One-Hot des variables catégorielles
        df_encoded = pd.get_dummies(df, drop_first=True)
        # Alignement avec les colonnes attendues par le modèle
        df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Inférence et extraction de la confiance de prédiction
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


@app.post("/predict_batch")
def predict_batch(payload: BatchInput):
    """Effectue des prédictions de churn par lot (batch) de taille max 100."""
    start_time = time.time()
    logger.info("Requête de prédiction batch reçue : %d éléments", len(payload.inputs))

    # Limitation stricte de la taille du batch (max 100 éléments) - HTTP 413
    if len(payload.inputs) > 100:
        logger.warning("Requête batch rejetée : %d éléments dépasse la taille limite (100)", len(payload.inputs))
        raise HTTPException(
            status_code=413,
            detail="Le batch dépasse la taille maximale autorisée (100 éléments)."
        )

    # Validation individuelle de chaque élément pour identifier précisément les entrées invalides
    validated_inputs = []
    for i, item in enumerate(payload.inputs):
        try:
            # Réutilise le schéma et la logique de validation de CustomerInput
            validated_inputs.append(CustomerInput(**item))
        except Exception as e:
            logger.warning("Erreur de validation sur l'entrée index %d : %s", i, e)
            raise HTTPException(
                status_code=400,
                detail=f"Entrée invalide à l'index {i}: {str(e)}"
            )

    # Vérification de la disponibilité du modèle
    if model is None:
        metrics["n_errors"] += 1
        logger.error("Échec du batch : modèle non chargé")
        raise HTTPException(status_code=500, detail="Modèle indisponible")

    try:
        predictions = []
        # Inférence séquentielle pour chaque client validé
        for input_item in validated_inputs:
            df = pd.DataFrame([input_item.model_dump()])
            df_encoded = pd.get_dummies(df, drop_first=True)
            df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

            prediction = model.predict(df_aligned)[0]
            confidence = model.predict_proba(df_aligned)[0].max()

            predictions.append({
                "prediction": int(prediction),
                "label": "churn" if prediction == 1 else "no_churn",
                "confidence": float(confidence),
            })
            metrics["n_predictions"] += 1

        latency = time.time() - start_time
        logger.info("Batch prédiction réussi : %d prédictions faites en %.4fs", len(predictions), latency)

        return {
            "predictions": predictions,
            "n_inputs": len(predictions)
        }
    except Exception as e:
        metrics["n_errors"] += 1
        logger.error("Erreur lors de l'inférence batch : %s", e)
        raise HTTPException(status_code=500, detail=str(e))
