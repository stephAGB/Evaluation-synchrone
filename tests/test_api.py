import numpy as np
from fastapi.testclient import TestClient

import app as api_app


class DummyModel:
    def predict(self, X):
        return [0]

    def predict_proba(self, X):
        return np.array([[0.8, 0.2]])


client = TestClient(api_app.app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(
        api_app,
        "feature_columns",
        [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_One year",
            "contract_Two year",
        ],
    )

    payload = {
        "tenure_months": 12,
        "monthly_charges": 75.5,
        "total_charges": 906.0,
        "contract": "Month-to-month",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction" in response.json()


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={"tenure_months": 12})

    assert response.status_code == 422


def test_predict_batch_valid(monkeypatch):
    class DummyOrderedModel:
        def predict(self, X):
            # Si tenure_months == 12, retourne [1] (churn), sinon [0] (no_churn)
            if "tenure_months" in X.columns and X["tenure_months"].iloc[0] == 12:
                return [1]
            return [0]

        def predict_proba(self, X):
            if "tenure_months" in X.columns and X["tenure_months"].iloc[0] == 12:
                return np.array([[0.1, 0.9]])
            return np.array([[0.8, 0.2]])

    monkeypatch.setattr(api_app, "model", DummyOrderedModel())
    monkeypatch.setattr(
        api_app,
        "feature_columns",
        [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_One year",
            "contract_Two year",
        ],
    )

    payload = {
        "inputs": [
            {"tenure_months": 12, "monthly_charges": 75.5, "total_charges": 906.0, "contract": "Month-to-month"},
            {"tenure_months": 24, "monthly_charges": 90.0, "total_charges": 2160.0, "contract": "One year"}
        ]
    }
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["n_inputs"] == 2
    assert len(data["predictions"]) == 2
    
    # Vérifie que la première prédiction correspond à l'entrée avec tenure_months=12 (churn)
    assert data["predictions"][0]["prediction"] == 1
    assert data["predictions"][0]["label"] == "churn"
    
    # Vérifie que la deuxième prédiction correspond à l'entrée avec tenure_months=24 (no_churn)
    assert data["predictions"][1]["prediction"] == 0
    assert data["predictions"][1]["label"] == "no_churn"



def test_predict_batch_exceeds_limit():
    payload = {
        "inputs": [
            {"tenure_months": 12, "monthly_charges": 75.5, "total_charges": 906.0, "contract": "Month-to-month"}
        ] * 101
    }
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 413
    assert "dépasse" in response.json()["detail"] or "maximum" in response.json()["detail"]


def test_predict_batch_invalid_element():
    payload = {
        "inputs": [
            {"tenure_months": 12, "monthly_charges": 75.5, "total_charges": 906.0, "contract": "Month-to-month"},
            {"tenure_months": -5, "monthly_charges": 90.0, "total_charges": 2160.0, "contract": "One year"}  # -5 est invalide (ge=0)
        ]
    }
    response = client.post("/predict_batch", json=payload)

    assert response.status_code == 400
    assert "index 1" in response.json()["detail"] or "l'index 1" in response.json()["detail"]


def test_predict_batch_metrics_increments(monkeypatch):
    monkeypatch.setattr(api_app, "model", DummyModel())
    monkeypatch.setattr(
        api_app,
        "feature_columns",
        [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "contract_One year",
            "contract_Two year",
        ],
    )

    # Get initial values
    initial_metrics = client.get("/metrics").json()
    init_requests = initial_metrics.get("n_batch_requests", 0)
    init_inputs = initial_metrics.get("n_batch_inputs_total", 0)

    # Perform batch prediction request
    payload = {
        "inputs": [
            {"tenure_months": 12, "monthly_charges": 75.5, "total_charges": 906.0, "contract": "Month-to-month"},
            {"tenure_months": 24, "monthly_charges": 90.0, "total_charges": 2160.0, "contract": "One year"}
        ]
    }
    client.post("/predict_batch", json=payload)

    # Verify metrics incremented correctly
    new_metrics = client.get("/metrics").json()
    assert new_metrics["n_batch_requests"] == init_requests + 1
    assert new_metrics["n_batch_inputs_total"] == init_inputs + 2


