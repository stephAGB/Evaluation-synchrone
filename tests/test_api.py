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
    assert data["predictions"][0]["prediction"] == 0
    assert "label" in data["predictions"][0]
    assert "confidence" in data["predictions"][0]


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

