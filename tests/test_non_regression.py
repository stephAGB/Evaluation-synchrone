import pytest
import joblib
from src.prepare import prepare_data
from sklearn.metrics import accuracy_score

# Constantes synchronisées avec main.py
DATA_PATH = "data/raw/churn.csv"
TARGET_COLUMN = "churn"
TEST_SIZE = 0.2
ACCURACY_THRESHOLD = 0.70

@pytest.fixture(scope="session")
def test_data():
    """Charge et prépare les données de test exactement comme le pipeline."""
    # Note: On récupère uniquement X_test et y_test
    _, X_test, _, y_test = prepare_data(
        csv_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE
    )
    return X_test, y_test

def test_model_accuracy_above_threshold(test_data):
    """Vérifie la performance sur le set de test généré."""
    model = joblib.load("artifacts/model.pkl")
    X_test, y_test = test_data
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy >= ACCURACY_THRESHOLD, f"Regression detectee: {accuracy:.3f}"

def test_model_can_be_serialized(tmp_path):
    """Vérifie l'intégrité de l'objet modèle."""
    model = joblib.load("artifacts/model.pkl")
    model_path = tmp_path / "model.pkl"
    
    joblib.dump(model, model_path)
    assert joblib.load(model_path) is not None