# Binnôme : Assa KEBE et Stéphanie AGBANGLANON

# Churn prediction MLOps

Projet de prédiction de churn client.

## Prérequis

- Python 3.10+ recommandé
- `data/raw/churn.csv` doit être présent

## Installation

```bash
python -m venv .venv
```

Sous macOS / Linux :

```bash
source .venv/bin/activate
```

Sous Windows PowerShell :

```powershell
.\.venv\Scripts\Activate.ps1
```

Sous Windows CMD :

```cmd
.\.venv\Scripts\activate
```

Puis installez les dépendances :

```bash
pip install -r requirements.txt
```

## Exécution du pipeline

```bash
python main.py
```

Le pipeline effectue :

- la préparation des données via `src/prepare.py`
- l'entraînement du modèle via `src/train.py`
- l'évaluation via `src/evaluate.py`
- la sauvegarde des artefacts via `src/save.py`

Les artefacts générés sont stockés dans `artifacts/` :

- `artifacts/model.pkl`
- `artifacts/metrics.json`
- `artifacts/feature_columns.json`

Un fichier `mlflow.db` peut aussi être créé pour le suivi d'expérience.

## API

Démarrez l'API FastAPI :

```bash
uvicorn app:app --reload --port 8000
```

Routes disponibles :

- `GET /` : point de santé basique
- `GET /health` : vérifie que le modèle est chargé
- `GET /metrics` : retourne les compteurs d'appels et d'erreurs
- `POST /predict` : fait une prédiction de churn

### Exemple de payload pour `/predict`

```json
{
  "tenure_months": 12,
  "monthly_charges": 75.5,
  "total_charges": 906.0,
  "contract": "Month-to-month"
}
```

Valeurs autorisées pour `contract` :

- `Month-to-month`
- `One year`
- `Two year`

## Tests

```bash
pytest
```

Le dossier `tests/` contient des tests pour :

- l'API (`tests/test_api.py`)
- la non-régression du modèle (`tests/test_non_regression.py`)
- la préparation des données (`tests/test_prepare.py`)
