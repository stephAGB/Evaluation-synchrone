import mlflow
import mlflow.sklearn

from src.evaluate import evaluate_model
from src.prepare import prepare_data
from src.save import save_artifacts
from src.train import train_model


DATA_PATH = "data/raw/churn.csv"
TARGET_COLUMN = "churn"
N_ESTIMATORS = 100
MAX_DEPTH = 5
TEST_SIZE = 0.2


def run_pipeline():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("churn_eval")

    with mlflow.start_run(run_name="baseline_rf"):
        X_train, X_test, y_train, y_test = prepare_data(
            csv_path=DATA_PATH,
            target_column=TARGET_COLUMN,
            test_size=TEST_SIZE,
        )

        model = train_model(
            X_train,
            y_train,
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
        )

        metrics = evaluate_model(model, X_test, y_test)
        paths = save_artifacts(model, metrics, X_train.columns)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("test_size", TEST_SIZE)

        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        mlflow.log_artifact(paths["model_path"])
        mlflow.log_artifact(paths["metrics_path"])
        mlflow.log_artifact(paths["columns_path"])
        mlflow.sklearn.log_model(model, name="model", registered_model_name="churn_model_eval_synchrone")

    return metrics


if __name__ == "__main__":
    print(run_pipeline())
