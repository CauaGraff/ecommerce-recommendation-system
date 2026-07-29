import json
from pathlib import Path

import mlflow
import numpy as np

from src.config.settings import get_settings
from src.evaluation.metrics import compute_regression_metrics
from src.models.baselines import GlobalAverageBaseline, LinearRegressionBaseline

BASELINES = {
    "global_average": GlobalAverageBaseline,
    "linear_regression": LinearRegressionBaseline,
}


def load_features(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Carrega os arrays de features de um split (treino ou teste).

    Args:
        processed_dir: Pasta com os dados processados.
        split: ``"train"`` ou ``"test"``.

    Returns:
        Tupla ``(user_item_idx, ratings)``.
    """
    features = np.load(processed_dir / f"{split}_features.npz")
    user_item_idx = np.stack([features["user_idx"], features["item_idx"]], axis=1)
    return user_item_idx, features["rating"]


def evaluate_baseline(name: str, baseline_cls: type, processed_dir: Path) -> dict[str, float]:
    """Treina e avalia um baseline, registrando o run no MLflow.

    Args:
        name: Nome identificador do baseline (usado no run do MLflow).
        baseline_cls: Classe do baseline a ser instanciada.
        processed_dir: Pasta com os dados processados.

    Returns:
        Dicionário de métricas do baseline no conjunto de teste.
    """
    train_x, train_y = load_features(processed_dir, "train")
    test_x, test_y = load_features(processed_dir, "test")

    baseline = baseline_cls()
    baseline.fit(train_x, train_y)
    predictions = baseline.predict(test_x)
    metrics = compute_regression_metrics(predictions, test_y)

    with mlflow.start_run(run_name=f"baseline-{name}"):
        mlflow.log_param("model_type", name)
        mlflow.log_metrics(metrics)

    return metrics


def main() -> None:
    """Treina e avalia todos os baselines registrados, imprimindo o resumo."""
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    processed_dir = Path(settings.data_processed_path)
    results = {
        name: evaluate_baseline(name, cls, processed_dir) for name, cls in BASELINES.items()
    }

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "baseline_metrics.json").write_text(json.dumps(results, indent=2))

    for name, metrics in results.items():
        print(f"{name}: RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} R2={metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
