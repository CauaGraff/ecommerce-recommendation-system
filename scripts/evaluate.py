"""Stage 4 do pipeline DVC: avaliação do modelo treinado no conjunto de teste.

Carrega o modelo salvo, calcula métricas de erro no conjunto de teste e as
registra no mesmo experimento do MLflow.

Uso:
    poetry run python scripts/evaluate.py
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import torch

from src.config.settings import get_settings
from src.factories.model_factory import ModelFactory


def load_test_tensors(processed_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Carrega os arrays de features de teste como tensores PyTorch.

    Args:
        processed_dir: Pasta com os arquivos gerados pelo feature_engineering.

    Returns:
        Tupla ``(inputs, targets)`` prontos para inferência.
    """
    features = np.load(processed_dir / "test_features.npz")
    inputs = torch.tensor(
        np.stack([features["user_idx"], features["item_idx"]], axis=1), dtype=torch.long
    )
    targets = torch.tensor(features["rating"], dtype=torch.float32)
    return inputs, targets


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Calcula métricas de erro entre previsões e valores reais.

    Args:
        predictions: Tensor com os ratings previstos pelo modelo.
        targets: Tensor com os ratings reais.

    Returns:
        Dicionário com RMSE, MAE e MSE.
    """
    errors = predictions - targets
    mse = torch.mean(errors**2).item()
    return {
        "test_mse": mse,
        "test_rmse": mse**0.5,
        "test_mae": torch.mean(torch.abs(errors)).item(),
    }


def main() -> None:
    """Executa a avaliação do modelo treinado e registra as métricas."""
    settings = get_settings()
    processed_dir = Path(settings.data_processed_path)
    metadata = json.loads((processed_dir / "feature_metadata.json").read_text())

    model = ModelFactory().create_model(
        model_type="embedding",
        params={
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "embedding_dim": 32,
        },
    )
    model_path = Path(settings.models_path) / "embedding_recommender.pt"
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    inputs, targets = load_test_tensors(processed_dir)
    with torch.no_grad():
        predictions = model.forward(inputs)

    metrics = compute_metrics(predictions, targets)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="evaluate-embedding-recommender"):
        mlflow.log_metrics(metrics)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"RMSE: {metrics['test_rmse']:.4f} | MAE: {metrics['test_mae']:.4f}")


if __name__ == "__main__":
    main()
