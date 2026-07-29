import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow.tracking import MlflowClient

from src.config.settings import get_settings
from src.evaluation.metrics import compute_regression_metrics
from src.factories.model_factory import ModelFactory

REGISTERED_MODEL_NAME = "ecommerce-recommender"
PRODUCTION_RMSE_THRESHOLD = 1.2


def load_test_arrays(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega os arrays de features de teste.

    Args:
        processed_dir: Pasta com os dados processados.

    Returns:
        Tupla ``(user_item_idx, ratings)`` do conjunto de teste.
    """
    features = np.load(processed_dir / "test_features.npz")
    user_item_idx = np.stack([features["user_idx"], features["item_idx"]], axis=1)
    return user_item_idx, features["rating"]


def load_trained_model(processed_dir: Path, models_path: Path, metadata: dict):
    """Reconstrói o modelo via factory e carrega os pesos treinados.

    Args:
        processed_dir: Pasta com os dados processados (não utilizada
            diretamente, mantida por simetria com outras funções do módulo).
        models_path: Pasta onde o modelo treinado foi salvo.
        metadata: Metadados de features (num_users, num_items).

    Returns:
        O modelo carregado, em modo de avaliação.
    """
    model = ModelFactory().create_model(
        model_type="embedding",
        params={
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "embedding_dim": 32,
        },
    )
    model.load_state_dict(torch.load(models_path / "embedding_recommender.pt", weights_only=True))
    model.eval()
    return model


def promote_if_qualified(rmse: float, threshold: float) -> str:
    """Promove a última versão do modelo no Registry, se qualificada.

    Args:
        rmse: RMSE obtido pelo modelo no conjunto de teste.
        threshold: Limiar máximo de RMSE aceito para produção.

    Returns:
        A stage final atribuída à versão ("Production" ou "Staging").
    """
    client = MlflowClient()
    latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Staging"])[0]

    target_stage = "Production" if rmse <= threshold else "Staging"
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest_version.version,
        stage=target_stage,
        archive_existing_versions=(target_stage == "Production"),
    )
    return target_stage


def main() -> None:
    """Executa a avaliação do modelo treinado e a promoção no Registry."""
    settings = get_settings()
    processed_dir = Path(settings.data_processed_path)
    metadata = json.loads((processed_dir / "feature_metadata.json").read_text())

    model = load_trained_model(processed_dir, Path(settings.models_path), metadata)
    user_item_idx, targets = load_test_arrays(processed_dir)

    with torch.no_grad():
        predictions = model.forward(torch.tensor(user_item_idx, dtype=torch.long)).numpy()

    metrics = compute_regression_metrics(predictions, targets)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name="evaluate-embedding-recommender"):
        mlflow.log_metrics({f"test_{key}": value for key, value in metrics.items()})

    final_stage = promote_if_qualified(metrics["rmse"], PRODUCTION_RMSE_THRESHOLD)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R2: {metrics['r2']:.4f}")
    print(f"Modelo promovido para: {final_stage}")


if __name__ == "__main__":
    main()
