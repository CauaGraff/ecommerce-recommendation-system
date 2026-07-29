"""Stage 3 do pipeline DVC: treino do modelo com tracking no MLflow.

Treina um ``EmbeddingRecommender`` (via ``ModelFactory``) para prever o
rating usuário-item, registrando parâmetros, métricas por época e o
artefato final do modelo no MLflow.

Uso:
    poetry run python scripts/train.py
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import get_settings
from src.factories.model_factory import ModelFactory
from src.models.base import RecommenderModel

EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.01
EMBEDDING_DIM = 32


def load_train_tensors(processed_dir: Path) -> TensorDataset:
    """Carrega os arrays de features de treino como um TensorDataset.

    Args:
        processed_dir: Pasta com os arquivos gerados pelo feature_engineering.

    Returns:
        Dataset PyTorch com pares (user_idx, item_idx) e o rating alvo.
    """
    features = np.load(processed_dir / "train_features.npz")
    inputs = torch.tensor(
        np.stack([features["user_idx"], features["item_idx"]], axis=1), dtype=torch.long
    )
    targets = torch.tensor(features["rating"], dtype=torch.float32)
    return TensorDataset(inputs, targets)


def train_one_epoch(
    model: RecommenderModel, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> float:
    """Executa uma época de treino e retorna a perda média.

    Args:
        model: Modelo de recomendação a ser treinado.
        loader: DataLoader com os batches de treino.
        optimizer: Otimizador já configurado com os parâmetros do modelo.

    Returns:
        Perda (MSE) média da época.
    """
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for inputs, targets in loader:
        optimizer.zero_grad()
        predictions = model.forward(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)

    return total_loss / len(loader.dataset)


def main() -> None:
    """Executa o treino completo com tracking no MLflow."""
    settings = get_settings()
    torch.manual_seed(settings.random_seed)

    processed_dir = Path(settings.data_processed_path)
    metadata = json.loads((processed_dir / "feature_metadata.json").read_text())
    dataset = load_train_tensors(processed_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ModelFactory().create_model(
        model_type="embedding",
        params={
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "embedding_dim": EMBEDDING_DIM,
        },
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="train-embedding-recommender"):
        mlflow.log_params(
            {**model.get_config(), "epochs": EPOCHS, "learning_rate": LEARNING_RATE}
        )

        for epoch in range(1, EPOCHS + 1):
            epoch_loss = train_one_epoch(model, loader, optimizer)
            mlflow.log_metric("train_mse", epoch_loss, step=epoch)
            print(f"Época {epoch}/{EPOCHS} - MSE treino: {epoch_loss:.4f}")

        models_path = Path(settings.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        model_path = models_path / "embedding_recommender.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        print(f"Modelo salvo em {model_path} e registrado no MLflow.")


if __name__ == "__main__":
    main()
