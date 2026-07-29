import json
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config.settings import get_settings
from src.factories.model_factory import ModelFactory
from src.training.early_stopping import EarlyStopping
from src.training.neural_trainer import NeuralNetworkTrainer

MAX_EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.01
EMBEDDING_DIM = 32
VALIDATION_SIZE = 0.1
REGISTERED_MODEL_NAME = "ecommerce-recommender"


def load_train_val_datasets(
    processed_dir: Path, validation_size: float, seed: int
) -> tuple[TensorDataset, TensorDataset]:
    """Carrega as features de treino e separa uma fatia para validação.

    Args:
        processed_dir: Pasta com os arquivos gerados pelo feature_engineering.
        validation_size: Fração do treino reservada para validação.
        seed: Seed para reprodutibilidade do split.

    Returns:
        Tupla ``(train_dataset, validation_dataset)``.
    """
    features = np.load(processed_dir / "train_features.npz")
    inputs = np.stack([features["user_idx"], features["item_idx"]], axis=1)
    targets = features["rating"]

    train_x, val_x, train_y, val_y = train_test_split(
        inputs, targets, test_size=validation_size, random_state=seed
    )

    train_dataset = TensorDataset(
        torch.tensor(train_x, dtype=torch.long), torch.tensor(train_y, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(val_x, dtype=torch.long), torch.tensor(val_y, dtype=torch.float32)
    )
    return train_dataset, val_dataset


def build_trainer(
    processed_dir: Path, metadata: dict, seed: int
) -> tuple[NeuralNetworkTrainer, ModelFactory]:
    """Monta o modelo, otimizador e o treinador com early stopping.

    Args:
        processed_dir: Pasta com os dados processados.
        metadata: Metadados de features (num_users, num_items).
        seed: Seed para reprodutibilidade.

    Returns:
        Tupla ``(trainer, factory)`` prontos para o treino.
    """
    train_dataset, val_dataset = load_train_val_datasets(processed_dir, VALIDATION_SIZE, seed)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    factory = ModelFactory()
    model = factory.create_model(
        model_type="embedding",
        params={
            "num_users": metadata["num_users"],
            "num_items": metadata["num_items"],
            "embedding_dim": EMBEDDING_DIM,
        },
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(patience=PATIENCE)

    trainer = NeuralNetworkTrainer(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        optimizer=optimizer,
        loss_fn=nn.MSELoss(),
        max_epochs=MAX_EPOCHS,
        early_stopping=early_stopping,
    )
    return trainer, factory


def main() -> None:
    """Executa o treino completo com early stopping e publica no Registry."""
    settings = get_settings()
    torch.manual_seed(settings.random_seed)

    processed_dir = Path(settings.data_processed_path)
    metadata = json.loads((processed_dir / "feature_metadata.json").read_text())

    trainer, _ = build_trainer(processed_dir, metadata, settings.random_seed)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name="train-embedding-recommender"):
        mlflow.log_params(
            {
                **trainer.model.get_config(),
                "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE,
                "learning_rate": LEARNING_RATE,
                "validation_size": VALIDATION_SIZE,
            }
        )

        summary = trainer.fit()

        mlflow.log_metric("best_validation_mse", summary.best_validation_loss)
        mlflow.log_metric("epochs_trained", len(summary.train_losses))
        mlflow.log_param("stopped_early", summary.stopped_early)

        models_path = Path(settings.models_path)
        models_path.mkdir(parents=True, exist_ok=True)
        torch.save(trainer.model.state_dict(), models_path / "embedding_recommender.pt")

        # Exemplo de entrada compatível com o formato de tupla/tensor esperado pelo modelo
        sample_input = (
            torch.tensor([0], dtype=torch.long),  # Exemplo de user_idx
            torch.tensor([0], dtype=torch.long),  # Exemplo de item_idx
        )

        mlflow.pytorch.log_model(
            pytorch_model=trainer.model,
            name="model",
            input_example=sample_input,
            serialization_format="pickle",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(
            f"Treino concluído em {len(summary.train_losses)} épocas "
            f"(early stop: {summary.stopped_early}). "
            f"Melhor MSE de validação: {summary.best_validation_loss:.4f}"
        )


if __name__ == "__main__":
    main()
