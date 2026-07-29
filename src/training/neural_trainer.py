from collections.abc import Callable
import mlflow
import torch
from torch.utils.data import DataLoader
from src.models.base import RecommenderModel
from src.training.base import Trainer
from src.training.early_stopping import EarlyStopping


class NeuralNetworkTrainer(Trainer):
    """Treina um ``RecommenderModel`` em PyTorch, com logging no MLflow.

    Implementa os passos variáveis do Template Method (``train_epoch`` e
    ``validate_epoch``) usando um otimizador e uma função de perda do
    PyTorch, e usa o hook ``on_epoch_end`` para registrar métricas por
    época diretamente no MLflow.
    """

    def __init__(
        self,
        model: RecommenderModel,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        max_epochs: int,
        early_stopping: EarlyStopping,
    ) -> None:
        """Inicializa o treinador neural.

        Args:
            model: Modelo a ser treinado (implementa ``RecommenderModel``).
            train_loader: DataLoader com os batches de treino.
            validation_loader: DataLoader com os batches de validação.
            optimizer: Otimizador já configurado com os parâmetros do modelo.
            loss_fn: Função de perda (ex.: ``torch.nn.MSELoss()``).
            max_epochs: Número máximo de épocas.
            early_stopping: Instância de ``EarlyStopping`` já configurada.
        """
        super().__init__(max_epochs=max_epochs, early_stopping=early_stopping)
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_epoch(self) -> float:
        """Executa uma época de treino (forward + backward + step).

        Returns:
            Perda média de treino da época.
        """
        self.model.train()
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            self.optimizer.zero_grad()
            predictions = self.model.forward(inputs)
            loss = self.loss_fn(predictions, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * len(targets)

        return total_loss / len(self.train_loader.dataset)

    def validate_epoch(self) -> float:
        """Executa a validação da época atual, sem atualizar pesos.

        Returns:
            Perda média de validação da época.
        """
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.validation_loader:
                predictions = self.model.forward(inputs)
                loss = self.loss_fn(predictions, targets)
                total_loss += loss.item() * len(targets)

        return total_loss / len(self.validation_loader.dataset)

    def on_epoch_end(self, epoch: int, train_loss: float, validation_loss: float) -> None:
        """Registra as perdas de treino/validação da época no MLflow.

        Args:
            epoch: Número da época concluída.
            train_loss: Perda de treino da época.
            validation_loss: Perda de validação da época.
        """
        mlflow.log_metric("train_mse", train_loss, step=epoch)
        mlflow.log_metric("validation_mse", validation_loss, step=epoch)
        print(f"Época {epoch} - treino: {train_loss:.4f} | validação: {validation_loss:.4f}")
