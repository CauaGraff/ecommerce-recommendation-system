from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from src.training.early_stopping import EarlyStopping


@dataclass
class TrainingSummary:
    """Resumo do resultado de um treino completo.

    Attributes:
        train_losses: Perda de treino registrada em cada época.
        validation_losses: Perda de validação registrada em cada época.
        stopped_early: Se o treino foi interrompido por early stopping.
        best_validation_loss: Menor perda de validação observada.
    """

    train_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    stopped_early: bool = False
    best_validation_loss: float = float("inf")


class Trainer(ABC):
    """Classe base que define o algoritmo geral de treino (Template Method).

    Subclasses devem implementar ``train_epoch`` e ``validate_epoch``, que
    concentram a lógica específica de forward/backward de cada tipo de
    modelo. O método ``fit`` é o "template" — não deve ser sobrescrito.
    """

    def __init__(self, max_epochs: int, early_stopping: EarlyStopping) -> None:
        """Inicializa o treinador.

        Args:
            max_epochs: Número máximo de épocas a treinar.
            early_stopping: Instância de ``EarlyStopping`` já configurada.
        """
        self.max_epochs = max_epochs
        self.early_stopping = early_stopping

    @abstractmethod
    def train_epoch(self) -> float:
        """Executa uma época de treino e retorna a perda média de treino.

        Returns:
            Perda média de treino da época.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_epoch(self) -> float:
        """Executa a validação da época atual e retorna a perda média.

        Returns:
            Perda média de validação da época.
        """
        raise NotImplementedError

    def on_epoch_end(self, epoch: int, train_loss: float, validation_loss: float) -> None:
        """Hook chamado ao fim de cada época (ex.: para logging externo).

        Implementação padrão não faz nada; subclasses podem sobrescrever
        para integrar com MLflow ou outro sistema de tracking.

        Args:
            epoch: Número da época concluída (começando em 1).
            train_loss: Perda de treino da época.
            validation_loss: Perda de validação da época.
        """

    def fit(self) -> TrainingSummary:
        """Executa o algoritmo completo de treino (o "template").

        A ordem das etapas é fixa: treinar a época, validar, notificar via
        hook, checar early stopping. Isso é o que caracteriza o Template
        Method — o "o quê" é fixo aqui, o "como" fica nas subclasses.

        Returns:
            Um ``TrainingSummary`` com o histórico do treino.
        """
        summary = TrainingSummary()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_epoch()
            validation_loss = self.validate_epoch()

            summary.train_losses.append(train_loss)
            summary.validation_losses.append(validation_loss)
            self.on_epoch_end(epoch, train_loss, validation_loss)

            if self.early_stopping.step(validation_loss):
                summary.stopped_early = True
                break

        summary.best_validation_loss = self.early_stopping.best_loss
        return summary
