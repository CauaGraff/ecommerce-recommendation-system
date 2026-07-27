"""Interfaces (contratos) para modelos de recomendação."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class RecommenderModel(ABC):
    """Contrato para qualquer modelo de recomendação da aplicação.

    Attributes:
        name: Nome identificador do modelo, usado em logs e no MLflow.
    """

    name: str

    @abstractmethod
    def forward(self, inputs: Tensor) -> Tensor:
        """Executa a passagem direta (forward pass) do modelo.

        Args:
            inputs: Tensor de entrada com as features do usuário/produto.

        Returns:
            Tensor com os scores de recomendação previstos pelo modelo.
        """
        raise NotImplementedError

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Retorna a configuração do modelo para fins de logging.

        Returns:
            Dicionário serializável com os hiperparâmetros do modelo,
            usado para registrar o experimento no MLflow.
        """
        raise NotImplementedError


class TorchRecommenderModel(RecommenderModel, torch.nn.Module):
    """Classe base para modelos de recomendação implementados em PyTorch."""

    def __init__(self, name: str) -> None:
        """Inicializa a classe base.

        Args:
            name: Nome identificador do modelo.
        """
        torch.nn.Module.__init__(self)
        self.name = name
