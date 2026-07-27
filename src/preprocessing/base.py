from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class PreprocessingStrategy(ABC):
    """Contrato para uma estratégia de pré-processamento de features.

    Cada estratégia concreta (ex.: padronização, normalização min-max)
    implementa ``fit`` e ``transform`` de forma independente. O código
    cliente (``Preprocessor``) trabalha apenas com esta interface, podendo
    trocar a estratégia em tempo de execução sem alterações estruturais.
    """

    @abstractmethod
    def fit(self, data: NDArray[np.float64]) -> "PreprocessingStrategy":
        """Ajusta os parâmetros internos da estratégia aos dados.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            A própria instância, para permitir encadeamento (method chaining).
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplica a transformação já ajustada aos dados.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            Matriz de features transformada, mesma shape da entrada.
        """
        raise NotImplementedError
