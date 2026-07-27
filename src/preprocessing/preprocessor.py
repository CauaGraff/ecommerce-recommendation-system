import numpy as np
from numpy.typing import NDArray
from preprocessing.base import PreprocessingStrategy


class Preprocessor:
    """Aplica uma estratégia de pré-processamento configurável aos dados.

    Attributes:
        strategy: A estratégia de pré-processamento em uso.
    """

    def __init__(self, strategy: PreprocessingStrategy) -> None:
        """Inicializa o preprocessor com a estratégia desejada.

        Args:
            strategy: Instância concreta de ``PreprocessingStrategy``.
        """
        self.strategy = strategy

    def fit_transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ajusta a estratégia aos dados e já retorna os dados transformados.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            Matriz de features transformada.
        """
        self.strategy.fit(data)
        return self.strategy.transform(data)

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transforma novos dados usando a estratégia já ajustada.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            Matriz de features transformada.
        """
        return self.strategy.transform(data)
