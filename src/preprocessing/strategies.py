import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from preprocessing.base import PreprocessingStrategy


class StandardScalingStrategy(PreprocessingStrategy):
    """Padroniza features para média 0 e desvio padrão 1.

    Útil quando as features têm distribuições aproximadamente normais e
    escalas muito diferentes entre si.
    """

    def __init__(self) -> None:
        """Inicializa o scaler interno do Scikit-Learn."""
        self._scaler = StandardScaler()

    def fit(self, data: NDArray[np.float64]) -> "StandardScalingStrategy":
        """Calcula média e desvio padrão de cada feature.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            A própria instância, para encadeamento.
        """
        self._scaler.fit(data)
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplica a padronização já ajustada.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            Matriz padronizada (média 0, desvio padrão 1 por coluna).
        """
        return self._scaler.transform(data)


class MinMaxScalingStrategy(PreprocessingStrategy):
    """Reescala features para o intervalo ``[0, 1]``.

    Útil para features sem distribuição normal, ou quando se quer manter
    todos os valores dentro de um intervalo fixo (ex.: entradas de rede
    neural sensíveis à escala).
    """

    def __init__(self) -> None:
        """Inicializa o scaler interno do Scikit-Learn."""
        self._scaler = MinMaxScaler()

    def fit(self, data: NDArray[np.float64]) -> "MinMaxScalingStrategy":
        """Calcula os valores mínimo e máximo de cada feature.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            A própria instância, para encadeamento.
        """
        self._scaler.fit(data)
        return self

    def transform(self, data: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplica o reescalonamento min-max já ajustado.

        Args:
            data: Matriz de features de shape ``(n_amostras, n_features)``.

        Returns:
            Matriz com valores reescalados para o intervalo [0, 1].
        """
        return self._scaler.transform(data)
