from typing import Protocol
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression


class RatingBaseline(Protocol):
    """Contrato mínimo que qualquer baseline de rating deve cumprir."""

    def fit(self, user_item_idx: NDArray[np.int64], ratings: NDArray[np.float64]) -> None:
        """Ajusta o baseline aos dados de treino."""
        ...

    def predict(self, user_item_idx: NDArray[np.int64]) -> NDArray[np.float64]:
        """Prediz ratings para os pares usuário-item fornecidos."""
        ...


class GlobalAverageBaseline:
    """Baseline trivial: sempre prevê a média global de rating do treino.

    Serve como "piso de sanidade": qualquer modelo mais sofisticado deve
    superar este baseline para justificar sua complexidade.
    """

    def __init__(self) -> None:
        """Inicializa o baseline sem nenhum valor ajustado ainda."""
        self._global_mean: float = 0.0

    def fit(self, user_item_idx: NDArray[np.int64], ratings: NDArray[np.float64]) -> None:
        """Calcula e armazena a média global dos ratings de treino.

        Args:
            user_item_idx: Array de pares (user_idx, item_idx). Não utilizado
                por este baseline, mas mantido para respeitar o contrato.
            ratings: Array de ratings reais de treino.
        """
        self._global_mean = float(np.mean(ratings))

    def predict(self, user_item_idx: NDArray[np.int64]) -> NDArray[np.float64]:
        """Retorna a média global para todos os pares fornecidos.

        Args:
            user_item_idx: Array de pares (user_idx, item_idx).

        Returns:
            Array preenchido com a média global de rating.
        """
        return np.full(shape=len(user_item_idx), fill_value=self._global_mean)


class LinearRegressionBaseline:
    """Baseline de regressão linear sobre os índices de usuário e item.

    Trata ``user_idx`` e ``item_idx`` como features numéricas diretas. É um
    baseline mais forte que a média global, mas ainda muito mais simples
    que uma rede neural com embeddings aprendidos.
    """

    def __init__(self) -> None:
        """Inicializa o modelo de regressão linear interno."""
        self._model = LinearRegression()

    def fit(self, user_item_idx: NDArray[np.int64], ratings: NDArray[np.float64]) -> None:
        """Ajusta a regressão linear aos dados de treino.

        Args:
            user_item_idx: Array de shape ``(n, 2)`` com (user_idx, item_idx).
            ratings: Array de ratings reais de treino.
        """
        self._model.fit(user_item_idx, ratings)

    def predict(self, user_item_idx: NDArray[np.int64]) -> NDArray[np.float64]:
        """Prediz ratings via regressão linear.

        Args:
            user_item_idx: Array de shape ``(n, 2)`` com (user_idx, item_idx).

        Returns:
            Array de ratings previstos.
        """
        return self._model.predict(user_item_idx)
