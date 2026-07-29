import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import LabelEncoder


class IdEncoder:
    """Codifica identificadores categóricos (ex.: userId) em índices inteiros."""

    def __init__(self) -> None:
        """Inicializa o encoder interno."""
        self._encoder = LabelEncoder()

    def fit(self, ids: NDArray[np.int64]) -> "IdEncoder":
        """Aprende o mapeamento de IDs originais para índices contíguos.

        Args:
            ids: Array de identificadores originais.

        Returns:
            A própria instância, para encadeamento.
        """
        self._encoder.fit(ids)
        return self

    def transform(self, ids: NDArray[np.int64]) -> NDArray[np.int64]:
        """Converte IDs originais em índices contíguos já aprendidos.

        Args:
            ids: Array de identificadores originais.

        Returns:
            Array de índices inteiros começando em zero.
        """
        return self._encoder.transform(ids)

    @property
    def num_classes(self) -> int:
        """Retorna a quantidade de identificadores distintos aprendidos.

        Returns:
            Número de classes (IDs únicos) vistas no ``fit``.
        """
        return len(self._encoder.classes_)
