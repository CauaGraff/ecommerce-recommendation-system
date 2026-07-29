import numpy as np
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.strategies import (
    MinMaxScalingStrategy,
    StandardScalingStrategy,
)

class TestStandardScalingStrategy:
    """Testes da estratégia de padronização (média 0, desvio padrão 1)."""

    def test_fit_transform_produces_zero_mean(self) -> None:
        """Após fit_transform, cada coluna deve ter média aproximadamente 0."""
        data = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        preprocessor = Preprocessor(strategy=StandardScalingStrategy())

        result = preprocessor.fit_transform(data)

        assert np.allclose(result.mean(axis=0), 0.0, atol=1e-8)


class TestMinMaxScalingStrategy:
    """Testes da estratégia de reescalonamento para o intervalo [0, 1]."""

    def test_fit_transform_scales_to_zero_one_range(self) -> None:
        """Após fit_transform, os valores devem estar entre 0 e 1."""
        data = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        preprocessor = Preprocessor(strategy=MinMaxScalingStrategy())

        result = preprocessor.fit_transform(data)

        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_strategies_are_interchangeable(self) -> None:
        """Trocar a estratégia no Preprocessor não deve exigir outra lógica."""
        data = np.array([[5.0], [10.0], [15.0]])

        result_standard = Preprocessor(StandardScalingStrategy()).fit_transform(data)
        result_minmax = Preprocessor(MinMaxScalingStrategy()).fit_transform(data)

        assert not np.allclose(result_standard, result_minmax)