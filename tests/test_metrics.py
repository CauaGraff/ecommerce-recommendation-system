"""Testes unitários para as métricas de avaliação."""

import numpy as np

from src.evaluation.metrics import compute_regression_metrics


class TestComputeRegressionMetrics:
    """Agrupa os testes do cálculo de métricas de regressão."""

    def test_returns_four_required_metrics(self) -> None:
        """O retorno deve conter exatamente as 4 métricas exigidas."""
        predictions = np.array([3.0, 4.0, 2.0])
        targets = np.array([3.0, 4.0, 2.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert set(metrics.keys()) == {"mse", "rmse", "mae", "r2"}

    def test_perfect_predictions_yield_zero_error(self) -> None:
        """Previsões perfeitas devem zerar MSE, RMSE e MAE, e dar R2 = 1."""
        predictions = np.array([5.0, 3.0, 1.0])
        targets = np.array([5.0, 3.0, 1.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["mae"] == 0.0
        assert metrics["r2"] == 1.0

    def test_rmse_is_square_root_of_mse(self) -> None:
        """RMSE deve ser sempre a raiz quadrada do MSE retornado."""
        predictions = np.array([4.0, 1.0, 2.0])
        targets = np.array([3.0, 2.0, 5.0])

        metrics = compute_regression_metrics(predictions, targets)

        assert np.isclose(metrics["rmse"], metrics["mse"] ** 0.5)
