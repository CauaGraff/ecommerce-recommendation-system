import numpy as np

from src.models.baselines import GlobalAverageBaseline, LinearRegressionBaseline


class TestGlobalAverageBaseline:
    """Agrupa os testes do baseline de média global."""

    def test_predicts_train_mean_for_any_input(self) -> None:
        """Deve prever a média do treino para qualquer par usuário-item."""
        train_ratings = np.array([1.0, 3.0, 5.0])
        baseline = GlobalAverageBaseline()

        baseline.fit(user_item_idx=np.zeros((3, 2)), ratings=train_ratings)
        predictions = baseline.predict(user_item_idx=np.zeros((5, 2)))

        assert np.allclose(predictions, 3.0)
        assert len(predictions) == 5


class TestLinearRegressionBaseline:
    """Agrupa os testes do baseline de regressão linear."""

    def test_fit_predict_returns_correct_shape(self) -> None:
        """A predição deve ter o mesmo número de linhas da entrada."""
        train_x = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        train_y = np.array([1.0, 2.0, 3.0, 4.0])
        baseline = LinearRegressionBaseline()

        baseline.fit(train_x, train_y)
        predictions = baseline.predict(train_x)

        assert predictions.shape == (4,)
