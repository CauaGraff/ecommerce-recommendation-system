"""Testes unitários para o EarlyStopping."""

from src.training.early_stopping import EarlyStopping


class TestEarlyStopping:
    """Agrupa os testes do critério de parada antecipada."""

    def test_does_not_stop_while_loss_improves(self) -> None:
        """Enquanto a perda melhora, o treino não deve ser interrompido."""
        early_stopping = EarlyStopping(patience=2)

        assert early_stopping.step(1.0) is False
        assert early_stopping.step(0.8) is False
        assert early_stopping.step(0.5) is False

    def test_stops_after_patience_epochs_without_improvement(self) -> None:
        """Após 'patience' épocas sem melhora, deve sinalizar parada."""
        early_stopping = EarlyStopping(patience=2, min_delta=1e-4)

        assert early_stopping.step(1.0) is False  # melhora inicial
        assert early_stopping.step(1.0) is False  # 1ª época sem melhora
        assert early_stopping.step(1.0) is True  # 2ª época sem melhora -> para

    def test_best_loss_tracks_minimum_observed(self) -> None:
        """O melhor valor de perda deve refletir o menor já observado."""
        early_stopping = EarlyStopping(patience=3)

        early_stopping.step(1.0)
        early_stopping.step(0.3)
        early_stopping.step(0.9)

        assert early_stopping.best_loss == 0.3
