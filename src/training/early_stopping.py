class EarlyStopping:
    """Interrompe o treino quando a perda de validação para de melhorar.

    Monitora a perda de validação a cada época; se ela não melhorar por
    ``patience`` épocas consecutivas, sinaliza que o treino deve parar.
    Isso evita overfitting e desperdício de tempo de treino.

    Attributes:
        patience: Quantidade de épocas sem melhora toleradas antes de parar.
        min_delta: Melhora mínima considerada significativa.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        """Inicializa o monitor de early stopping.

        Args:
            patience: Épocas sem melhora toleradas antes de sinalizar parada.
            min_delta: Diferença mínima para considerar uma melhora real.
        """
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss = float("inf")
        self._epochs_without_improvement = 0

    @property
    def best_loss(self) -> float:
        """Retorna a menor perda de validação observada até agora.

        Returns:
            O melhor valor de perda de validação registrado.
        """
        return self._best_loss

    def step(self, validation_loss: float) -> bool:
        """Avalia a perda de validação da época atual.

        Args:
            validation_loss: Perda de validação medida na época atual.

        Returns:
            True se o treino deve ser interrompido, False caso contrário.
        """
        if validation_loss < self._best_loss - self.min_delta:
            self._best_loss = validation_loss
            self._epochs_without_improvement = 0
            return False

        self._epochs_without_improvement += 1
        return self._epochs_without_improvement >= self.patience
