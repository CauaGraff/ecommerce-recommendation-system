from typing import Any
from models.base import RecommenderModel
from models.mlp_model import EmbeddingRecommender, MLPRecommender


class UnknownModelTypeError(ValueError):
    """Erro lançado quando um tipo de modelo não registrado é solicitado."""


class ModelFactory:
    """Cria instâncias de modelos de recomendação a partir de um tipo textual."""

    _registry: dict[str, type[RecommenderModel]] = {
        "mlp": MLPRecommender,
        "embedding": EmbeddingRecommender,
    }

    def create_model(self, model_type: str, params: dict[str, Any]) -> RecommenderModel:
        """Cria e retorna um modelo de recomendação.

        Args:
            model_type: Chave do modelo desejado (``"mlp"`` ou ``"embedding"``).
            params: Argumentos a serem repassados ao construtor do modelo.

        Returns:
            Uma instância concreta de ``RecommenderModel``.

        Raises:
            UnknownModelTypeError: Se ``model_type`` não estiver registrado.
        """
        model_class = self._registry.get(model_type)
        if model_class is None:
            registered = ", ".join(self._registry.keys())
            raise UnknownModelTypeError(
                f"Tipo de modelo '{model_type}' desconhecido. "
                f"Tipos disponíveis: {registered}."
            )
        return model_class(**params)

    @classmethod
    def register_model(cls, model_type: str, model_class: type[RecommenderModel]) -> None:
        """Registra um novo tipo de modelo na factory em tempo de execução.

        Args:
            model_type: Chave textual que identificará o modelo.
            model_class: Classe concreta do modelo (deve herdar de
                ``RecommenderModel``).
        """
        cls._registry[model_type] = model_class
