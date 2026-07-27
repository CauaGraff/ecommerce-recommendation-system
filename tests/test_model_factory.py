import pytest
from factories.model_factory import ModelFactory, UnknownModelTypeError
from models.mlp_model import EmbeddingRecommender, MLPRecommender


class TestModelFactory:
    """Agrupa os testes relacionados à criação de modelos via factory."""

    def test_create_mlp_model_returns_mlp_instance(self) -> None:
        """A factory deve retornar um MLPRecommender ao pedir tipo 'mlp'."""
        factory = ModelFactory()

        model = factory.create_model(
            model_type="mlp",
            params={"input_dim": 10, "hidden_dims": [16, 8], "output_dim": 1},
        )

        assert isinstance(model, MLPRecommender)
        assert model.input_dim == 10

    def test_create_embedding_model_returns_embedding_instance(self) -> None:
        """A factory deve retornar um EmbeddingRecommender ao pedir 'embedding'."""
        factory = ModelFactory()

        model = factory.create_model(
            model_type="embedding",
            params={"num_users": 100, "num_items": 50, "embedding_dim": 16},
        )

        assert isinstance(model, EmbeddingRecommender)
        assert model.embedding_dim == 16

    def test_create_unknown_model_type_raises_error(self) -> None:
        """Um tipo de modelo não registrado deve levantar UnknownModelTypeError."""
        factory = ModelFactory()

        with pytest.raises(UnknownModelTypeError):
            factory.create_model(model_type="does_not_exist", params={})
