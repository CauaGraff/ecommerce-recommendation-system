from typing import Any
import torch
from torch import Tensor, nn
from src.models.base import TorchRecommenderModel

class MLPRecommender(TorchRecommenderModel):
    """Rede neural feed-forward para recomendação de produtos.

    Attributes:
        input_dim: Dimensão do vetor de entrada.
        hidden_dims: Dimensões das camadas ocultas.
        output_dim: Dimensão da saída (nº de produtos candidatos ou score único).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.2,
    ) -> None:
        """Inicializa a arquitetura do MLP.

        Args:
            input_dim: Dimensão do vetor de features de entrada.
            hidden_dims: Lista com o tamanho de cada camada oculta.
            output_dim: Dimensão da saída da rede.
            dropout: Probabilidade de dropout aplicada entre as camadas.
        """
        super().__init__(name="mlp_recommender")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self._network = self._build_network()

    def _build_network(self) -> nn.Sequential:
        """Monta a sequência de camadas lineares da rede.

        Returns:
            Um ``nn.Sequential`` com camadas Linear, ReLU e Dropout
            intercaladas, terminando na camada de saída.
        """
        layers: list[nn.Module] = []
        dims = [self.input_dim, *self.hidden_dims]

        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(dims[-1], self.output_dim))
        return nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        """Executa a passagem direta pela rede.

        Args:
            inputs: Tensor de shape ``(batch_size, input_dim)``.

        Returns:
            Tensor de shape ``(batch_size, output_dim)`` com os scores.
        """
        return self._network(inputs)

    def get_config(self) -> dict[str, Any]:
        """Retorna os hiperparâmetros do modelo para logging no MLflow.

        Returns:
            Dicionário com input_dim, hidden_dims, output_dim e dropout.
        """
        return {
            "model_type": self.name,
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "dropout": self.dropout,
        }


class EmbeddingRecommender(TorchRecommenderModel):
    """Modelo de recomendação baseado em embeddings de usuário e item.

    Attributes:
        num_users: Quantidade de usuários distintos no dataset.
        num_items: Quantidade de produtos distintos no dataset.
        embedding_dim: Dimensão dos vetores de embedding.
    """

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 32) -> None:
        """Inicializa as tabelas de embedding de usuário e item.

        Args:
            num_users: Quantidade de usuários distintos.
            num_items: Quantidade de produtos distintos.
            embedding_dim: Dimensão do vetor de embedding.
        """
        super().__init__(name="embedding_recommender")
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        """Calcula a afinidade usuário-item via produto escalar.

        Args:
            inputs: Tensor de shape ``(batch_size, 2)`` com pares
                ``(user_id, item_id)``.

        Returns:
            Tensor de shape ``(batch_size,)`` com o score de afinidade.
        """
        user_ids = inputs[:, 0]
        item_ids = inputs[:, 1]
        user_vecs = self.user_embedding(user_ids)
        item_vecs = self.item_embedding(item_ids)
        return torch.sum(user_vecs * item_vecs, dim=1)

    def get_config(self) -> dict[str, Any]:
        """Retorna os hiperparâmetros do modelo para logging no MLflow.

        Returns:
            Dicionário com num_users, num_items e embedding_dim.
        """
        return {
            "model_type": self.name,
            "num_users": self.num_users,
            "num_items": self.num_items,
            "embedding_dim": self.embedding_dim,
        }
