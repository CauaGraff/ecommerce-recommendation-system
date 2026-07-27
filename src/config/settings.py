from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações centrais do sistema de recomendação.

    Attributes:
        mlflow_tracking_uri: Endereço do servidor de tracking do MLflow.
        mlflow_experiment_name: Nome do experimento no MLflow.
        model_type: Tipo de modelo padrão a ser treinado ("mlp" ou "embedding").
        random_seed: Seed usada para tornar os experimentos reprodutíveis.
        data_raw_path: Caminho para os dados brutos (versionados via DVC).
        data_processed_path: Caminho para os dados já processados.
        models_path: Caminho onde modelos treinados são salvos.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ecommerce-recommender"
    model_type: str = "mlp"
    random_seed: int = 42
    data_raw_path: Path = Path("data/raw")
    data_processed_path: Path = Path("data/processed")
    models_path: Path = Path("models")


@lru_cache
def get_settings() -> Settings:
    """Retorna uma instância única (cacheada) de ``Settings``.

    Usar ``lru_cache`` evita reler e revalidar o ``.env`` toda vez que a
    configuração é solicitada em algum ponto do código.

    Returns:
        A instância cacheada de ``Settings``.
    """
    return Settings()
