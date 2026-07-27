from pathlib import Path
from config.settings import Settings

"""Testes unitários para o carregamento de configurações."""
class TestSettings:
    """Agrupa os testes de carregamento e defaults de Settings."""

    def test_default_values_are_used_when_no_env_is_set(self, monkeypatch) -> None:
        """Sem variáveis de ambiente definidas, os defaults devem valer."""
        monkeypatch.delenv("MODEL_TYPE", raising=False)
        monkeypatch.delenv("RANDOM_SEED", raising=False)

        settings = Settings(_env_file=None)

        assert settings.model_type == "mlp"
        assert settings.random_seed == 42
        assert settings.models_path == Path("models")

    def test_environment_variable_overrides_default(self, monkeypatch) -> None:
        """Uma variável de ambiente definida deve sobrescrever o default."""
        monkeypatch.setenv("MODEL_TYPE", "embedding")
        monkeypatch.setenv("RANDOM_SEED", "7")

        settings = Settings(_env_file=None)

        assert settings.model_type == "embedding"
        assert settings.random_seed == 7
