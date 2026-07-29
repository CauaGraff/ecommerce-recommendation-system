"""Stage 2 do pipeline DVC: engenharia de features.

Converte ``user_id`` e ``item_id`` em índices contíguos (necessário para as
camadas de embedding do modelo) e salva os arrays resultantes junto com os
encoders ajustados, para reuso na inferência.

Uso:
    poetry run python scripts/feature_engineering.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config.settings import get_settings
from src.data.encoders import IdEncoder


def encode_ids(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[IdEncoder, IdEncoder]:
    """Ajusta encoders de usuário e item usando treino + teste combinados.

    Ajustar nos dois conjuntos garante que nenhum ID do teste fique fora do
    vocabulário aprendido (evita erro na camada de embedding).

    Args:
        train_df: DataFrame de treino com colunas ``userId`` e ``movieId``.
        test_df: DataFrame de teste com as mesmas colunas.

    Returns:
        Tupla ``(user_encoder, item_encoder)`` já ajustados.
    """
    all_users = pd.concat([train_df["userId"], test_df["userId"]]).to_numpy()
    all_items = pd.concat([train_df["movieId"], test_df["movieId"]]).to_numpy()

    user_encoder = IdEncoder().fit(all_users)
    item_encoder = IdEncoder().fit(all_items)
    return user_encoder, item_encoder


def build_feature_arrays(
    df: pd.DataFrame, user_encoder: IdEncoder, item_encoder: IdEncoder
) -> dict[str, np.ndarray]:
    """Constrói os arrays de features codificadas a partir de um dataframe.

    Args:
        df: DataFrame com colunas ``userId``, ``movieId`` e ``rating``.
        user_encoder: Encoder de usuários já ajustado.
        item_encoder: Encoder de itens já ajustado.

    Returns:
        Dicionário com arrays numpy ``user_idx``, ``item_idx`` e ``rating``.
    """
    return {
        "user_idx": user_encoder.transform(df["userId"].to_numpy()),
        "item_idx": item_encoder.transform(df["movieId"].to_numpy()),
        "rating": df["rating"].to_numpy(dtype=np.float32),
    }


def main() -> None:
    """Executa o pipeline de feature engineering completo."""
    settings = get_settings()
    processed_dir = Path(settings.data_processed_path)

    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")

    user_encoder, item_encoder = encode_ids(train_df, test_df)

    train_features = build_feature_arrays(train_df, user_encoder, item_encoder)
    test_features = build_feature_arrays(test_df, user_encoder, item_encoder)

    np.savez(processed_dir / "train_features.npz", **train_features)
    np.savez(processed_dir / "test_features.npz", **test_features)
    joblib.dump(user_encoder, processed_dir / "user_encoder.joblib")
    joblib.dump(item_encoder, processed_dir / "item_encoder.joblib")

    metadata = {
        "num_users": user_encoder.num_classes,
        "num_items": item_encoder.num_classes,
    }
    (processed_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Usuários únicos: {metadata['num_users']} | Itens únicos: {metadata['num_items']}")


if __name__ == "__main__":
    main()
