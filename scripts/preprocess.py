from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.loader import load_raw_ratings


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas e valores nulos do dataframe de avaliações.

    Args:
        ratings: DataFrame bruto de avaliações.

    Returns:
        DataFrame limpo, sem duplicatas ou valores nulos.
    """
    return ratings.drop_duplicates().dropna()


def split_train_test(
    ratings: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide as avaliações em conjuntos de treino e teste.

    Args:
        ratings: DataFrame limpo de avaliações.
        test_size: Fração dos dados reservada para teste (ex.: 0.2).
        seed: Seed para reprodutibilidade da divisão.

    Returns:
        Tupla ``(train_df, test_df)``.
    """
    return train_test_split(ratings, test_size=test_size, random_state=seed)


def main() -> None:
    """Executa o pipeline de pré-processamento completo."""
    settings = get_settings()

    raw_ratings = load_raw_ratings(settings.data_raw_path)
    clean = clean_ratings(raw_ratings)
    train_df, test_df = split_train_test(clean, test_size=0.2, seed=settings.random_seed)

    output_dir = Path(settings.data_processed_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"Treino: {len(train_df)} linhas | Teste: {len(test_df)} linhas")
    print(f"Salvo em {output_dir}")


if __name__ == "__main__":
    main()
