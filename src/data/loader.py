from pathlib import Path
import pandas as pd

RATINGS_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def load_raw_ratings(raw_data_path: Path) -> pd.DataFrame:
    """Carrega o arquivo de avaliações do MovieLens.

    Args:
        raw_data_path: Caminho da pasta ``data/raw`` contendo
            ``ratings.csv`` extraída do dataset.

    Returns:
        DataFrame com as colunas ``userId``, ``movieId``, ``rating`` e
        ``timestamp``.

    Raises:
        FileNotFoundError: Se o arquivo ``ratings.csv`` não for encontrado.
    """
    ratings_file = raw_data_path / "ratings.csv"
    if not ratings_file.exists():
        raise FileNotFoundError(
            f"Arquivo {ratings_file} não encontrado. "
        )

    return pd.read_csv(ratings_file)