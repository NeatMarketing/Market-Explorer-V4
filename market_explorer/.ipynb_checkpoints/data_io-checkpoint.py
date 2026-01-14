"""
Data loading & export utilities.
"""

from pathlib import Path
import pandas as pd
from .schema import enforce_schema


def load_dataset(path: Path, zone: str | None = None) -> pd.DataFrame:
    """
    Load a CSV dataset and enforce schema.

    If `zone` is provided and the CSV contains a `zone` column, rows are filtered
    to that zone. This supports monolithic files like `..._zone_cleaned.csv`.
    """
    df = pd.read_csv(path)
    df = enforce_schema(df)

    if zone and "zone" in df.columns:
        z = str(zone).strip().lower()
        df = df[df["zone"].astype(str).str.strip().str.lower() == z]

    return df


def load_panorama(paths: list[Path]) -> pd.DataFrame:
    dfs = [load_dataset(p) for p in paths]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def load_clean_revenue(path: Path) -> pd.DataFrame:
    return load_dataset(path)
