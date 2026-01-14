"""
Dataset schema enforcement.

Defines expected columns and ensures consistency across datasets.
"""

import pandas as pd


EXPECTED_COLUMNS = [
    "Name",
    "Country",
    "Revenue_M",
    "Sector",
    "LinkedIn URL",
    "Company Type",
    "Main Broker",
    "Main Insurer",
]


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure that all expected columns exist in the DataFrame.

    Missing columns are added with None values.
    """
    df = df.copy()
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df
