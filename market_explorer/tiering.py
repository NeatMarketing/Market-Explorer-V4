"""
Tiering logic.

- Adds tier columns based on revenue thresholds
- Filters datasets by tier
"""

import pandas as pd


def add_tier(df: pd.DataFrame, t1: float, t2: float, revenue_col: str = "Revenue_M") -> pd.DataFrame:
    """
    Adds:
      - 'Tier' (internal codes): Tier 1 / Tier 2 / Tier 3
      - 'Market Tier' (UI labels): Large Market / Mid-Market / Low-Market
    """
    out = df.copy()
    if revenue_col not in out.columns:
        raise KeyError(f"Missing column '{revenue_col}' in dataframe.")

    r = pd.to_numeric(out[revenue_col], errors="coerce").fillna(0)

    # Internal tier codes
    out["Tier"] = "Tier 3"
    out.loc[r >= t2, "Tier"] = "Tier 2"
    out.loc[r >= t1, "Tier"] = "Tier 1"

    # UI labels (what you want to show)
    tier_ui = {
        "Tier 1": "Large Market",
        "Tier 2": "Mid-Market",
        "Tier 3": "Low-Market",
    }
    out["Market Tier"] = out["Tier"].map(tier_ui)

    return out


def filter_by_tier(df: pd.DataFrame, tier: str) -> pd.DataFrame:
    """
    Accepts either:
      - internal codes: "Tier 1", "Tier 2", "Tier 3"
      - UI labels: "Large Market", "Mid-Market", "Low-Market"
      - "All" / "All Markets" => no filter
    """
    if tier in (None, "All", "All Markets"):
        return df

    tier_ui = {
        "Tier 1": "Large Market",
        "Tier 2": "Mid-Market",
        "Tier 3": "Low-Market",
    }

    # If the caller gave an internal code, filter on 'Tier'
    if tier in tier_ui and "Tier" in df.columns:
        return df[df["Tier"] == tier]

    # Otherwise treat it as a UI label and filter on 'Market Tier'
    if "Market Tier" in df.columns:
        return df[df["Market Tier"] == tier]

    return df
