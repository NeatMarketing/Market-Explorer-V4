"""
Analytics & business logic.

No UI. No labels. Only numbers and facts.
"""

import pandas as pd
from typing import Optional, Iterable

def apply_filters(
    df: pd.DataFrame,
    revenue_min_m: Optional[float] = None,
    revenue_max_m: Optional[float] = None,
    country: Optional[Iterable[str]] = None,
    company_type: Optional[Iterable[str]] = None,
    sector: Optional[Iterable[str]] = None,
    revenue_col: str = "Revenue_M",
    country_col: str = "Country",
    company_type_col: str = "Company Type",
    sector_col: str = "Sector",
) -> pd.DataFrame:
    """
    Filter dataframe based on UI filters.
    All filters are optional.
    - revenue_* are in millions (M)
    - country/company_type/sector accept list-like or single string
    """
    out = df.copy()

    # Revenue filter
    if revenue_col in out.columns:
        r = pd.to_numeric(out[revenue_col], errors="coerce")
        if revenue_min_m is not None:
            out = out[r >= float(revenue_min_m)]
            r = pd.to_numeric(out[revenue_col], errors="coerce")
        if revenue_max_m is not None:
            out = out[r <= float(revenue_max_m)]
    else:
        # if UI passes revenue filters but col missing -> explicit error
        if revenue_min_m is not None or revenue_max_m is not None:
            raise KeyError(f"Missing revenue column '{revenue_col}'")

    def _norm_list(x):
        if x is None:
            return None
        if isinstance(x, str):
            return {x}
        return set(list(x))

    # Categorical filters
    cset = _norm_list(country)
    if cset and country_col in out.columns:
        out = out[out[country_col].isin(cset)]

    tset = _norm_list(company_type)
    if tset and company_type_col in out.columns:
        out = out[out[company_type_col].isin(tset)]

    sset = _norm_list(sector)
    if sset and sector_col in out.columns:
        out = out[out[sector_col].isin(sset)]

    return out

def compute_kpis(df: pd.DataFrame, revenue_col: str = "Revenue_M", country_col: str = "Country") -> dict:
    """
    KPI contract expected by pages/1_Market_Explorer.py
    Keys used in the UI:
      - companies
      - total_rev_m
      - median_rev_m
      - countries
      (optionnel: max_rev_m)
    """
    if df is None or len(df) == 0:
        return {
            "companies": 0,
            "total_rev_m": 0.0,
            "median_rev_m": 0.0,
            "countries": 0,
            "max_rev_m": 0.0,
            # aliases (optional)
            "n_companies": 0,
            "n_countries": 0,
        }

    if revenue_col not in df.columns:
        raise KeyError(f"Missing revenue column '{revenue_col}'")

    rev = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0)

    companies = int(len(df))
    total_rev_m = float(rev.sum())
    median_rev_m = float(rev.median())
    max_rev_m = float(rev.max())

    countries = int(df[country_col].nunique(dropna=True)) if country_col in df.columns else 0

    return {
        # keys expected by the page
        "companies": companies,
        "total_rev_m": total_rev_m,
        "median_rev_m": median_rev_m,
        "countries": countries,
        "max_rev_m": max_rev_m,

        # aliases (optional but useful)
        "n_companies": companies,
        "n_countries": countries,
    }


def top_companies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return top companies by revenue."""
    return df.sort_values("Revenue_M", ascending=False).head(n)


def top_by_country(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate revenue by country."""
    return (
        df.groupby("Country", dropna=True)["Revenue_M"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def compute_insights(df: pd.DataFrame) -> dict:
    """
    Generate simple business insights in structured form.
    """
    
    if df.empty:
        return {}

    total_rev = df["Revenue_M"].sum(skipna=True)
    top5 = top_companies(df, n=5)
    top5_share = top5["Revenue_M"].sum(skipna=True) / total_rev if total_rev else 0

    by_country = top_by_country(df)
    top_country = None
    top_country_share = 0.0
    if not by_country.empty and total_rev:
        top_country = by_country.iloc[0]["Country"]
        top_country_share = float(by_country.iloc[0]["Revenue_M"]) / total_rev

    return {
        "top5_share_pct": round(top5_share * 100, 1),
        "top_country": top_country,
        "top_country_share_pct": round(top_country_share * 100, 1),
    }
