from __future__ import annotations

from typing import Mapping

import pandas as pd

from market_explorer.labels import zone_label_ui
from market_explorer.notes import company_key


ZONE_LABELS = {
    "france": "France",
    "eu": "Europe",
    "eu_fr": "Europe + France",
}

def fmt_money(value: float) -> str:
    return f"{value:,.1f} M$".replace(",", " ")

def format_zone_option(value: str, labels: Mapping[str, str] | None = None) -> str:
    normalized = str(value).strip().lower()
    lookup = labels or ZONE_LABELS
    return lookup.get(normalized, zone_label_ui(value))

def get_company_context_row(row: pd.Series) -> dict:
    company_id = row.get("Company ID")
    if pd.isna(company_id) or str(company_id).strip() == "":
        company_id = company_key(str(row["Name"]), str(row.get("Country", "")))
    return {
        "company_id": str(company_id),
        "company_name": str(row["Name"]),
        "country": str(row.get("Country", "")),
    }

def company_label(row: pd.Series) -> str:
    label_country = str(row.get("Country", "")).strip()
    return f"{row['Name']} ({label_country})" if label_country else str(row["Name"])

def compute_bp_simple(
    base_rev_y1: float,
    market_growth: float,
    direct_rate: float,
    take_rate: float,
    price_rate: float,
    neat_commission: float,
    years: int = 5,
) -> pd.DataFrame:
    rows = []
    for y in range(1,years+1):
        hotel_rev = hotel_rev_y1 * ((1 + market_growth) ** (y - 1))
        direct_rev = hotel_rev * direct_rate
        premium = direct_rev * take_rate * price_rate
        neat_rev = premium * neat_commission
        rows.append(
            {
                "Year": f"Year {y}",
                "Hotel revenue (M$)": hotel_rev,
                "Direct revenue (M$)": direct_rev,
                "Premium (M$)": premium,
                "Neat revenue (M$)": neat_rev,
            }
        )
    return pd.DataFrame(rows)