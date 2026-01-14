"""
Label & formatting helpers.

This module contains ONLY presentation logic:
- converting slugs to readable labels
- mapping internal codes to UI-friendly strings

No business logic here.
"""

from typing import Optional

def titleize_slug(value: Optional[str]) -> str:
    if not value:
        return ""
    value = str(value).strip().replace("_", " ").replace("-", " ")
    return " ".join(word.capitalize() for word in value.split())


def market_label(market: Optional[str]) -> str:
    return titleize_slug(market)


def zone_label(zone: Optional[str]) -> str:
    return titleize_slug(zone)


def zone_label_ui(zones) -> str:
    if not zones:
        return "All Zones"

    def format_zone(value) -> str:
        normalized = str(value).strip().lower()
        if normalized == "france":
            return "France"
        if normalized == "eu":
            return "Europe"
        if normalized in {"eu_fr", "fr+eu", "fr_eu"}:
            return "Europe + France"
        return titleize_slug(normalized)

    if isinstance(zones, (list, tuple, set)):
        return " + ".join(format_zone(z) for z in zones)

    return format_zone(zones)


def zones_in_scope_from_ui(selection) -> list[str]:
    """
    Parse a UI zone selection into a list of zone codes.
    Supports the current UI codes: "france", "eu", "eu_fr".
    """
    if not selection:
        return []

    if isinstance(selection, list):
        return [str(z).strip().lower() for z in selection if str(z).strip()]

    sel = str(selection).strip().lower()

    # Current UI codes
    if sel == "eu":
        return ["europe"]
    if sel in {"eu_fr", "fr+eu", "fr_eu"}:
        return ["france", "europe"]

    # Generic parsing for human readable (ex: "France + Europe")
    parts = [p.strip().lower() for p in sel.split("+")]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        if p == "eu":
            p = "europe"
        out.append(p)
    return out