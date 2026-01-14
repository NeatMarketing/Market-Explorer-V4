"""
Market Explorer — public API.

This package provides a clean, business-oriented interface to:
- discover available market datasets
- load and filter data
- compute KPIs and insights
- manage sales notes

Only stable, high-level functions are exposed here.
"""

# -----------------------
# Dataset discovery
# -----------------------
from .discovery import (
    DatasetInfo,
    DatasetCatalog,
    list_datasets,
)

# -----------------------
# Data loading & export
# -----------------------
from .data_io import (
    load_dataset,
    load_panorama,
    to_csv_bytes,
    load_clean_revenue
)

# -----------------------
# Schema
# -----------------------
from .schema import (
    enforce_schema,
    EXPECTED_COLUMNS,
)

# -----------------------
# Analytics & insights
# -----------------------
from .analytics import (
    apply_filters,
    compute_kpis,
    compute_insights,
    top_companies,
    top_by_country,
)

# -----------------------
# Tiering
# -----------------------
from .tiering import (
    add_tier,
    filter_by_tier,
)

# -----------------------
# Labels / UI helpers
# -----------------------
from .labels import (
    titleize_slug,
    market_label,
    zone_label,
    zone_label_ui,
    zones_in_scope_from_ui,
)

# -----------------------
# Notes (sales / user)
# -----------------------
from .notes import (
    load_notes,
    save_notes,
    reset_notes,
    upsert_note,
    company_key,
)

__all__ = [
    # Discovery
    "DatasetInfo",
    "DatasetCatalog",
    "list_datasets",

    # IO
    "load_dataset",
    "load_panorama",
    "to_csv_bytes",

    # Schema
    "enforce_schema",
    "EXPECTED_COLUMNS",

    # Analytics
    "apply_filters",
    "compute_kpis",
    "compute_insights",
    "top_companies",
    "top_by_country",

    # Tiering
    "add_tier",
    "filter_by_tier",

    # Labels
    "titleize_slug",
    "market_label",
    "zone_label",
    "zone_label_ui",
    "zones_in_scope_from_ui",

    # Notes
    "load_notes",
    "save_notes",
    "reset_notes",
    "upsert_note",
    "company_key"
]
