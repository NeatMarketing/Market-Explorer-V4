"""
Dataset discovery & catalog.

Expected filename format (always):
  market_vertical_zone_cleaned.csv

Examples:
  goods_ameublement_france_cleaned.csv
  travel_airline_europe_cleaned.csv
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable


@dataclass(frozen=True)
class DatasetInfo:
    market: str
    vertical: str
    zone: str
    path: Path


@dataclass(frozen=True)
class DatasetInfoV2:
    vertical: str
    subvertical: str
    country: str
    path: Path


def parse_dataset_filename(path: Path) -> Optional[DatasetInfo]:
    stem = path.stem

    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]

    tokens = stem.split("_")
    if len(tokens) < 3:
        return None

    market = tokens[0].strip().lower()
    zone = tokens[-1].strip().lower()
    vertical = "_".join(tokens[1:-1]).strip().lower()

    if not market or not zone or not vertical:
        return None

    return DatasetInfo(market=market, vertical=vertical, zone=zone, path=path)


def list_datasets(data_dir: Path) -> List[DatasetInfo]:
    if not data_dir.exists():
        return []

    out: List[DatasetInfo] = []
    for p in sorted(data_dir.glob("*.csv")):
        info = parse_dataset_filename(p)
        if not info:
            continue

        # Special-case: monolithic file "*_zone_cleaned.csv"
        # ex: "market_verticale_zone_cleaned.csv"
        # -> the real zone is inside the CSV column "zone" ∈ {"france","europe"}
        if info.zone == "zone":
            out.append(DatasetInfo(market=info.market, vertical=info.vertical, zone="france", path=info.path))
            out.append(DatasetInfo(market=info.market, vertical=info.vertical, zone="europe", path=info.path))
        else:
            out.append(info)

    return out


def parse_dataset_filename_v2(path: Path) -> Optional[DatasetInfoV2]:
    stem = path.stem

    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]

    tokens = stem.split("_")
    if len(tokens) < 3:
        return None

    vertical = tokens[0].strip().lower()
    subvertical = tokens[1].strip().lower()
    country = "_".join(tokens[2:]).strip().lower()

    if not vertical or not subvertical or not country:
        return None

    return DatasetInfoV2(vertical=vertical, subvertical=subvertical, country=country, path=path)


def parse_country_from_filename(path: Path) -> Optional[str]:
    stem = path.stem

    if stem.endswith("_cleaned"):
        stem = stem[: -len("_cleaned")]

    tokens = stem.split("_")
    if len(tokens) < 3:
        return None

    country = "_".join(tokens[2:]).strip().lower()
    if not country:
        return None

    return country

def list_datasets_v2(data_dir: Path) -> List[DatasetInfoV2]:
    if not data_dir.exists():
        return []

    out: List[DatasetInfoV2] = []
    for p in sorted(data_dir.rglob("*.csv")):
        try:
            rel_parts = p.relative_to(data_dir).parts
        except ValueError:
            rel_parts = ()

        if len(rel_parts) >= 3:
            vertical = rel_parts[0].strip().lower()
            subvertical = rel_parts[1].strip().lower()
            country = parse_country_from_filename(p)
            if vertical and subvertical and country:
                out.append(
                    DatasetInfoV2(
                        vertical=vertical,
                        subvertical=subvertical,
                        country=country,
                        path=p,
                    )
                )
                continue

        info = parse_dataset_filename_v2(p)
        if info:
            out.append(info)

    return out
    
class DatasetCatalog:
    def __init__(self, datasets: List[DatasetInfo]):
        self.datasets = datasets

    @classmethod
    def from_dir(cls, data_dir: Path) -> "DatasetCatalog":
        return cls(list_datasets(data_dir))

    def zones(self) -> List[str]:
        return sorted({d.zone for d in self.datasets})

    def markets_for_zones(self, zones: Iterable[str]) -> List[str]:
        zones = {z.lower().strip() for z in zones}
        return sorted({d.market for d in self.datasets if d.zone in zones})

    def verticals_for(self, market: str, zones: Iterable[str]) -> List[str]:
        market = market.lower().strip()
        zones = {z.lower().strip() for z in zones}
        return sorted({d.vertical for d in self.datasets if d.market == market and d.zone in zones})

    def paths_for(self, market: str, vertical: str, zones: Iterable[str]) -> List[Path]:
        market = market.lower().strip()
        vertical = vertical.lower().strip()
        zones = {z.lower().strip() for z in zones}
        return sorted(
            d.path for d in self.datasets
            if d.market == market and d.vertical == vertical and d.zone in zones
        )
