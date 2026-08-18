"""Frozen workload definitions for the BTS GPU-sort study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

EXPECTED_ROWS = 80_738_761
EXPECTED_BLOCKS = 627
EXPECTED_FULL_BYTES = 68_591_423_897
EXPECTED_START = (2013, 4)
EXPECTED_END = (2025, 12)
FOUR_KEYS = ("Origin", "Dest", "FlightDate", "CRSDepTime")
SMOKE_KEYS = ("Origin", "OriginAirportID", "FlightDate", "row_id")
SCALES = ((1, 1), (2, 1), (245, 100))


@dataclass(frozen=True)
class Cell:
    """One projection and key choice from the trend matrix."""

    name: str
    label: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["columns"] = list(self.columns)
        value["keys"] = list(self.keys)
        return value


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_manifest(root: Path) -> dict[str, Any]:
    """Load a staged BTS manifest and validate its stable public shape."""

    path = root / "manifest.json"
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    names = manifest.get("schema_names")
    if not isinstance(names, list) or len(names) != 109:
        raise ValueError(f"Expected a normalized 109-column BTS corpus: {path}")
    if not isinstance(manifest.get("row_groups"), list):
        raise ValueError(f"BTS manifest has no row-group plan: {path}")
    return manifest


def cells(manifest: Mapping[str, Any]) -> tuple[Cell, ...]:
    native = tuple(str(name) for name in manifest["schema_names"])
    if native[55] != "DistanceGroup":
        raise ValueError("BTS core projection no longer ends at DistanceGroup")
    full = (*native, "row_id")
    result = (
        Cell(
            "narrow",
            "Narrow",
            ("Origin", "Dest", "FlightDate", "CRSDepTime", "row_id"),
            FOUR_KEYS,
        ),
        Cell("core", "Core", (*native[:56], "row_id"), FOUR_KEYS),
        Cell("full", "Full", full, FOUR_KEYS),
        Cell("origin-string", "Origin string", full, ("Origin",)),
        Cell("origin-integer", "Origin integer", full, ("OriginAirportID",)),
        Cell("route", "Route", full, ("Origin", "Dest")),
    )
    if any(not set(cell.keys).issubset(cell.columns) for cell in result):
        raise ValueError("A trend cell projects out one of its sort keys")
    return result


def cell_by_name(manifest: Mapping[str, Any], name: str) -> Cell:
    matches = [cell for cell in cells(manifest) if cell.name == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown BTS trend cell: {name}")
    return matches[0]
