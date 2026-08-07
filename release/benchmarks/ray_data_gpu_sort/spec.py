"""Fixed BTS workload definitions and historical DGX comparison points."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

GIB = 1 << 30
RAY_VERSION = "2.55.1"
RAY_COMMIT = "237c2455ebb1ea15a32dd9e1fdeb2d617badc37f"
RAY_WHEEL = "ray-2.55.1-cp310-cp310-manylinux2014_x86_64.whl"
RAY_WHEEL_SHA256 = "bb49fbbe53a1d931e1f92d17f9271338f0b738885f8f70b7f531aa33f019d8af"
DATASET_ROOT = Path("/raid/spark-team/bobbwang/datasets/bts-airline-on-time")
EXPECTED_ROWS = 80_738_761
EXPECTED_BLOCKS = 627
EXPECTED_START = (2013, 4)
EXPECTED_END = (2025, 12)
EXPECTED_FULL_BYTES = 68_591_423_897
LARGE_COPIES = 3
LARGE_ROWS = EXPECTED_ROWS * LARGE_COPIES
FOUR_KEYS = ("Origin", "Dest", "FlightDate", "CRSDepTime")
SMOKE_KEYS = ("Origin", "OriginAirportID", "FlightDate", "row_id")


@dataclass(frozen=True)
class Cell:
    name: str
    label: str
    columns: tuple[str, ...]
    keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["columns"] = list(self.columns)
        value["keys"] = list(self.keys)
        return value


def load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if len(manifest.get("schema_names", ())) != 109:
        raise ValueError(f"Expected the normalized 109-column BTS corpus: {path}")
    return manifest


def cells(manifest: dict[str, Any]) -> tuple[Cell, ...]:
    native = tuple(str(name) for name in manifest["schema_names"])
    if native[55] != "DistanceGroup":
        raise ValueError("BTS core projection no longer ends at DistanceGroup")
    full = (*native, "row_id")
    result = (
        Cell(
            "narrow",
            "Narrow payload",
            ("Origin", "Dest", "FlightDate", "CRSDepTime", "row_id"),
            FOUR_KEYS,
        ),
        Cell("core", "Core payload", (*native[:56], "row_id"), FOUR_KEYS),
        Cell("full", "Full baseline", full, FOUR_KEYS),
        Cell("origin-string", "Origin string", full, ("Origin",)),
        Cell("origin-integer", "Origin integer", full, ("OriginAirportID",)),
        Cell("route", "Route", full, ("Origin", "Dest")),
    )
    if any(not set(cell.keys).issubset(cell.columns) for cell in result):
        raise ValueError("A trend cell projected out one of its sort keys")
    return result


def cell_by_name(manifest: dict[str, Any], name: str) -> Cell:
    matches = [cell for cell in cells(manifest) if cell.name == name]
    if len(matches) != 1:
        raise ValueError(f"Unknown BTS trend cell: {name}")
    return matches[0]


OLD_DGX = {
    "narrow": {"pyarrow_s": 198.408, "gpu_s": [22.324, 22.201]},
    "core": {"pyarrow_s": 236.588, "gpu_s": [37.070, 36.479]},
    "full": {"pyarrow_s": 257.806, "gpu_s": [51.743, 50.747]},
    "origin-string": {"pyarrow_s": 152.018, "gpu_s": [50.261, 49.640]},
    "origin-integer": {"pyarrow_s": 78.982, "gpu_s": [50.399, 49.600]},
    "route": {"pyarrow_s": 262.923, "gpu_s": [49.895, 51.501]},
}


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
