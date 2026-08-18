"""Deterministic projection readers for the exact BTS cohort."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .spec import (
    EXPECTED_BLOCKS,
    EXPECTED_END,
    EXPECTED_ROWS,
    EXPECTED_START,
    digest,
)


@dataclass(frozen=True)
class Slice:
    path: str
    row_group: int
    rows: int
    row_id_start: int
    year: int
    month: int
    copy: int = 0


def _local_path(root: Path, recorded: str) -> Path:
    """Relocate a manifest made on another machine to ``root`` when needed."""

    path = Path(recorded)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        return (root / path).resolve()
    try:
        parquet_index = path.parts.index("parquet")
    except ValueError as error:
        raise ValueError(
            f"Cannot relocate BTS path outside a parquet tree: {path}"
        ) from error
    return root.joinpath(*path.parts[parquet_index:]).resolve()


def cohort_slices(root: Path, manifest: Mapping[str, Any]) -> tuple[Slice, ...]:
    records = sorted(
        manifest["row_groups"],
        key=lambda item: (
            int(item["year"]),
            int(item["month"]),
            str(item["path"]),
            int(item["row_group"]),
        ),
    )
    selected = [
        item
        for item in records
        if EXPECTED_START <= (int(item["year"]), int(item["month"])) <= EXPECTED_END
    ]
    result = []
    next_row = 0
    for item in selected:
        rows = int(item["rows"])
        result.append(
            Slice(
                path=str(_local_path(root, str(item["path"]))),
                row_group=int(item["row_group"]),
                rows=rows,
                row_id_start=next_row,
                year=int(item["year"]),
                month=int(item["month"]),
            )
        )
        next_row += rows
    if len(result) != EXPECTED_BLOCKS or next_row != EXPECTED_ROWS:
        raise ValueError(
            f"BTS cohort changed: {len(result)} blocks/{next_row} rows, expected "
            f"{EXPECTED_BLOCKS}/{EXPECTED_ROWS}"
        )
    return tuple(result)


def scaled_slices(
    root: Path,
    manifest: Mapping[str, Any],
    numerator: int,
    denominator: int = 1,
) -> tuple[Slice, ...]:
    """Repeat a deterministic cohort prefix with contiguous unique row IDs."""

    if numerator < 1 or denominator < 1:
        raise ValueError("Scale numerator and denominator must be positive")
    target_rows = EXPECTED_ROWS * numerator // denominator
    base = cohort_slices(root, manifest)
    result = []
    next_row = 0
    copy = 0
    while next_row < target_rows:
        for item in base:
            if next_row >= target_rows:
                break
            rows = min(item.rows, target_rows - next_row)
            result.append(
                Slice(
                    path=item.path,
                    row_group=item.row_group,
                    rows=rows,
                    row_id_start=next_row,
                    year=item.year,
                    month=item.month,
                    copy=copy,
                )
            )
            next_row += rows
        copy += 1
    return tuple(result)


def smoke_slices(
    root: Path, manifest: Mapping[str, Any], rows_per_block: int = 10_000
) -> tuple[Slice, ...]:
    chosen = cohort_slices(root, manifest)[-16:]
    result = []
    next_row = 0
    for item in chosen:
        rows = min(rows_per_block, item.rows)
        result.append(
            Slice(
                path=item.path,
                row_group=item.row_group,
                rows=rows,
                row_id_start=next_row,
                year=item.year,
                month=item.month,
            )
        )
        next_row += rows
    return tuple(result)


def plan_dict(slices: Iterable[Slice], *, kind: str) -> dict[str, Any]:
    values = [asdict(item) for item in slices]
    identity = {
        "kind": kind,
        "blocks": len(values),
        "rows": sum(int(item["rows"]) for item in values),
        "slices": values,
    }
    identity["digest"] = digest(identity)
    return identity


def read_projection(value: Mapping[str, Any], columns: tuple[str, ...]):
    """Read one projected Arrow block and append a globally unique ``row_id``."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    item = Slice(**value)
    native = [name for name in columns if name != "row_id"]
    table = pq.ParquetFile(item.path).read_row_group(item.row_group, columns=native)
    table = table.slice(0, item.rows)

    # Normalizing through IPC removes spare variable-width buffer capacity, so
    # decoded-byte measurements do not depend on the Parquet reader's buffers.
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    table = pa.ipc.open_stream(sink.getvalue()).read_all()
    if "row_id" in columns:
        ids = pa.array(
            np.arange(item.row_id_start, item.row_id_start + item.rows, dtype=np.int64)
        )
        table = table.append_column("row_id", ids)
    table = table.select(columns)
    if table.num_rows != item.rows:
        raise RuntimeError(f"Read {table.num_rows} rows, expected {item.rows}")
    return table
