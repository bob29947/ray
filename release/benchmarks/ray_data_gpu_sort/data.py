"""Deterministic readers for the exact BTS cohort used by the prior study."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

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


def cohort_slices(manifest: dict[str, Any], copies: int = 1) -> tuple[Slice, ...]:
    records = sorted(
        manifest["row_groups"],
        key=lambda item: (item["year"], item["month"], item["path"], item["row_group"]),
    )
    selected = [
        item
        for item in records
        if EXPECTED_START <= (int(item["year"]), int(item["month"])) <= EXPECTED_END
    ]
    base: list[Slice] = []
    next_row = 0
    for item in selected:
        rows = int(item["rows"])
        base.append(
            Slice(
                path=str(item["path"]),
                row_group=int(item["row_group"]),
                rows=rows,
                row_id_start=next_row,
                year=int(item["year"]),
                month=int(item["month"]),
            )
        )
        next_row += rows
    if len(base) != EXPECTED_BLOCKS or next_row != EXPECTED_ROWS:
        raise ValueError(
            f"BTS cohort changed: {len(base)} blocks/{next_row} rows, expected "
            f"{EXPECTED_BLOCKS}/{EXPECTED_ROWS}"
        )
    return tuple(
        Slice(
            **{
                **asdict(item),
                "copy": copy,
                "row_id_start": item.row_id_start + copy * EXPECTED_ROWS,
            }
        )
        for copy in range(copies)
        for item in base
    )


def scaled_cohort_slices(
    manifest: dict[str, Any], numerator: int, denominator: int = 1
) -> tuple[Slice, ...]:
    """Repeat a deterministic cohort prefix with contiguous unique row IDs."""

    if numerator < 1 or denominator < 1:
        raise ValueError("Cohort scale numerator and denominator must be positive")
    target_rows = EXPECTED_ROWS * numerator // denominator
    base = cohort_slices(manifest)
    result: list[Slice] = []
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
    manifest: dict[str, Any], rows_per_block: int = 10_000
) -> tuple[Slice, ...]:
    base = cohort_slices(manifest)
    chosen = list(base[-16:])
    result: list[Slice] = []
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
    result = {
        "kind": kind,
        "slices": values,
        "blocks": len(values),
        "rows": sum(int(item["rows"]) for item in values),
    }
    result["digest"] = digest(result)
    return result


def read_projection(value: dict[str, Any], columns: tuple[str, ...]):
    """Read one projected Arrow block and append a globally unique row_id."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    item = Slice(**value)
    native = [name for name in columns if name != "row_id"]
    table = pq.ParquetFile(Path(item.path)).read_row_group(
        item.row_group, columns=native
    )
    table = table.slice(0, item.rows)
    # Remove spare string-buffer capacity so Plasma bytes are stable.
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
