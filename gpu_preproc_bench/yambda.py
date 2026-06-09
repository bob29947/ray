"""Loader + null-injection helpers for the real yandex/yambda benchmark.

Downloads the ``flat/500m`` parquet files from the Hugging Face Hub
(``huggingface_hub`` only -- no ``datasets``), reads them with
``ray.data.read_parquet``, reports natural null fractions straight from the
parquet row-group statistics (no full scan), and provides deterministic, seeded
null injection that is materialized once and reused for the CPU and GPU runs so
both see identical null masks.

Dataset status (flat/500m, measured from parquet metadata):

* ``multi_event.parquet`` -- 480,255,564 rows. ``played_ratio_pct`` /
  ``track_length_seconds`` are NATURALLY null for non-listen events (~2.86%);
  ``event_type`` is a dictionary<string>; the id/timestamp/flag columns have no
  nulls. This is the main recommender-preprocessing benchmark input.
* ``likes.parquet`` -- 9,033,960 rows, four non-nullable uint columns (no native
  nulls), used as a targeted imputer stress with injected nulls.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import ray

REPO_ID = "yandex/yambda"
SIZE = "flat/500m"
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")


# --------------------------------------------------------------------------- #
# Download / paths
# --------------------------------------------------------------------------- #
def local_path(file: str) -> str:
    """Local cached path for ``flat/500m/<file>.parquet``."""
    return os.path.join(DATA_DIR, SIZE, f"{file}.parquet")


def download(file: str) -> str:
    """Download ``flat/500m/<file>.parquet`` into :data:`DATA_DIR` (idempotent)."""
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{SIZE}/{file}.parquet",
        local_dir=DATA_DIR,
    )


def ensure(file: str) -> str:
    """Return the local path, downloading the file first if it is missing."""
    path = local_path(file)
    return path if os.path.exists(path) else download(file)


# --------------------------------------------------------------------------- #
# Metadata helpers (no data scan)
# --------------------------------------------------------------------------- #
def num_rows(file_or_path: str) -> int:
    path = file_or_path if os.path.sep in file_or_path else ensure(file_or_path)
    return pq.ParquetFile(path).metadata.num_rows


def parquet_null_fracs(file_or_path: str) -> Dict[str, float]:
    """Per-column null fraction read from parquet row-group stats (no scan)."""
    path = file_or_path if os.path.sep in file_or_path else ensure(file_or_path)
    md = pq.ParquetFile(path).metadata
    names = pq.read_schema(path).names
    n = md.num_rows or 1
    nulls = dict.fromkeys(names, 0)
    for rg in range(md.num_row_groups):
        rgm = md.row_group(rg)
        for c in range(rgm.num_columns):
            col = rgm.column(c)
            st = col.statistics
            if st is not None and col.path_in_schema in nulls:
                nulls[col.path_in_schema] += st.null_count
    return {nm: nulls[nm] / n for nm in names}


def print_status(file: str) -> None:
    path = ensure(file)
    print(f"[{file}] {num_rows(path):,} rows  ({path})", flush=True)
    for name, frac in parquet_null_fracs(path).items():
        tag = "  <- natural nulls" if frac > 0 else ""
        print(f"    null_frac[{name}] = {frac:.4f}{tag}", flush=True)


def _itemsize(t: "pa.DataType") -> int:
    """Approximate fixed-width byte size of an Arrow type (for byte estimates)."""
    if pa.types.is_dictionary(t):
        return _itemsize(t.value_type)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return 8  # rough average; strings are variable-width
    try:
        return max(1, t.bit_width // 8)
    except Exception:
        return 8


def selected_bytes(schema: "pa.Schema", cols: Sequence[str], rows: int) -> int:
    """Estimated bytes for ``cols`` x ``rows`` (the H2D payload of a transform)."""
    total = 0
    for c in cols:
        idx = schema.get_field_index(c)
        if idx != -1:
            total += _itemsize(schema.field(idx).type) * rows
    return total


# --------------------------------------------------------------------------- #
# Reading + deterministic null injection
# --------------------------------------------------------------------------- #
def read_ray(
    file: str, *, override_num_blocks: Optional[int] = None
) -> "ray.data.Dataset":
    return ray.data.read_parquet(ensure(file), override_num_blocks=override_num_blocks)


def _decode(arr) -> "pa.Array":
    """Combine chunks and decode dictionary arrays to their value type."""
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    if pa.types.is_dictionary(arr.type):
        arr = arr.cast(arr.type.value_type)
    return arr


def inject_batch(
    tbl: "pa.Table", columns: Sequence[str], frac: float, seed: int
) -> "pa.Table":
    """Mask ~``frac`` of each column's values to null (deterministic per seed).

    Dictionary columns (e.g. ``event_type``) are decoded to their value type so
    the GPU path counts/fills them as plain strings (no categorical edge cases).
    """
    rng = np.random.default_rng(seed)
    out = tbl
    for col in columns:
        if col not in tbl.column_names:
            continue
        arr = _decode(tbl.column(col))
        vals = arr.to_numpy(zero_copy_only=False)
        mask = rng.random(len(vals)) < frac
        new = pa.array(vals, mask=mask)
        idx = out.schema.get_field_index(col)
        out = out.set_column(idx, col, new)
    return out


def inject_nulls(
    ds: "ray.data.Dataset",
    columns: Sequence[str],
    *,
    frac: float = 0.05,
    seed: int = 0,
) -> "ray.data.Dataset":
    """Lazily inject ~``frac`` nulls into ``columns`` (call ``.materialize()``).

    Materialize once and reuse the result for both the CPU and GPU runs so they
    see byte-identical null masks.
    """
    cols = list(columns)

    def fn(tbl: "pa.Table") -> "pa.Table":
        return inject_batch(tbl, cols, frac, seed)

    return ds.map_batches(fn, batch_format="pyarrow", zero_copy_batch=True)


def sample_with_row_id(ds: "ray.data.Dataset", n: int) -> "ray.data.Dataset":
    """A small materialized sample with a stable ``row_id`` for parity checks."""
    pdf = ds.limit(n).to_pandas().reset_index(drop=True)
    pdf["row_id"] = np.arange(len(pdf), dtype="int64")
    return ray.data.from_pandas(pdf).materialize()


def _finalize(tbl, columns, inject, frac, seed) -> "pa.Table":
    if inject:
        return inject_batch(tbl, list(inject), frac, seed)
    # Still decode dictionary columns for a representative device path.
    return pa.table({c: _decode(tbl.column(c)) for c in tbl.column_names})


def arrow_batches(
    file: str,
    columns: Sequence[str],
    batch_size: int,
    k: int,
    *,
    inject: Optional[Sequence[str]] = None,
    frac: float = 0.05,
    seed: int = 0,
) -> List["pa.Table"]:
    """First ``k`` Arrow batches of exactly ``batch_size`` rows (microbench input).

    Reads straight from parquet with PyArrow (no Ray). PyArrow caps ``iter_batches``
    at the row-group size, so we accumulate row groups and slice to the requested
    ``batch_size`` to faithfully test large GPU batches. Optionally injects nulls
    into the ``inject`` columns so ``fillna`` actually does work.
    """
    path = ensure(file)
    need = batch_size * k
    parts, got = [], 0
    for rb in pq.ParquetFile(path).iter_batches(
        batch_size=262_144, columns=list(columns)
    ):
        parts.append(rb)
        got += rb.num_rows
        if got >= need:
            break
    big = pa.Table.from_batches(parts)
    out: List["pa.Table"] = []
    for i in range(k):
        head = big.slice(i * batch_size, batch_size)
        if head.num_rows == 0:
            break
        out.append(_finalize(head, columns, inject, frac, seed))
    return out
