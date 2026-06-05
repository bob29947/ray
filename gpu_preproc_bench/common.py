"""Shared harness for the GPU preprocessor benchmarks.

Mirrors the methodology of ``gpu_sort_bench``: a warm, best-of-N, in-memory ->
in-memory ("RAM in -> RAM out") measurement of ``fit_transform`` on a realistic
recommender-style event table, comparing a CPU preprocessor against its GPU
drop-in. Each benchmark script imports from here and can be run on its own.
"""

from __future__ import annotations

import statistics
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import ray


def make_recsys_dataset(
    n_rows: int,
    n_blocks: int,
    *,
    item_card: int = 200_000,
    user_card: int = 50_000,
    null_frac: float = 0.05,
    seed: int = 0,
) -> "ray.data.Dataset":
    """Build a materialized (RAM-resident) recommender-style event table.

    Columns mirror a clickstream: high-cardinality string ids (``item_id``,
    ``user_id``), low-cardinality string categoricals (``event_type``,
    ``device_type``, ``country``) with injected nulls (for the imputer), and a
    numeric ``price`` with nulls.
    """
    rng = np.random.default_rng(seed)

    def maybe_null(values: np.ndarray) -> pa.Array:
        arr = values.astype(object)
        arr[rng.random(len(arr)) < null_frac] = None
        return pa.array(arr)

    events = np.array(["click", "cart", "purchase", "view"])
    devices = np.array(["ios", "android", "web", "tv"])
    countries = np.array(["US", "CA", "GB", "DE", "FR", "JP", "IN", "BR"])

    item = pc.cast(pa.array(rng.integers(0, item_card, n_rows)), pa.string())
    user = pc.cast(pa.array(rng.integers(0, user_card, n_rows)), pa.string())
    # High-cardinality nullable categorical (e.g. the previous item in a session,
    # null for the first event) -- the realistic "impute most_frequent on a
    # high-cardinality column" case where the CPU Counter merge is slow.
    last_item = np.char.mod("%d", rng.integers(0, item_card, n_rows))
    last_item_col = pa.array(last_item, mask=(rng.random(n_rows) < null_frac))
    # Real Arrow nulls (not NaN) so the CPU Mean aggregator skips missing values.
    price_values = rng.random(n_rows) * 100.0
    price_mask = rng.random(n_rows) < null_frac

    table = pa.table(
        {
            "item_id": item,
            "user_id": user,
            "last_item_id": last_item_col,
            "event_type": maybe_null(events[rng.integers(0, len(events), n_rows)]),
            "device_type": maybe_null(devices[rng.integers(0, len(devices), n_rows)]),
            "country": maybe_null(countries[rng.integers(0, len(countries), n_rows)]),
            "price": pa.array(price_values, mask=price_mask),
            "row_id": pa.array(np.arange(n_rows, dtype=np.int64)),
        }
    )
    return ray.data.from_arrow(table).repartition(n_blocks).materialize()


def best_of(
    make: Callable[[], None], *, trials: int = 3, warmup: int = 1
) -> Tuple[float, float, List[float]]:
    """Run ``warmup`` (uncounted) + ``trials`` timed; return best/median/all."""
    for _ in range(warmup):
        make()
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        make()
        times.append(time.perf_counter() - start)
    return min(times), statistics.median(times), times


def fit_transform_wall(preprocessor_factory, ds) -> None:
    """One RAM -> RAM fit_transform (materialized output)."""
    preprocessor_factory().fit_transform(ds).materialize()


def nbytes(ds) -> int:
    return ds.size_bytes() or 0


def gib(n: int) -> float:
    return n / (1024 ** 3)
