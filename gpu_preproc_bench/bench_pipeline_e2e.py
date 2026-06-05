"""End-to-end pipeline benchmark: CPU stack vs our GPU drop-ins.

The pipeline (raw, unordered events in RAM -> result in RAM):

    raw events  ──►  SimpleImputer.fit_transform        (impute missing values)
                ──►  add session_bucket + parse event_ts (shared CPU map)
                ──►  Dataset.sort(["session_bucket","session_id","event_ts"])
                ──►  OrdinalEncoder.fit_transform        (encode categoricals)

Two pipelines run the SAME stages on the SAME data; only three operators differ:

    stage          CPU pipeline            GPU pipeline
    -----          -----------             -----------
    impute         SimpleImputer           GpuSimpleImputer
    derive         (identical CPU map -- same code in both)
    sort           ds.sort(backend="cpu")  ds.sort(backend="gpu")   # cuDF+rapidsmpf
    encode         OrdinalEncoder          GpuOrdinalEncoder

Why this is easy to verify:
  * Every stage is ``.materialize()``-d, so each stage starts and ends in RAM
    (Arrow in the object store). The end-to-end time is just the sum of the four
    stage times, and both pipelines pay the exact same staging -- a fair, like-
    for-like comparison.
  * The sort key is made UNIQUE (event timestamps are a unique, shuffled set), so
    the CPU sort and the GPU sort produce the *same* deterministic row order. The
    GPU result must therefore equal the CPU result row-for-row.
  * Correctness is three independent checks (printed PASS/FAIL):
      (a) content parity  -- re-key both results by the unique ``event_id`` and
          compare every column (order-independent: proves impute+derive+encode
          produced identical values);
      (b) sortedness      -- each result is globally non-decreasing by the sort
          key (proves the sort worked in both pipelines);
      (c) order parity    -- the materialized ``event_id`` sequence is identical
          (proves the CPU and GPU sorts agree exactly).

If no GPU / RAPIDS stack is present, the GpuX operators fall back to CPU and the
sort is forced to CPU, so the script still runs and the correctness harness still
proves out (the "GPU" timings are then just CPU-fallback and are labeled as such).

Run:
    RAY_DATA_GPU_PREPROC_BATCH_SIZE=2000000 \
        .venv/bin/python gpu_preproc_bench/bench_pipeline_e2e.py \
        --rows 20000000 --blocks 32 --gpus 8 --trials 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _configure_gpus(gpus: int, batch_size: int):
    """Give EVERY GPU stage all ``gpus`` GPUs, used sequentially.

    The general GPU sort now returns *executor-owned* output blocks (emitted as
    task results, not ``ray.put`` inside the actors) and tears down its actor
    pool after the sort (``RAY_DATA_GPU_SORT_RELEASE=1``). So the sort no longer
    has to hold a private half of the GPUs for the whole run: impute, sort, and
    encode each grab all ``gpus`` GPUs in turn, and the GPUs are released
    between stages. This removes the old disjoint 4/4 split.

    Must run before ``import ray``. We deliberately do NOT set
    ``RAY_DATA_GPU_SORT_IMPL`` so ``sort(gpu=True)`` uses the *general* engine,
    which is the one that handles our mixed int/string multi-key sort.
    """
    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(gpus)
    os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(gpus)
    os.environ["RAY_DATA_GPU_SORT_RELEASE"] = "1"
    os.environ.setdefault("RAY_DATA_GPU_PREPROC_BATCH_SIZE", str(batch_size))
    # Silence a cosmetic Ray 3.0 teardown crash: the OpenTelemetry metrics thread
    # can fire a gauge callback while a GPU worker is being destroyed ("pure
    # virtual method called"). It happens after the worker has returned its result
    # so it does not affect correctness or timing -- it just floods the log.
    os.environ.setdefault("RAY_enable_open_telemetry", "0")
    return gpus


# Parse early so the env is set before ray imports below.
_ap = argparse.ArgumentParser()
_ap.add_argument("--rows", type=int, default=20_000_000, help="rows in the timing dataset")
_ap.add_argument("--blocks", type=int, default=32, help="blocks (partitions)")
_ap.add_argument("--gpus", type=int, default=8, help="GPUs for sort + preprocessors")
_ap.add_argument("--batch-size", type=int, default=2_000_000, help="per-GPU batch size")
_ap.add_argument("--trials", type=int, default=2, help="timed trials (best is reported)")
_ap.add_argument("--corr-rows", type=int, default=200_000, help="rows for the correctness check")
_ap.add_argument("--corr-blocks", type=int, default=8)
ARGS = _ap.parse_args()
GPUS = _configure_gpus(ARGS.gpus, ARGS.batch_size)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

import ray  # noqa: E402
from common import gib, nbytes  # noqa: E402

from ray.data.preprocessors import (  # noqa: E402
    GpuOrdinalEncoder,
    GpuSimpleImputer,
    OrdinalEncoder,
    SimpleImputer,
    _gpu,
)

P = lambda *a: print(*a, flush=True)  # noqa: E731

# --- pipeline configuration (the columns each stage touches) --------------- #
IMPUTE_COLS = ["device_type", "country"]            # nullable low-card categoricals
SORT_KEY = ["session_bucket", "session_id", "event_ts"]
ENCODE_COLS = ["item_id", "device_type", "country"]  # high-card id + the imputed cats
BUCKET_MS = 3_600_000                                # 1-hour session buckets

HAS_GPU = _gpu.gpu_available()


# --------------------------------------------------------------------------- #
# Stage 2 (shared, identical for both pipelines): a plain CPU map that parses
# the raw timestamp string into an int64 and derives the session_bucket.
# --------------------------------------------------------------------------- #
def add_session_and_parse_ts(table: "pa.Table") -> "pa.Table":
    # Done in pyarrow (not pandas) so no pandas index leaks into the block as a
    # spurious column -- every output block keeps an identical schema, which the
    # GPU sort requires. Identical code runs in both pipelines, so it can never
    # be the source of a CPU-vs-GPU difference.
    #
    # "parse event_ts": raw epoch-millis string -> int64 (unambiguous).
    event_ts = pc.cast(table.column("event_ts_raw"), pa.int64())
    session_bucket = pc.divide(event_ts, pa.scalar(BUCKET_MS, pa.int64()))
    table = table.drop_columns(["event_ts_raw"])
    table = table.append_column("event_ts", event_ts)
    table = table.append_column("session_bucket", session_bucket)
    return table


# --------------------------------------------------------------------------- #
# Raw, unordered event table (RAM-resident before timing).
# --------------------------------------------------------------------------- #
def make_events_dataset(
    n_rows: int,
    n_blocks: int,
    *,
    n_sessions: int = 50_000,
    item_card: int = 200_000,
    null_frac: float = 0.08,
    seed: int = 0,
) -> "ray.data.Dataset":
    rng = np.random.default_rng(seed)

    event_id = np.arange(n_rows, dtype=np.int64)  # unique payload + verify key

    # Unique, shuffled event timestamps -> a UNIQUE composite sort key, so the
    # CPU and GPU sorts produce identical, deterministic order (easy to verify).
    base_ms = 1_767_225_600_000  # 2026-01-01T00:00:00Z, in epoch millis
    event_ts_ms = base_ms + rng.permutation(n_rows).astype(np.int64) * 137
    event_ts_raw = event_ts_ms.astype(str)  # the "raw" string we will parse

    session_id = np.char.add("s", rng.integers(0, n_sessions, n_rows).astype(str))
    item_id = np.char.add("i", rng.integers(0, item_card, n_rows).astype(str))

    # Low-card categoricals with a clearly DOMINANT mode (so most_frequent has no
    # tie -> CPU's insertion-order and GPU's smallest-value tie-break agree), then
    # punch in nulls for the imputer to fill.
    devices = np.array(["ios", "ios", "ios", "ios", "android", "web", "tv"])
    countries = np.array(["US", "US", "US", "CA", "GB", "DE", "FR", "JP"])
    device_type = devices[rng.integers(0, len(devices), n_rows)].astype(object)
    country = countries[rng.integers(0, len(countries), n_rows)].astype(object)
    device_type[rng.random(n_rows) < null_frac] = None
    country[rng.random(n_rows) < null_frac] = None

    table = pa.table(
        {
            "event_id": event_id,
            "session_id": pa.array(session_id),
            "event_ts_raw": pa.array(event_ts_raw),
            "item_id": pa.array(item_id),
            "device_type": pa.array(device_type),
            "country": pa.array(country),
        }
    )
    return ray.data.from_arrow(table).repartition(n_blocks).materialize()


# --------------------------------------------------------------------------- #
# The pipeline. ``gpu`` switches the three operators; everything else is shared.
# Each stage is materialized so it is a clean RAM -> RAM step we can time.
# --------------------------------------------------------------------------- #
@contextmanager
def stage(timings: Dict[str, float], name: str):
    t = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - t


def _release_sort_actors() -> None:
    """Tear down the GPU sort's detached actor pool (called once, at the end)."""
    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool

        kill_actor_pool(GPUS)
    except Exception:
        pass


def _wait_for_gpus(gpus: int, timeout_s: float = 60.0) -> None:
    """Block (best-effort) until at least ``gpus`` GPUs are free.

    Each GPU stage uses all the GPUs and frees them when it finishes: Ray Data
    releases a map operator's actor pool when the operator completes, and the
    GPU sort releases its pool after the sort (``RAY_DATA_GPU_SORT_RELEASE=1``).
    Both are async, so we wait here so the next stage starts with all GPUs
    available (deterministic, sequential hand-off instead of the old 4/4 split).
    """
    import ray

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ray.available_resources().get("GPU", 0.0) >= gpus - 0.5:
            return
        time.sleep(0.2)


def run_pipeline(ds, *, gpu: bool, timings: Dict[str, float]):
    use_gpu = gpu and HAS_GPU
    Imputer = GpuSimpleImputer if gpu else SimpleImputer
    Encoder = GpuOrdinalEncoder if gpu else OrdinalEncoder
    sort_backend = "gpu" if use_gpu else "cpu"

    with stage(timings, "1_impute"):
        ds = (
            Imputer(columns=IMPUTE_COLS, strategy="most_frequent")
            .fit_transform(ds)
            .materialize()
        )
    with stage(timings, "2_derive"):
        # The imputer's large batch_size can coalesce the data into very few
        # blocks; repartition so the multi-GPU sort has >= one non-empty block
        # per rank (an empty rank would emit a null-typed key sample and break
        # the sort's sample concat). shuffle=False keeps this a cheap split.
        ds = (
            ds.map_batches(add_session_and_parse_ts, batch_format="pyarrow")
            .repartition(max(ARGS.blocks, ARGS.gpus), shuffle=False)
            .materialize()
        )
    # Sequential all-GPU hand-off: wait for the impute pool's GPUs to free
    # before the sort grabs all of them (and again before encode).
    if use_gpu:
        _wait_for_gpus(GPUS)
    with stage(timings, "3_sort"):
        ds = ds.sort(SORT_KEY, backend=sort_backend).materialize()
    if use_gpu:
        _wait_for_gpus(GPUS)
    with stage(timings, "4_encode"):
        ds = Encoder(columns=ENCODE_COLS).fit_transform(ds).materialize()
    return ds


# --------------------------------------------------------------------------- #
# Correctness: compare the CPU result and the GPU result three ways.
# --------------------------------------------------------------------------- #
def verify(cpu_ds, gpu_ds) -> Dict[str, bool]:
    cpu = cpu_ds.to_pandas()
    gpu = gpu_ds.to_pandas()

    # (a) content parity, order-independent: re-key both by the unique event_id.
    c = cpu.sort_values("event_id").reset_index(drop=True)
    g = gpu.sort_values("event_id").reset_index(drop=True)
    compare_cols = [
        "session_id", "item_id", "device_type", "country", "event_ts", "session_bucket"
    ]
    content_ok = all((c[col].fillna(-1) == g[col].fillna(-1)).all() for col in compare_cols)

    # (b) each result is globally sorted by the key (in materialized order).
    def is_sorted(df: pd.DataFrame) -> bool:
        k = df[SORT_KEY].reset_index(drop=True)
        return k.equals(k.sort_values(SORT_KEY, kind="stable").reset_index(drop=True))

    cpu_sorted = is_sorted(cpu)
    gpu_sorted = is_sorted(gpu)

    # (c) order parity: identical row order (the key is unique -> deterministic).
    order_ok = bool((cpu["event_id"].to_numpy() == gpu["event_id"].to_numpy()).all())

    return {
        "content_parity": content_ok,
        "cpu_sorted": cpu_sorted,
        "gpu_sorted": gpu_sorted,
        "order_parity": order_ok,
    }


def timed_best(ds, *, gpu: bool, trials: int) -> Tuple[float, Dict[str, float]]:
    best_total, best_stages = float("inf"), {}
    for _ in range(trials):
        stages: Dict[str, float] = {}
        t0 = time.perf_counter()
        run_pipeline(ds, gpu=gpu, timings=stages)
        total = time.perf_counter() - t0
        if total < best_total:
            best_total, best_stages = total, stages
    return best_total, best_stages


def main():
    import logging

    ray.init(logging_level="ERROR", include_dashboard=False)
    # Quiet Ray Data's per-stage progress logging so the correctness PASS/FAIL
    # block and the timing table are easy to read and verify.
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    # Keep the globally-sorted order through the downstream encode stage. Ray Data
    # map operators do NOT preserve block order by default, so without this the
    # post-sort encode would scramble the sorted blocks (and "sorted output" would
    # silently be lost). Applies to both pipelines, so the comparison stays fair.
    ctx.execution_options.preserve_order = True
    P("=" * 74)
    P("end-to-end pipeline: impute -> derive -> sort -> encode   (CPU vs GPU)")
    P("=" * 74)
    P(f"GPU / RAPIDS available: {HAS_GPU}"
      + ("" if HAS_GPU else "   (GPU pipeline runs as CPU fallback)"))
    P(f"GPUs={ARGS.gpus} (each stage uses all {GPUS} sequentially; sort releases "
      f"its pool after sorting)  "
      f"batch_size={os.environ['RAY_DATA_GPU_PREPROC_BATCH_SIZE']}")

    # ---- correctness on a small dataset ---------------------------------- #
    P(f"\n[correctness] dataset: {ARGS.corr_rows:,} rows x {ARGS.corr_blocks} blocks")
    corr = make_events_dataset(ARGS.corr_rows, ARGS.corr_blocks)
    cpu_corr = run_pipeline(corr, gpu=False, timings={})
    gpu_corr = run_pipeline(corr, gpu=True, timings={})
    checks = verify(cpu_corr, gpu_corr)
    for name, ok in checks.items():
        P(f"  {name:<16}: {'PASS' if ok else 'FAIL'}")
    all_ok = all(checks.values())
    P(f"  -> overall      : {'PASS' if all_ok else 'FAIL'}")

    # ---- timing on the full dataset -------------------------------------- #
    ds = make_events_dataset(ARGS.rows, ARGS.blocks)
    P(f"\n[timing] dataset: {ARGS.rows:,} rows x {ARGS.blocks} blocks "
      f"({gib(nbytes(ds)):.2f} GiB)")

    P("warming up (uncounted)...")
    run_pipeline(ds, gpu=False, timings={})
    run_pipeline(ds, gpu=True, timings={})

    cpu_total, cpu_st = timed_best(ds, gpu=False, trials=ARGS.trials)
    gpu_total, gpu_st = timed_best(ds, gpu=True, trials=ARGS.trials)

    gpu_label = "GPU" if HAS_GPU else "GPU(fallback)"
    P("\nper-stage wall, RAM -> RAM (best end-to-end run of each):")
    P(f"  {'stage':<12} {'CPU (s)':>10} {gpu_label + ' (s)':>16} {'speedup':>9}")
    for key in ["1_impute", "2_derive", "3_sort", "4_encode"]:
        c, g = cpu_st.get(key, 0.0), gpu_st.get(key, 0.0)
        spd = (c / g) if g else float("nan")
        P(f"  {key:<12} {c:>10.2f} {g:>16.2f} {spd:>8.2f}x")
    P("  " + "-" * 48)
    spd_total = (cpu_total / gpu_total) if gpu_total else float("nan")
    P(f"  {'TOTAL e2e':<12} {cpu_total:>10.2f} {gpu_total:>16.2f} {spd_total:>8.2f}x")

    # ---- release the detached GPU sort actors ---------------------------- #
    if HAS_GPU:
        _release_sort_actors()
    ray.shutdown()


if __name__ == "__main__":
    main()
