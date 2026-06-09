"""Scan GpuOrdinalEncoder batch size on the post-impute 30-day CriteoPrivateAd
data, INDEPENDENTLY of the GPU sort sizing.

The encoder's block/partition sizing (``RAY_DATA_GPU_PREPROC_BATCH_SIZE``) has a
different optimum than the GPU sort's per-rank partition sizing, so it is tuned
on its own here. We build the post-impute dataset ONCE -- exactly the pipeline
that feeds the encoder in the main run:

    read -> prep -> GPU sort -> post-sort repartition -> CPU impute

then time ``GpuOrdinalEncoder`` fit + transform for each candidate batch size,
reporting fit/transform/total wall, rows/s, output block count, and peak GPU
memory (and any OOM). The fastest successful batch size is the one to pass to
``bench_criteo_gpu_sort_encode.py --batch-size``.

Run:
    RAY_DATA_GPU_SORT_NUM_GPUS=16 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
      .venv/bin/python gpu_preproc_bench/scan_encoder_blocksize.py --days 1-30
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_ap = argparse.ArgumentParser()
_ap.add_argument("--days", default="1-30")
_ap.add_argument("--gpus", type=int, default=16)
_ap.add_argument("--batch-sizes", default="250000,500000,1000000,2000000,4000000,8000000")
_ap.add_argument("--blocks", type=int, default=None,
                 help="post-sort repartition target; default = read block count")
_ap.add_argument("--rows", type=int, default=0, help="row cap for a quick smoke run")
_ap.add_argument("--trials", type=int, default=1, help="timed trials per batch size (best reported)")
_ap.add_argument("--object-store-gb", type=float, default=None)
_ap.add_argument("--out", default=None, help="optional JSON results path (written incrementally)")
_ap.add_argument("--cache-dir", default=None,
                 help="parquet dir for the post-impute dataset: read it if present "
                 "(skip the expensive build), else build and write it. Lets the scan "
                 "run in resilient chunks (subset --batch-sizes per invocation).")
_ap.add_argument("--no-warmup", action="store_true",
                 help="skip the uncounted warmup (use when resuming from --cache-dir)")
ARGS = _ap.parse_args()

os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(ARGS.gpus)
os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(ARGS.gpus)
os.environ["RAY_DATA_GPU_SORT_RELEASE"] = "1"
os.environ.setdefault("RAY_enable_open_telemetry", "0")
# Scan measures clean walls -> no per-batch profiling sync overhead.
os.environ["RAY_DATA_GPU_PREPROC_PROFILE"] = "0"

import criteo  # noqa: E402
import ray  # noqa: E402

import bench_criteo_cpu_baseline as cpu_base  # noqa: E402
from bench_criteo_cpu_baseline import (  # noqa: E402
    add_indicators,
    prep_batch,
    _resolve_object_store_bytes,
)
from ray.data.preprocessors import GpuOrdinalEncoder, SimpleImputer  # noqa: E402
from ray.data._internal.planner import gpu_sort_general as G  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731
GIB = 1024 ** 3


class GpuMemSampler:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_mib = 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                vals = [float(x) for x in out.split() if x.strip()]
                if vals:
                    self.peak_mib = max(self.peak_mib, max(vals))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
        self.peak_mib = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.peak_mib


def _wait_for_gpus(gpus: int, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if ray.available_resources().get("GPU", 0.0) >= gpus - 0.5:
            return
        time.sleep(0.25)


def build_post_impute(days, roles, gpus, blocks_arg, cache_dir=None):
    """read -> prep -> GPU sort -> post-sort repartition -> CPU impute (the exact
    dataset the encoder sees in the main run). Materialized + returned.

    If ``cache_dir`` exists with parquet, read it back instead of rebuilding (so
    a chunked / resumed scan pays the expensive build only once); if it is given
    but empty, build then write it there."""
    import glob

    if cache_dir and glob.glob(os.path.join(cache_dir, "*.parquet")):
        P(f"  reading cached post-impute dataset from {cache_dir} ...")
        ds = ray.data.read_parquet(cache_dir).materialize()
        target_blocks = blocks_arg or ds.num_blocks()
        return ds, target_blocks

    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)
    indicator_cols = tuple(roles.indicator_cols)

    ds = criteo.read_ray_days(days)
    if ARGS.rows:
        ds = ds.limit(ARGS.rows)
    ds = ds.materialize()
    target_blocks = blocks_arg or ds.num_blocks()
    ds = ds.map_batches(
        lambda t: prep_batch(t, numeric_raw, categorical, list_features),
        batch_format="pyarrow", batch_size=None,
    ).materialize()

    _wait_for_gpus(gpus)
    ds = ds.sort(roles.sort_key, backend="gpu").materialize()
    ds = ds.repartition(target_blocks, shuffle=False).materialize()

    if indicator_cols:
        ds = ds.map_batches(
            lambda t: add_indicators(t, indicator_cols),
            batch_format="pyarrow", batch_size=None,
        )
    ds = SimpleImputer(columns=roles.impute_numeric, strategy="mean").fit_transform(ds)
    if roles.impute_categorical:
        ds = SimpleImputer(
            columns=roles.impute_categorical, strategy="most_frequent"
        ).fit_transform(ds)
    ds = ds.materialize()
    if cache_dir:
        import shutil

        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        P(f"  caching post-impute dataset -> {cache_dir} ...")
        ds.write_parquet(cache_dir)
        ds = ray.data.read_parquet(cache_dir).materialize()
        target_blocks = blocks_arg or ds.num_blocks()
    return ds, target_blocks


def scan_one(ds, categorical, bs: int, gpus: int) -> Dict[str, Any]:
    """One batch size: fit + transform (separately timed), best-of-trials."""
    os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(bs)
    fits: List[float] = []
    trans: List[float] = []
    out_blocks = None
    peak_mib = 0.0
    err = None
    sampler = GpuMemSampler().start()
    try:
        for _ in range(max(1, ARGS.trials)):
            _wait_for_gpus(gpus)
            enc = GpuOrdinalEncoder(columns=list(categorical))
            t0 = time.perf_counter()
            enc.fit(ds)
            fits.append(time.perf_counter() - t0)
            _wait_for_gpus(gpus)
            t0 = time.perf_counter()
            out = enc.transform(ds).materialize()
            trans.append(time.perf_counter() - t0)
            out_blocks = out.num_blocks()
    except Exception as e:
        err = f"{type(e).__name__}: {str(e).strip().splitlines()[-1][:200]}"
    finally:
        peak_mib = sampler.stop()
    res: Dict[str, Any] = {"batch_size": bs, "out_blocks": out_blocks,
                           "peak_mib": peak_mib, "error": err}
    if fits and trans and err is None:
        res["fit_s"] = min(fits)
        res["transform_s"] = min(trans)
        res["total_s"] = min(fits) + min(trans)
    return res


def main() -> None:
    import logging

    available = criteo.discover_days()
    days = criteo.parse_days(ARGS.days, available)
    days_label = f"{days[0]}-{days[-1]}" if len(days) > 1 else f"{days[0]}"
    batch_sizes = [int(x) for x in ARGS.batch_sizes.split(",") if x.strip()]

    osm = _resolve_object_store_bytes(ARGS.object_store_gb)
    ray.init(logging_level="ERROR", include_dashboard=False, object_store_memory=osm)
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    ctx.execution_options.preserve_order = True

    roles = criteo.column_roles_multi(days, null_indicator_threshold=0.01)

    P("=" * 80)
    P(f"GpuOrdinalEncoder batch-size scan  days={days_label}  gpus={ARGS.gpus}")
    P("=" * 80)
    P(f"categorical columns ({len(roles.categorical)}): {roles.categorical}")
    P("building post-impute dataset once (read -> prep -> GPU sort -> "
      "post-sort repartition -> CPU impute) ...")
    t0 = time.perf_counter()
    ds, target_blocks = build_post_impute(days, roles, ARGS.gpus, ARGS.blocks,
                                          ARGS.cache_dir)
    rows = ds.count()
    P(f"  post-impute dataset: {rows:,} rows, {ds.num_blocks()} blocks "
      f"(target {target_blocks}), ready in {time.perf_counter() - t0:.1f}s")
    _wait_for_gpus(ARGS.gpus)

    # Warmup (uncounted) so the first measured batch size is not penalized by
    # one-time CUDA/import costs (skippable when resuming from a cache).
    if not ARGS.no_warmup:
        P("warmup (uncounted) ...")
        try:
            GpuOrdinalEncoder(columns=list(roles.categorical)).fit_transform(ds).materialize()
        except Exception as e:
            P(f"  warmup error (continuing): {e}")
        _wait_for_gpus(ARGS.gpus)

    def _dump(results_so_far):
        if not ARGS.out:
            return
        with open(ARGS.out, "w") as fh:
            json.dump({"days": days, "rows": rows, "gpus": ARGS.gpus,
                       "target_blocks": target_blocks,
                       "results": results_so_far}, fh, indent=2)

    P(f"\nscanning batch sizes: {batch_sizes}")
    P(f"  {'batch_size':>11} {'fit(s)':>8} {'xform(s)':>9} {'total(s)':>9} "
      f"{'Mrows/s':>8} {'out_blk':>8} {'peakGPU(GiB)':>12}  result")
    results: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        r = scan_one(ds, roles.categorical, bs, ARGS.gpus)
        results.append(r)
        _dump(results)  # incremental: survive a kill / allow resume
        if r.get("error"):
            P(f"  {bs:>11,} {'--':>8} {'--':>9} {'--':>9} {'--':>8} "
              f"{str(r.get('out_blocks')):>8} {r['peak_mib'] / 1024:>12.2f}  "
              f"FAIL {r['error']}")
        else:
            mrows = rows / r["total_s"] / 1e6 if r["total_s"] else 0.0
            P(f"  {bs:>11,} {r['fit_s']:>8.2f} {r['transform_s']:>9.2f} "
              f"{r['total_s']:>9.2f} {mrows:>8.2f} {r['out_blocks']:>8} "
              f"{r['peak_mib'] / 1024:>12.2f}  ok")

    ok = [r for r in results if not r.get("error") and r.get("total_s")]
    P("\n" + "-" * 80)
    if ok:
        best = min(ok, key=lambda r: r["total_s"])
        P(f"FASTEST encoder batch size: {best['batch_size']:,}  "
          f"(total {best['total_s']:.2f}s, fit {best['fit_s']:.2f}s, "
          f"transform {best['transform_s']:.2f}s, out_blocks {best['out_blocks']}, "
          f"peak {best['peak_mib'] / 1024:.2f} GiB)")
        P(f"-> run: bench_criteo_gpu_sort_encode.py --days {days_label} "
          f"--batch-size {best['batch_size']}")
        if best["out_blocks"] is not None and best["out_blocks"] < 0.5 * target_blocks:
            P(f"   note: out_blocks {best['out_blocks']} < target/2 "
              f"({target_blocks}) -> the main run will post-encode-repartition.")
    else:
        P("No batch size succeeded (all OOM/failed). Try smaller sizes or more GPUs.")

    if ARGS.out:
        with open(ARGS.out, "w") as fh:
            json.dump({"days": days, "rows": rows, "gpus": ARGS.gpus,
                       "target_blocks": target_blocks, "results": results}, fh, indent=2)
        P(f"wrote scan results -> {ARGS.out}")

    G.kill_actor_pool(ARGS.gpus)
    ray.shutdown()


if __name__ == "__main__":
    main()
