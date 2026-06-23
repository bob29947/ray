"""Standalone StandardScaler GPU-vs-CPU benchmark for CriteoPrivateAd.

Isolates the scaler. ``read -> prep -> materialize`` is built ONCE and cached in
RAM (untimed); then ONLY ``StandardScaler`` (fit + transform + materialize back
to RAM) is timed. This models an isolated GPU StandardScaler dropped into a real
pipeline: everything upstream is already materialized, the scaler runs, and the
clock stops when all data is scaled and back in RAM.

ALL scaler overhead is inside the timed region -- the mean/std fit reduction, the
GPU actor-pool / CUDA-context / cuDF startup, H2D/D2H transfer, compute, and the
final materialize. ``--warmup`` defaults to 0 so the cold start counts; every
repeat re-creates the actor pools, so each pays setup. The headline per variant
is ``total_s`` (fit + transform + materialize); fit and transform are also
reported separately for insight.

Columns scaled = exactly what the full pipeline scales (``roles.numeric_features``;
44 on the lean recipe = 43 float features + 1 list-length feature). Raw numerics
may contain nulls (the full pipeline imputes upstream); StandardScaler ignores
nulls in both fit and transform on CPU and GPU, so this is a valid throughput
benchmark for the scaler in isolation.

CPU baseline: ``StandardScaler`` with default settings (natural batch size /
concurrency), bounded by ``--cpus``. GPU sweep: device batch size x actor count x
GPU fraction. The actor-count axis is realized via fractional GPUs on the pinned
GPU set (e.g. 4 actors @1.0 = 1/GPU, 8 @0.5 = 2/GPU, 16 @0.25 = 4/GPU), using the
``RAY_DATA_GPU_PREPROC_GPU_FRACTION`` knob. A standalone scaler is light compute
(one subtract+divide per column), so unlike the compute-bound fused stage it is a
candidate to benefit from large batches + fractional packing.

Resources are pinned: ``--cpus 64 --gpus 4`` by default (CPU baseline uses only
CPUs; the GPU sweep uses the 4 GPUs). GPU runs need a real CUDA device, so run
this OUTSIDE any sandbox.

Run (quick smoke):
    .venv/bin/python benchmarks/criteo/bench_scaler_local.py --days 1 --rows 2000000

Run (real sweep, a few days):
    .venv/bin/python benchmarks/criteo/bench_scaler_local.py --days 1-3
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import criteo  # noqa: E402
import ray  # noqa: E402

import cpu_pipeline as cpu  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731

_GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Build the cached read -> prep -> materialize input (ONCE, untimed)
# --------------------------------------------------------------------------- #
def _build_cached_input(args, days: List[int], roles):
    """read -> prep -> materialize, cached in RAM. Same as the pipeline's load +
    prep (so the list-length numeric columns exist and the frame is pruned to the
    pipeline's kept columns), but WITHOUT sort / indicators -- the scaler needs
    neither. This is the 'materialize everything before the scaler' step and is
    deliberately NOT timed."""
    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)

    t0 = time.perf_counter()
    ds = criteo.read_ray_days(days)
    if args.rows:
        ds = ds.limit(args.rows)
    ds = ds.map_batches(
        lambda t: cpu.prep_batch(t, numeric_raw, categorical, list_features),
        batch_format="pyarrow",
        batch_size=None,
    )
    ds = ds.materialize()
    dt = time.perf_counter() - t0

    rows, ncols, gib = cpu.ds_stats(ds)
    bytes_per_row = (gib * _GiB / rows) if rows else 0.0
    return ds, {
        "rows": rows,
        "cols": ncols,
        "gib": round(gib, 2),
        "bytes_per_row": round(bytes_per_row, 1),
        "blocks": ds.num_blocks(),
        "build_s": round(dt, 1),
    }


# --------------------------------------------------------------------------- #
# Fitted-stat sampling + parity (a tuning win must not be a correctness regress)
# --------------------------------------------------------------------------- #
def _collect_fitted(scaler, cols: List[str]) -> Dict[str, List[Any]]:
    """Sample a few columns' fitted (mean, std) for a CPU-vs-GPU parity check."""
    out: Dict[str, List[Any]] = {}
    for c in cols[:5]:
        out[c] = [scaler.stats_.get(f"mean({c})"), scaler.stats_.get(f"std({c})")]
    return out


def _close(a, b, rtol=1e-6, atol=1e-6) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    except (TypeError, ValueError):
        return a == b


def _parity(cpu_fit: Optional[Dict], gpu_fit: Optional[Dict]) -> Dict[str, Any]:
    if not cpu_fit or not gpu_fit:
        return {"ok": False, "detail": "missing fitted sample"}
    issues: List[str] = []
    for c, (m, s) in cpu_fit.items():
        gm, gs = (gpu_fit.get(c) or [None, None])
        if not _close(m, gm) or not _close(s, gs):
            issues.append(f"{c}: cpu=({m},{s}) gpu=({gm},{gs})")
    return {"ok": not issues, "detail": "ok" if not issues else "; ".join(issues[:4])}


# --------------------------------------------------------------------------- #
# Variant runners
# --------------------------------------------------------------------------- #
def _resolve_batch(spec: str, ds, num_gpus: int) -> int:
    from ray.data.preprocessors import _gpu

    if str(spec).lower() == "auto":
        return _gpu.auto_gpu_block_rows(ds, num_gpus=num_gpus)
    return int(spec)


def _run_cpu(cached_ds, cols: List[str], *, warmup: int, repeats: int) -> Dict[str, Any]:
    """Time the CPU StandardScaler with default settings (natural concurrency)."""
    from ray.data.preprocessors import StandardScaler

    tot_s: List[float] = []
    rows = 0
    fitted: Optional[Dict] = None
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        scaler = StandardScaler(columns=cols)
        out = scaler.fit_transform(cached_ds).materialize()
        rows = out.count()
        t1 = time.perf_counter()
        del out
        if i >= warmup:
            tot_s.append(t1 - t0)
            if fitted is None:
                fitted = _collect_fitted(scaler, cols)
    return {
        "kind": "cpu",
        "total_s": round(statistics.median(tot_s), 3),
        "total_s_all": [round(x, 3) for x in tot_s],
        "rows": rows,
        "mrows_per_s": round(rows / statistics.median(tot_s) / 1e6, 2) if tot_s else None,
        "fitted_sample": fitted,
    }


def _run_gpu_variant(
    cached_ds,
    cols: List[str],
    *,
    fraction: float,
    actors: int,
    batch_rows: int,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    """Time the GPU scaler (fit + transform + materialize) for one config.

    Sets the env knobs that both the fit reduction (``gpu_mean_std``) and the
    transform (``run_fused_device_transform`` -> ``gpu_transform``) read, so a
    single (fraction, actors, batch) triple drives the whole scaler op. All
    overhead is inside the timed region; ``--warmup`` defaults to 0.
    """
    from ray.data.preprocessors.gpu_scaler import GpuStandardScaler

    os.environ["RAY_DATA_GPU_PREPROC_GPU_FRACTION"] = str(fraction)
    os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(actors)
    os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(batch_rows)

    fit_s: List[float] = []
    tr_s: List[float] = []
    tot_s: List[float] = []
    rows = 0
    fitted: Optional[Dict] = None

    for i in range(warmup + repeats):
        scaler = GpuStandardScaler(columns=cols)
        t0 = time.perf_counter()
        scaler.fit(cached_ds)
        t1 = time.perf_counter()
        out = scaler.transform(cached_ds).materialize()
        rows = out.count()
        t2 = time.perf_counter()
        del out
        if i >= warmup:
            fit_s.append(t1 - t0)
            tr_s.append(t2 - t1)
            tot_s.append(t2 - t0)
            if fitted is None:
                fitted = _collect_fitted(scaler, cols)

    med_tot = statistics.median(tot_s)
    return {
        "kind": "gpu",
        "fraction": fraction,
        "actors": actors,
        "actors_per_gpu": round(1.0 / fraction, 2) if fraction else None,
        "batch_rows": batch_rows,
        "fit_s": round(statistics.median(fit_s), 3),
        "transform_s": round(statistics.median(tr_s), 3),
        "total_s": round(med_tot, 3),
        "total_s_all": [round(x, 3) for x in tot_s],
        "rows": rows,
        "mrows_per_s": round(rows / med_tot / 1e6, 2) if med_tot else None,
        "fitted_sample": fitted,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _label(v: Dict[str, Any]) -> str:
    return (
        f"frac={v['fraction']:<4} actors={v['actors']:<3} "
        f"({v['actors_per_gpu']:g}/gpu) batch={v['batch_rows'] / 1e6:.2f}M"
    )


def _print_report(input_meta, cpu_res, gpu_results, args) -> None:
    P("\n" + "=" * 100)
    P("STANDALONE StandardScaler BENCHMARK -- GPU sweep vs CPU baseline")
    P("=" * 100)
    P(f"cached input   : {input_meta['rows']:,} rows x {input_meta['cols']} cols  "
      f"{input_meta['gib']} GiB  ({input_meta['bytes_per_row']} B/row, "
      f"{input_meta['blocks']} blocks, built in {input_meta['build_s']}s)")
    P(f"scaling        : {input_meta['n_scaled']} columns "
      f"(roles.numeric_features, {args.feature_set} recipe)")
    P(f"resources      : {args.gpus} GPU(s) + {args.cpus} CPU(s)   "
      f"days={args.days}   repeats={args.repeats} (warmup={args.warmup})")
    base = cpu_res["total_s"] if cpu_res else None
    if cpu_res:
        P(f"\nCPU baseline   : {cpu_res['total_s']:.2f}s  "
          f"({cpu_res['mrows_per_s']} Mrows/s)  (runs={cpu_res['total_s_all']})")
    P("\n" + "-" * 100)
    P(f"{'variant':<42} {'fit_s':>7} {'xform_s':>8} {'total_s':>8} "
      f"{'Mrows/s':>9} {'vs CPU':>8}")
    P("-" * 100)
    best = None
    for v in gpu_results:
        if v.get("total_s") is None:  # errored / skipped variant
            tag = v.get("error", v.get("skipped", "n/a"))
            P(f"{_label(v):<42} {'--':>7} {'--':>8} {'--':>8} {'--':>9} {'--':>8}  {tag}")
            continue
        spd = f"{base / v['total_s']:.2f}x" if base and v["total_s"] else "-"
        P(f"{_label(v):<42} {v['fit_s']:>7.2f} {v['transform_s']:>8.2f} "
          f"{v['total_s']:>8.2f} {v['mrows_per_s']:>9} {spd:>8}")
        if best is None or v["total_s"] < best["total_s"]:
            best = v
    P("-" * 100)
    if best:
        spd = f"  ({base / best['total_s']:.2f}x vs CPU)" if base else ""
        P(f"FASTEST GPU    : {_label(best)} -> {best['total_s']:.2f}s "
          f"({best['mrows_per_s']} Mrows/s){spd}")
    P("=" * 100)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", default="1-3", help="day(s): '1' | '1-5' | 'all'")
    ap.add_argument("--rows", type=int, default=0, help="row cap (0 = full)")
    ap.add_argument("--data-root", default=None, help="dataset root (local path or s3://)")
    ap.add_argument("--feature-set", choices=["lean", "wide"], default="lean")
    ap.add_argument("--null-indicator-threshold", type=float, default=0.01)
    ap.add_argument("--object-store-gb", type=float, default=None)

    ap.add_argument("--cpus", type=int, default=64, help="CPUs for ray.init")
    ap.add_argument("--gpus", type=int, default=4, help="physical GPUs for ray.init")

    ap.add_argument("--gpu-fractions", default="1.0,0.5,0.25",
                    help="comma list of GPU fractions per actor (1.0 = 1 actor/GPU)")
    ap.add_argument("--gpu-actors", default="4,8,16",
                    help="comma list of actor counts (concurrency)")
    ap.add_argument("--gpu-batch-sizes", default="auto,1000000,2000000,4000000",
                    help="comma list of device batch rows; 'auto' = VRAM-aware sizer")

    ap.add_argument("--repeats", type=int, default=3, help="timed GPU repeats per variant")
    ap.add_argument("--warmup", type=int, default=0,
                    help="discarded GPU warmup runs (0 keeps cold-start overhead in the number)")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU baseline")
    ap.add_argument("--cpu-repeats", type=int, default=1)
    ap.add_argument("--cpu-warmup", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON results path")
    args = ap.parse_args()

    if args.data_root:
        criteo.set_data_root(args.data_root)
    available = criteo.discover_days()
    days = criteo.parse_days(args.days, available)

    here = os.path.dirname(os.path.abspath(__file__))

    import logging

    osm_bytes = cpu._resolve_object_store_bytes(args.object_store_gb)
    ray.init(
        num_cpus=args.cpus,
        num_gpus=args.gpus,
        logging_level="ERROR",
        include_dashboard=False,
        object_store_memory=osm_bytes,
    )
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    ctx.execution_options.preserve_order = True

    n_gpus_cluster = int(ray.cluster_resources().get("GPU", 0))
    if n_gpus_cluster == 0:
        P("WARNING: no GPUs visible to Ray; the GPU scaler will fall back to CPU.")

    roles = criteo.column_roles_multi(
        days,
        null_indicator_threshold=args.null_indicator_threshold,
        feature_set=args.feature_set,
    )
    cols = list(roles.numeric_features)

    P("=" * 100)
    P("building cached read -> prep -> materialize input (once, untimed) ...")
    P(f"  days={days}  rows_cap={args.rows or 'full'}  feature_set={args.feature_set}  "
      f"GPUs={n_gpus_cluster}  CPUs={args.cpus}")
    sorted_ds, input_meta = _build_cached_input(args, days, roles)
    input_meta["n_scaled"] = len(cols)
    P(f"  done: {input_meta['rows']:,} rows, {input_meta['cols']} cols, "
      f"{input_meta['gib']} GiB, {input_meta['blocks']} blocks in {input_meta['build_s']}s")
    P(f"  scaling {len(cols)} columns (roles.numeric_features): {cols}")

    cpu_res = None
    if not args.no_cpu:
        P("\nrunning CPU StandardScaler baseline (default settings) ...")
        cpu_res = _run_cpu(
            sorted_ds, cols, warmup=args.cpu_warmup, repeats=args.cpu_repeats
        )
        P(f"  CPU baseline median: {cpu_res['total_s']:.2f}s "
          f"({cpu_res['mrows_per_s']} Mrows/s)")

    fractions = [float(x) for x in args.gpu_fractions.split(",") if x.strip()]
    actor_counts = [int(x) for x in args.gpu_actors.split(",") if x.strip()]
    batch_specs = [s.strip() for s in args.gpu_batch_sizes.split(",") if s.strip()]

    gpu_results: List[Dict[str, Any]] = []
    variants = list(itertools.product(fractions, actor_counts, batch_specs))
    P(f"\nsweeping GPU variants (fraction x actors x batch), capacity guard "
      f"actors*fraction <= {args.gpus} GPUs ...")
    for fraction, actors, batch_spec in variants:
        gpus_needed = math.ceil(actors * fraction)
        if gpus_needed > args.gpus:
            P(f"  -> SKIP frac={fraction} actors={actors} "
              f"(needs {gpus_needed} > {args.gpus} GPUs)")
            gpu_results.append({
                "kind": "gpu", "fraction": fraction, "actors": actors,
                "actors_per_gpu": round(1.0 / fraction, 2) if fraction else None,
                "batch_rows": -1, "total_s": None,
                "skipped": f"needs {gpus_needed}>{args.gpus} gpus",
            })
            continue
        batch_rows = _resolve_batch(batch_spec, sorted_ds, actors)
        P(f"  -> frac={fraction} actors={actors} batch={batch_rows:,} ...")
        try:
            res = _run_gpu_variant(
                sorted_ds, cols,
                fraction=fraction, actors=actors, batch_rows=batch_rows,
                warmup=args.warmup, repeats=args.repeats,
            )
            P(f"     fit={res['fit_s']:.2f}s transform={res['transform_s']:.2f}s "
              f"total={res['total_s']:.2f}s ({res['mrows_per_s']} Mrows/s)"
              + (f"  ({cpu_res['total_s'] / res['total_s']:.2f}x vs CPU)"
                 if cpu_res else ""))
        except Exception as e:  # keep the sweep going if one config errors
            res = {
                "kind": "gpu", "fraction": fraction, "actors": actors,
                "actors_per_gpu": round(1.0 / fraction, 2) if fraction else None,
                "batch_rows": batch_rows, "total_s": None,
                "error": f"{type(e).__name__}: {e}".splitlines()[0][:200],
            }
            P(f"     FAILED: {res['error']}")
        gpu_results.append(res)

    _print_report(input_meta, cpu_res, gpu_results, args)

    # --- parity (sampled fitted stats: each GPU variant vs CPU) ------------- #
    parity = None
    if cpu_res and cpu_res.get("fitted_sample"):
        parity = []
        P("\nparity (sampled fitted mean/std, each GPU variant vs CPU):")
        for v in gpu_results:
            if v.get("total_s") is None:
                continue
            res = _parity(cpu_res["fitted_sample"], v.get("fitted_sample"))
            res["variant"] = _label(v)
            parity.append(res)
            P(f"  {'MATCH   ' if res['ok'] else 'MISMATCH'}  {_label(v)}"
              + ("" if res["ok"] else f"   {res['detail']}"))

    out_path = args.out or os.path.join(
        here, "data", f"bench_scaler_local_{int(time.time())}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                "days": days,
                "rows_cap": args.rows,
                "feature_set": args.feature_set,
                "n_scaled": len(cols),
                "scaled_columns": cols,
                "gpus": args.gpus,
                "cpus": args.cpus,
                "input": input_meta,
                "cpu": cpu_res,
                "gpu": gpu_results,
                "parity": parity,
                "object_store_gib": round(osm_bytes / _GiB, 1),
            },
            fh,
            indent=2,
            default=cpu._jsonable,
        )
    P(f"\nwrote results -> {out_path}")
    ray.shutdown()


if __name__ == "__main__":
    main()
