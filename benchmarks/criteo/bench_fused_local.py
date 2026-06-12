"""Local fused GPU-vs-CPU profiling harness for the CriteoPrivateAd benchmark.

The fused ``impute + encode + scale`` device stage is the ONLY thing this script
optimizes; everything before it (read -> prep -> sort -> add indicators) is CPU
and identical for both targets, so it is materialized **once** and cached. Each
benchmark variant then runs only the cheap-to-vary fused work against that same
cached input, which is what makes "iterate over and over" fast.

For every variant the harness times the GPU fused **fit** and **transform**
separately (calling the same internal ``fused_fit`` / ``run_fused_device_transform``
that ``Chain(..., backend="gpu")`` uses), reads the per-worker H2D / compute /
D2H device-time split, and compares the fused total against the CPU
``impute + encode + scale`` subtotal run on the identical cached dataset.

Design notes / guardrails:

* **Ray read/sort blocks stay automatic** -- the harness never repartitions. The
  only sizing it sweeps is the GPU device batch (rows per fused block) and the
  number of one-GPU fused workers.
* **Cluster-shape emulation** (``--emulate-cluster``): the DGX box's GPUs (V100,
  32 GB, NVLink) are not the deploy target (L4, 24 GB, PCIe). Emulation pins the
  worker count to ``--cluster-gpus`` and caps per-GPU VRAM to ``--cluster-vram-gb``
  (via ``RAY_DATA_GPU_PREPROC_VRAM_BYTES``) so the auto device batch matches what
  the cluster would pick. Treat local numbers as directional; confirm on AWS.
* GPU runs need a real CUDA device, so run this OUTSIDE any sandbox.

Run (quick local loop, a few days, default sweep):
    .venv/bin/python benchmarks/criteo/bench_fused_local.py \
        --days 1-3 \
        --data-root /bobbwang/datasets/CriteoPrivateAd/data

Run (emulate the 8x L4 cluster shape, A/B the one-scan fit):
    .venv/bin/python benchmarks/criteo/bench_fused_local.py \
        --days 1-5 --emulate-cluster --cluster-gpus 8 \
        --gpu-batch-sizes auto --fit-modes three,one
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import shutil
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import criteo  # noqa: E402
import ray  # noqa: E402

import cpu_pipeline as cpu  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731

_GiB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Preprocessor op construction (identical order to cpu_pipeline / gpu_pipeline)
# --------------------------------------------------------------------------- #
def _build_ops(roles) -> List[Any]:
    """The impute+encode+scale ops, in pipeline order, as CPU classes.

    For the CPU subtotal they are used directly; for the GPU fused run each is
    upgraded to its device-fusable ``Gpu*`` counterpart.
    """
    from ray.data.preprocessors import (
        OrdinalEncoder,
        SimpleImputer,
        StandardScaler,
    )

    ops: List[Any] = [SimpleImputer(columns=roles.impute_numeric, strategy="mean")]
    if roles.impute_categorical:
        ops.append(
            SimpleImputer(columns=roles.impute_categorical, strategy="most_frequent")
        )
    ops.append(OrdinalEncoder(columns=roles.categorical))
    ops.append(StandardScaler(columns=roles.numeric_features))
    return ops


def _collect_fitted(ops: List[Any], roles) -> Dict[str, Any]:
    """Pull a small, comparable sample of fitted stats for a parity check.

    Same ``stats_`` keys on the CPU and GPU ops, so a CPU-vs-GPU diff of this
    dict catches any fused-fit regression without dumping every column.
    """
    from ray.data.preprocessors import (
        OrdinalEncoder,
        SimpleImputer,
        StandardScaler,
    )

    out: Dict[str, Any] = {"scaler": {}, "encoder_vocab": {}, "imputer_mean": {}}
    for op in ops:
        if op is None:
            continue
        if isinstance(op, StandardScaler):
            for c in roles.numeric_features[:5]:
                key_m, key_s = f"mean({c})", f"std({c})"
                if key_m in op.stats_:
                    out["scaler"][c] = [op.stats_.get(key_m), op.stats_.get(key_s)]
        elif isinstance(op, OrdinalEncoder):
            for c in roles.categorical[:5]:
                k = f"unique_values({c})"
                if k in op.stats_:
                    out["encoder_vocab"][c] = cpu._vocab_size(op.stats_[k])
        elif isinstance(op, SimpleImputer) and op.strategy == "mean":
            for c in roles.impute_numeric[:5]:
                k = f"mean({c})"
                if k in op.stats_:
                    out["imputer_mean"][c] = op.stats_[k]
    return out


# --------------------------------------------------------------------------- #
# Profile (H2D / compute / D2H) capture
# --------------------------------------------------------------------------- #
def _clear_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _read_profile(profile_dir: str) -> Optional[Dict[str, Any]]:
    """Aggregate the per-worker profile JSONs written during the transform."""
    h2d = compute = d2h = 0.0
    blocks = workers = 0
    for f in glob.glob(os.path.join(profile_dir, "prof_*.json")):
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception:
            continue
        h2d += float(d.get("h2d", 0.0))
        compute += float(d.get("compute", 0.0))
        d2h += float(d.get("d2h", 0.0))
        blocks += int(d.get("n", 0))
        workers += 1
    tot = h2d + compute + d2h
    if tot <= 0:
        return None
    return {
        "workers": workers,
        "blocks": blocks,
        "h2d_s": round(h2d, 3),
        "compute_s": round(compute, 3),
        "d2h_s": round(d2h, 3),
        "h2d_pct": round(100 * h2d / tot, 1),
        "compute_pct": round(100 * compute / tot, 1),
        "d2h_pct": round(100 * d2h / tot, 1),
    }


# --------------------------------------------------------------------------- #
# Variant runners
# --------------------------------------------------------------------------- #
def _resolve_batch(spec: str, sorted_ds, num_gpus: int) -> int:
    from ray.data.preprocessors import _gpu

    if str(spec).lower() == "auto":
        return _gpu.auto_gpu_block_rows(sorted_ds, num_gpus=num_gpus)
    return int(spec)


def _run_gpu_variant(
    sorted_ds,
    roles,
    *,
    num_gpus: int,
    batch_rows: int,
    fit_mode: str,
    profile_dir: str,
    warmup: int,
    repeats: int,
) -> Dict[str, Any]:
    """Time the GPU fused fit + transform for one config (warmup + median-of-N)."""
    from ray.data.preprocessors._gpu_fused import (
        fused_fit,
        run_fused_device_transform,
        upgrade_to_device_op,
    )

    os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(num_gpus)
    os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(batch_rows)
    # One-scan vs three-scan fit.
    os.environ["RAY_DATA_GPU_PREPROC_FUSED_FIT"] = "1" if fit_mode == "one" else "0"
    fit_conc = num_gpus
    xform_conc = num_gpus
    os.environ["RAY_DATA_GPU_PREPROC_PROFILE"] = "1"
    os.environ["RAY_DATA_GPU_PREPROC_PROFILE_DIR"] = profile_dir

    fit_s: List[float] = []
    tr_s: List[float] = []
    tot_s: List[float] = []
    rows = 0
    fitted: Optional[Dict[str, Any]] = None
    last_profile: Optional[Dict[str, Any]] = None

    for i in range(warmup + repeats):
        _clear_dir(profile_dir)
        ops = [upgrade_to_device_op(o) for o in _build_ops(roles)]
        t0 = time.perf_counter()
        fused_fit(sorted_ds, ops, batch_size=batch_rows, concurrency=fit_conc)
        t1 = time.perf_counter()
        out = run_fused_device_transform(
            sorted_ds, ops, batch_size=batch_rows, concurrency=xform_conc
        ).materialize()
        rows = out.count()
        t2 = time.perf_counter()
        del out
        if i >= warmup:
            fit_s.append(t1 - t0)
            tr_s.append(t2 - t1)
            tot_s.append(t2 - t0)
            last_profile = _read_profile(profile_dir)
            if fitted is None:
                fitted = _collect_fitted(ops, roles)

    return {
        "kind": "gpu",
        "num_gpus": num_gpus,
        "batch_rows": batch_rows,
        "fit_mode": fit_mode,
        "xform_concurrency": xform_conc,
        "fit_s": round(statistics.median(fit_s), 3),
        "transform_s": round(statistics.median(tr_s), 3),
        "total_s": round(statistics.median(tot_s), 3),
        "fit_s_all": [round(x, 3) for x in fit_s],
        "total_s_all": [round(x, 3) for x in tot_s],
        "rows": rows,
        "profile": last_profile,
        "fitted_sample": fitted,
    }


def _run_cpu_subtotal(sorted_ds, roles, *, warmup: int, repeats: int) -> Dict[str, Any]:
    """Time the CPU impute+encode+scale subtotal on the identical cached input."""
    from ray.data.preprocessors import (
        OrdinalEncoder,
        SimpleImputer,
        StandardScaler,
    )

    tot_s: List[float] = []
    rows = 0
    fitted: Optional[Dict[str, Any]] = None

    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        ds = sorted_ds
        num_imp = SimpleImputer(columns=roles.impute_numeric, strategy="mean")
        ds = num_imp.fit_transform(ds)
        cat_imp = None
        if roles.impute_categorical:
            cat_imp = SimpleImputer(
                columns=roles.impute_categorical, strategy="most_frequent"
            )
            ds = cat_imp.fit_transform(ds)
        enc = OrdinalEncoder(columns=roles.categorical)
        ds = enc.fit_transform(ds)
        sca = StandardScaler(columns=roles.numeric_features)
        ds = sca.fit_transform(ds).materialize()
        rows = ds.count()
        t1 = time.perf_counter()
        del ds
        if i >= warmup:
            tot_s.append(t1 - t0)
            if fitted is None:
                fitted = _collect_fitted([num_imp, cat_imp, enc, sca], roles)

    return {
        "kind": "cpu",
        "total_s": round(statistics.median(tot_s), 3),
        "total_s_all": [round(x, 3) for x in tot_s],
        "rows": rows,
        "fitted_sample": fitted,
    }


# --------------------------------------------------------------------------- #
# Build the cached read -> prep -> sort -> indicators input (ONCE)
# --------------------------------------------------------------------------- #
def _build_sorted_input(args, roles):
    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)
    indicator_cols = tuple(roles.indicator_cols)

    t0 = time.perf_counter()
    ds = criteo.read_ray_days(roles.days)
    if args.rows:
        ds = ds.limit(args.rows)
    ds = ds.map_batches(
        lambda t: cpu.prep_batch(t, numeric_raw, categorical, list_features),
        batch_format="pyarrow",
        batch_size=None,
    )
    ds = ds.sort(roles.sort_key)
    if indicator_cols:
        ds = ds.map_batches(
            lambda t: cpu.add_indicators(t, indicator_cols),
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
# Reporting
# --------------------------------------------------------------------------- #
def _label(v: Dict[str, Any]) -> str:
    if v["kind"] == "cpu":
        return "CPU impute+encode+scale"
    return (
        f"GPU g={v['num_gpus']:<2} batch={v['batch_rows']/1e6:.2f}M "
        f"fit={v['fit_mode']:<5}"
    )


def _print_report(input_meta, cpu_res, gpu_results, args) -> None:
    P("\n" + "=" * 92)
    P("FUSED STAGE BENCHMARK -- GPU fused impute+encode+scale vs CPU subtotal")
    P("=" * 92)
    P(f"cached input   : {input_meta['rows']:,} rows x {input_meta['cols']} cols  "
      f"{input_meta['gib']} GiB  ({input_meta['bytes_per_row']} B/row, "
      f"{input_meta['blocks']} blocks, built in {input_meta['build_s']}s)")
    P(f"feature set    : {args.feature_set}   days={args.days}   "
      f"repeats={args.repeats} (warmup={args.warmup})")
    if args.emulate_cluster:
        P(f"emulation      : cluster shape -> {args.cluster_gpus} GPUs, "
          f"{args.cluster_vram_gb} GB/GPU VRAM (auto batch is L4-representative)")
    base = cpu_res["total_s"] if cpu_res else None
    if cpu_res:
        P(f"\nCPU subtotal   : {cpu_res['total_s']:.2f}s  "
          f"(runs={cpu_res['total_s_all']})")
    P("\n" + "-" * 92)
    P(f"{'variant':<40} {'fit_s':>8} {'xform_s':>8} {'total_s':>8} "
      f"{'vs CPU':>8}  {'device split (compute/H2D/D2H)':>28}")
    P("-" * 92)
    best = None
    for v in gpu_results:
        if v.get("total_s") is None:  # errored variant
            P(f"{_label(v):<40} {'--':>8} {'--':>8} {'FAILED':>8}  "
              f"{v.get('error', 'error')}")
            continue
        spd = f"{base / v['total_s']:.2f}x" if base and v["total_s"] else "-"
        if v.get("profile"):
            pr = v["profile"]
            split = f"{pr['compute_pct']:.0f}/{pr['h2d_pct']:.0f}/{pr['d2h_pct']:.0f}%"
        else:
            split = "-"
        P(f"{_label(v):<40} {v['fit_s']:>8.2f} {v['transform_s']:>8.2f} "
          f"{v['total_s']:>8.2f} {spd:>8}  {split:>28}")
        if best is None or v["total_s"] < best["total_s"]:
            best = v
    P("-" * 92)
    if best:
        spd = f"{base / best['total_s']:.2f}x vs CPU" if base else ""
        P(f"FASTEST GPU    : {_label(best)}  -> {best['total_s']:.2f}s  {spd}")
    P("=" * 92)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", default="1-3", help="day(s): '1' | '1-5' | 'all'")
    ap.add_argument("--rows", type=int, default=0, help="row cap (0 = full)")
    ap.add_argument("--data-root", default=None, help="dataset root (local path or s3://)")
    ap.add_argument("--feature-set", choices=["lean", "wide"], default="lean",
                    help="lean = current pruned recipe; wide = keep pruned cols as features")
    ap.add_argument("--null-indicator-threshold", type=float, default=0.01)
    ap.add_argument("--object-store-gb", type=float, default=None)

    ap.add_argument("--gpu-num-gpus-list", default=None,
                    help="comma list of fused worker counts (default: cluster GPU total)")
    ap.add_argument("--gpu-batch-sizes", default="auto",
                    help="comma list of device batch rows; 'auto' = VRAM-aware sizer")
    ap.add_argument("--fit-modes", default="three",
                    help="comma list of fit modes: 'three' (per-kind scans), 'one' (single scan)")

    ap.add_argument("--repeats", type=int, default=3, help="timed GPU repeats per variant")
    ap.add_argument("--warmup", type=int, default=1, help="discarded GPU warmup runs per variant")
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU subtotal baseline")
    ap.add_argument("--cpu-repeats", type=int, default=1,
                    help="timed CPU subtotal repeats (it doesn't vary and is slow at scale)")
    ap.add_argument("--cpu-warmup", type=int, default=0, help="discarded CPU warmup runs")

    ap.add_argument("--emulate-cluster", action="store_true",
                    help="pin worker count + per-GPU VRAM to cluster shape (L4)")
    ap.add_argument("--cluster-gpus", type=int, default=8, help="GPUs to emulate")
    ap.add_argument("--cluster-vram-gb", type=float, default=24.0, help="per-GPU VRAM GB to emulate")
    ap.add_argument("--out", default=None, help="JSON results path")
    args = ap.parse_args()

    if args.data_root:
        criteo.set_data_root(args.data_root)
    available = criteo.discover_days()
    days = criteo.parse_days(args.days, available)

    here = os.path.dirname(os.path.abspath(__file__))

    import logging

    osm_bytes = cpu._resolve_object_store_bytes(args.object_store_gb)
    ray.init(logging_level="ERROR", include_dashboard=False, object_store_memory=osm_bytes)
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    ctx.execution_options.preserve_order = True

    n_gpus_cluster = int(ray.cluster_resources().get("GPU", 0))
    if n_gpus_cluster == 0:
        P("WARNING: no GPUs visible to Ray; the fused path will fall back to CPU.")

    # Cluster-shape emulation: cap per-GPU VRAM so the auto device batch matches
    # the deploy target, and pin the worker count to the cluster GPU count.
    if args.emulate_cluster:
        os.environ["RAY_DATA_GPU_PREPROC_VRAM_BYTES"] = str(
            int(args.cluster_vram_gb * _GiB)
        )
        num_gpus_list = [args.cluster_gpus]
    elif args.gpu_num_gpus_list:
        num_gpus_list = [int(x) for x in args.gpu_num_gpus_list.split(",") if x]
    else:
        num_gpus_list = [max(1, n_gpus_cluster)]

    batch_specs = [s.strip() for s in args.gpu_batch_sizes.split(",") if s.strip()]
    fit_modes = [s.strip() for s in args.fit_modes.split(",") if s.strip()]

    # Resolve column roles (lean today; 'wide' once that lever lands in criteo).
    try:
        roles = criteo.column_roles_multi(
            days, null_indicator_threshold=args.null_indicator_threshold,
            feature_set=args.feature_set,
        )
    except TypeError:
        if args.feature_set != "lean":
            P(f"NOTE: feature_set={args.feature_set!r} not yet supported by criteo; using lean.")
        roles = criteo.column_roles_multi(
            days, null_indicator_threshold=args.null_indicator_threshold,
        )

    P("=" * 92)
    P("building cached read -> prep -> sort -> indicators input (once) ...")
    P(f"  days={days}  rows_cap={args.rows or 'full'}  feature_set={args.feature_set}  "
      f"cluster GPUs visible={n_gpus_cluster}")
    sorted_ds, input_meta = _build_sorted_input(args, roles)
    P(f"  done: {input_meta['rows']:,} rows, {input_meta['cols']} cols, "
      f"{input_meta['gib']} GiB, {input_meta['blocks']} blocks in {input_meta['build_s']}s")

    profile_dir = os.path.join(here, "data", "_bench_profile")

    cpu_res = None
    if not args.no_cpu:
        P("\nrunning CPU impute+encode+scale subtotal ...")
        cpu_res = _run_cpu_subtotal(
            sorted_ds, roles, warmup=args.cpu_warmup, repeats=args.cpu_repeats
        )
        P(f"  CPU subtotal median: {cpu_res['total_s']:.2f}s")

    gpu_results: List[Dict[str, Any]] = []
    variants = list(itertools.product(num_gpus_list, batch_specs, fit_modes))
    P(f"\nsweeping {len(variants)} GPU variant(s) ...")
    for num_gpus, batch_spec, fit_mode in variants:
        batch_rows = _resolve_batch(batch_spec, sorted_ds, num_gpus)
        P(f"  -> g={num_gpus} batch={batch_rows:,} fit={fit_mode} ...")
        try:
            res = _run_gpu_variant(
                sorted_ds, roles,
                num_gpus=num_gpus, batch_rows=batch_rows, fit_mode=fit_mode,
                profile_dir=profile_dir,
                warmup=args.warmup, repeats=args.repeats,
            )
            P(f"     fit={res['fit_s']:.2f}s transform={res['transform_s']:.2f}s "
              f"total={res['total_s']:.2f}s"
              + (f"  ({cpu_res['total_s']/res['total_s']:.2f}x vs CPU)" if cpu_res else ""))
        except Exception as e:  # keep the sweep going if one config errors
            res = {
                "kind": "gpu", "num_gpus": num_gpus, "batch_rows": batch_rows,
                "fit_mode": fit_mode, "total_s": None,
                "error": f"{type(e).__name__}: {e}".splitlines()[0][:200],
            }
            P(f"     FAILED: {res['error']}")
        gpu_results.append(res)

    if os.path.isdir(profile_dir):
        shutil.rmtree(profile_dir, ignore_errors=True)

    _print_report(input_meta, cpu_res, gpu_results, args)

    # --- parity sanity (sampled fitted stats: every GPU variant vs CPU) ----- #
    parity = None
    if cpu_res and gpu_results and cpu_res.get("fitted_sample"):
        parity = []
        P("\nparity (sampled fitted stats, each GPU variant vs CPU):")
        for v in gpu_results:
            if v.get("total_s") is None:
                continue  # errored variant -- nothing fitted to compare
            res = _parity(cpu_res["fitted_sample"], v.get("fitted_sample"))
            res["variant"] = _label(v)
            parity.append(res)
            P(f"  {'MATCH   ' if res['ok'] else 'MISMATCH'}  {_label(v)}"
              + ("" if res["ok"] else f"   {res['detail']}"))

    out_path = args.out or os.path.join(
        here, "data", f"bench_fused_local_{int(time.time())}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(
            {
                "days": days,
                "rows_cap": args.rows,
                "feature_set": args.feature_set,
                "emulate_cluster": args.emulate_cluster,
                "cluster_gpus": args.cluster_gpus if args.emulate_cluster else None,
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


def _parity(cpu_fit: Dict[str, Any], gpu_fit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare sampled fitted stats; tolerate tiny float drift (GPU vs CPU)."""
    if not gpu_fit:
        return {"ok": False, "detail": "no GPU fitted sample"}
    issues: List[str] = []
    for c, (m, s) in cpu_fit.get("scaler", {}).items():
        gm, gs = (gpu_fit.get("scaler", {}).get(c) or [None, None])
        if not _close(m, gm) or not _close(s, gs):
            issues.append(f"scaler[{c}] cpu=({m},{s}) gpu=({gm},{gs})")
    for c, n in cpu_fit.get("encoder_vocab", {}).items():
        gn = gpu_fit.get("encoder_vocab", {}).get(c)
        if n != gn:
            issues.append(f"vocab[{c}] cpu={n} gpu={gn}")
    for c, m in cpu_fit.get("imputer_mean", {}).items():
        gm = gpu_fit.get("imputer_mean", {}).get(c)
        if not _close(m, gm):
            issues.append(f"imputer_mean[{c}] cpu={m} gpu={gm}")
    return {"ok": not issues, "detail": "ok" if not issues else "; ".join(issues[:4])}


def _close(a, b, rtol=1e-6, atol=1e-6) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    except (TypeError, ValueError):
        return a == b


if __name__ == "__main__":
    main()
