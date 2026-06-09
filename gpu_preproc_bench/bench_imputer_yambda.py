"""CPU ``SimpleImputer`` vs ``GpuSimpleImputer`` on the real yandex/yambda data.

Inputs (flat/500m, downloaded by ``yambda.py`` from the Hugging Face Hub):

* ``multi_event.parquet`` (~480M rows) -- the main recommender-preprocessing
  benchmark. ``played_ratio_pct`` / ``track_length_seconds`` are NATURALLY null
  for non-listen events (~2.86%); ``item_id`` / ``event_type`` get injected nulls
  so ``most_frequent`` transform has something to fill (the fit/value_counts cost
  is unaffected by injection).
* ``likes.parquet`` (~9M rows, no native nulls) -- a targeted imputer stress with
  injected nulls.

Metrics
-------
* **E2E (Ray, cold-inclusive).** CPU vs GPU ``fit`` / ``transform`` / ``fit_transform``,
  best/median over repeated runs. Each GPU run builds and tears down a fresh
  ``_GpuBatchActor`` pool, so *every* trial includes cold start; fit and transform
  are two separate GPU passes -> two separate cold starts.
* **Operator cold-start floors.** ``fit`` and ``transform`` on a single-block slice
  isolate each pass's actor-pool + first-batch startup cost.
* **Device microbench.** ``cudf.from_arrow -> value_counts/sum_count | fillna -> to_arrow``
  replayed on real batches, each phase in its OWN subprocess so both the fit cold
  start and the transform cold start pay a true CUDA/cuDF init; batches 1..n are
  steady, split into H2D / compute / D2H with bytes/s. This is a DEVICE-PATH
  microbench: it excludes Ray scheduling, the object store, actor lifecycle, and
  block formation, so it is the device floor, not the full operator cost.
* **Batch-size sweep.** Fit size (``RAY_DATA_GPU_PREPROC_BATCH_SIZE``) and transform
  size (``transform(batch_size=)``) swept independently, with rows + estimated
  selected-column bytes; recommended sizes emitted.

Run (16 GPUs):

    RAY_DATA_GPU_PREPROC_NUM_GPUS=16 .venv/bin/python \
        gpu_preproc_bench/bench_imputer_yambda.py --dataset multi_event --rows 50000000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yambda  # noqa: E402

P = lambda *a: print(*a, flush=True)  # noqa: E731

# Columns we inject nulls into, per dataset (mean columns in multi_event use the
# natural nulls and are intentionally absent here).
INJECT = {
    "multi_event": ["item_id", "event_type"],
    "likes": ["item_id", "is_organic", "timestamp"],
}
# (label, columns, strategy, uses_injected_nulls)
CASES = {
    "multi_event": [
        ("mean played_ratio_pct (natural nulls)", ["played_ratio_pct"], "mean"),
        ("mean track_length_seconds (natural nulls)", ["track_length_seconds"], "mean"),
        ("most_frequent item_id (high-card)", ["item_id"], "most_frequent"),
        ("most_frequent event_type (low-card)", ["event_type"], "most_frequent"),
    ],
    "likes": [
        ("most_frequent item_id (high-card)", ["item_id"], "most_frequent"),
        ("most_frequent is_organic (low-card)", ["is_organic"], "most_frequent"),
        ("mean timestamp", ["timestamp"], "mean"),
    ],
}
DEFAULT_BATCH_SIZES = [1 << 20, 2_000_000, 4_000_000, 8_000_000, 16_000_000]


# --------------------------------------------------------------------------- #
# Device microbench (runs as its own subprocess: one true CUDA cold start each)
# --------------------------------------------------------------------------- #
def _micro_main(a) -> None:
    import statistics as st
    import time

    import cudf
    import cupy as cp
    import pyarrow.compute as pc

    sync = cp.cuda.runtime.deviceSynchronize
    cols = a.columns
    batches = yambda.arrow_batches(
        a.dataset,
        cols,
        a.batch_size,
        a.k,
        inject=(cols if a.inject else None),
        frac=a.null_frac,
        seed=0,
    )
    if not batches:
        P("    (no batches)")
        return
    schema = batches[0].schema
    rows0 = batches[0].num_rows
    sel = yambda.selected_bytes(schema, cols, rows0)

    # Fill values are computed on the host and are NOT part of the timed region.
    fill, ffloat = {}, {}
    for c in cols:
        arr = batches[0].column(c)
        if a.strategy == "mean":
            fill[c], ffloat[c] = float(pc.mean(arr).as_py() or 0.0), True
        else:
            nn = arr.drop_null()
            fill[c] = nn[0].as_py() if len(nn) else 0
            ffloat[c] = isinstance(fill[c], float)

    def one(tbl):
        t0 = time.perf_counter()
        gdf = cudf.DataFrame.from_arrow(tbl.select(cols))
        sync()
        t1 = time.perf_counter()
        if a.phase == "fit" and a.strategy == "mean":
            for c in cols:
                float(gdf[c].sum())
                int(gdf[c].count())
            sync()
            t2 = time.perf_counter()
            t3 = t2
        elif a.phase == "fit":
            outs = [gdf[c].value_counts(dropna=True) for c in cols]
            sync()
            t2 = time.perf_counter()
            for vc in outs:
                vc.index.to_arrow()
                vc.to_arrow()
            sync()
            t3 = time.perf_counter()
        else:  # transform
            outs = []
            for c in cols:
                s = gdf[c]
                if ffloat[c] and s.dtype.kind in ("i", "u"):
                    s = s.astype("float64")
                outs.append(s.fillna(fill[c]))
            sync()
            t2 = time.perf_counter()
            for s in outs:
                s.to_arrow()
            sync()
            t3 = time.perf_counter()
        return (t1 - t0, t2 - t1, t3 - t2, t3 - t0)

    times = [one(t) for t in batches]
    cold = times[0]
    steady = times[1:] or times
    med = lambda i: st.median([x[i] for x in steady])  # noqa: E731
    gb = sel / 1e9
    ms = lambda s: s * 1e3  # noqa: E731
    P(
        f"    [{a.phase}/{a.strategy} {cols} bs={a.batch_size:>10,} rows={rows0:>10,} sel={gb:6.3f}GB]"
    )
    P(
        f"      COLD   batch0 : total={ms(cold[3]):8.1f}  H2D={ms(cold[0]):8.1f}  "
        f"compute={ms(cold[1]):8.1f}  D2H={ms(cold[2]):8.1f}  ms"
    )
    h2d_gbs = (gb / med(0)) if med(0) > 0 else 0.0
    P(
        f"      STEADY median : total={ms(med(3)):8.1f}  H2D={ms(med(0)):8.1f}  "
        f"compute={ms(med(1)):8.1f}  D2H={ms(med(2)):8.1f}  ms  | H2D {h2d_gbs:6.1f} GB/s"
    )


def device_microbench(dataset: str, null_frac: float, batch_sizes) -> None:
    if dataset == "multi_event":
        specs = [
            ("fit", "mean", "played_ratio_pct", False),
            ("transform", "mean", "played_ratio_pct", False),
            ("fit", "most_frequent", "item_id", True),
            ("transform", "most_frequent", "item_id", True),
        ]
    else:
        specs = [
            ("fit", "most_frequent", "item_id", True),
            ("transform", "most_frequent", "item_id", True),
            ("fit", "mean", "timestamp", True),
            ("transform", "mean", "timestamp", True),
        ]
    P(
        "\n=== device microbench (each phase = fresh process = true cold start; "
        "device floor, excludes Ray) ==="
    )
    for bs in batch_sizes:
        P(f"\n  -- batch_size = {bs:,} --")
        for phase, strat, col, inj in specs:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "_micro",
                "--dataset",
                dataset,
                "--phase",
                phase,
                "--strategy",
                strat,
                "--columns",
                col,
                "--batch-size",
                str(bs),
                "--k",
                "6",
                "--null-frac",
                str(null_frac),
            ]
            if inj:
                cmd.append("--inject")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
            subprocess.run(cmd, env=env, check=False)


# --------------------------------------------------------------------------- #
# E2E + correctness + sweep (orchestrator process; imports Ray)
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", choices=["multi_event", "likes"], default="multi_event"
    )
    ap.add_argument(
        "--rows",
        type=int,
        default=50_000_000,
        help="row cap for the fair CPU-vs-GPU comparison (0 = full).",
    )
    ap.add_argument("--null-frac", type=float, default=0.05)
    ap.add_argument(
        "--gpus",
        type=int,
        default=int(os.environ.get("RAY_DATA_GPU_PREPROC_NUM_GPUS", "16")),
    )
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--fit-batch-size", type=int, default=8_000_000)
    ap.add_argument("--transform-batch-size", type=int, default=8_000_000)
    ap.add_argument(
        "--sweep-batch-sizes", default=",".join(str(b) for b in DEFAULT_BATCH_SIZES)
    )
    ap.add_argument("--micro-batch-sizes", default="2000000,8000000")
    ap.add_argument("--no-cpu", action="store_true", help="skip CPU runs (GPU-only).")
    ap.add_argument("--no-micro", action="store_true")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--no-coldstart", action="store_true")
    ap.add_argument(
        "--no-string-id",
        action="store_true",
        help="skip the high-cardinality string headline (item_id cast to string).",
    )
    ap.add_argument(
        "--gpu-full-fit",
        action="store_true",
        help="also time GPU fit of most_frequent(item_id) on the full dataset.",
    )
    args = ap.parse_args()

    os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(args.gpus)
    sweep_sizes = [int(x) for x in args.sweep_batch_sizes.split(",") if x]
    micro_sizes = [int(x) for x in args.micro_batch_sizes.split(",") if x]

    import ray  # noqa: E402
    from common import best_of  # noqa: E402

    from ray.data.preprocessors import GpuSimpleImputer, SimpleImputer  # noqa: E402

    P("=" * 78)
    P(
        f"yandex/yambda flat/500m  dataset={args.dataset}  gpus={args.gpus}  "
        f"rows={'full' if args.rows == 0 else f'{args.rows:,}'}  trials={args.trials}"
    )
    P("=" * 78)
    yambda.print_status(args.dataset)

    ray.init(num_gpus=args.gpus, include_dashboard=False, logging_level="ERROR")

    # ---- load + inject (materialize ONCE, reuse for CPU + GPU) -------------- #
    ds = yambda.read_ray(args.dataset)
    if args.rows:
        ds = ds.limit(args.rows)
    ds = yambda.inject_nulls(
        ds, INJECT[args.dataset], frac=args.null_frac, seed=0
    ).materialize()
    n = ds.count()
    P(
        f"\nmaterialized {n:,} rows; injected ~{args.null_frac:.0%} nulls into "
        f"{INJECT[args.dataset]} (mean columns use natural nulls)"
    )

    pq_schema = yambda.pq.read_schema(yambda.ensure(args.dataset))

    # ---- correctness (tie-aware), on a small sample ------------------------ #
    P("\n=== correctness vs CPU SimpleImputer (200k sample) ===")
    sample = yambda.sample_with_row_id(ds, 200_000)
    spdf = sample.to_pandas()
    all_ok = True
    for label, cols, strategy in CASES[args.dataset]:
        col = cols[0]
        cpu_imp = SimpleImputer(columns=cols, strategy=strategy).fit(sample)
        gpu_imp = GpuSimpleImputer(columns=cols, strategy=strategy).fit(sample)
        if strategy == "mean":
            cv = cpu_imp.stats_[f"mean({col})"]
            gv = gpu_imp.stats_[f"mean({col})"]
            ok = (cv is None and gv is None) or abs(cv - gv) <= 1e-6 * max(1.0, abs(cv))
            note = f"cpu={cv:.6g} gpu={gv:.6g}"
        else:
            cv = cpu_imp.stats_[f"most_frequent({col})"]
            gv = gpu_imp.stats_[f"most_frequent({col})"]
            vc = spdf[col].value_counts(dropna=True)
            cands = sorted(vc[vc == vc.max()].index.tolist()) if len(vc) else []
            if len(cands) <= 1:
                ok = (cv == gv) and (not cands or gv == cands[0])
                note = f"cpu={cv} gpu={gv} (no tie)"
            else:
                ok = gv == cands[0]  # GPU must pick the smallest top-count value
                note = f"TIE x{len(cands)}; GPU smallest={cands[0]} gpu={gv} (CPU={cv} may differ)"
        all_ok &= ok
        P(f"  [{'PASS' if ok else 'FAIL'}] {label}: {note}")
    P(f"  overall: {'PASS' if all_ok else 'FAIL'}")

    # ---- E2E: fit / transform / fit_transform, cold-inclusive -------------- #
    def e2e(label, cols, strategy):
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(args.fit_batch_size)
        mf = strategy == "most_frequent"
        P(f"\n--- {label} ---")

        gfit = best_of(
            lambda: GpuSimpleImputer(columns=cols, strategy=strategy).fit(ds),
            trials=args.trials,
            warmup=1,
        )[0]
        g = GpuSimpleImputer(columns=cols, strategy=strategy)
        g.fit(ds)
        gtf = best_of(
            lambda: g.transform(ds, batch_size=args.transform_batch_size).materialize(),
            trials=args.trials,
            warmup=1,
        )[0]
        gft = best_of(
            lambda: GpuSimpleImputer(columns=cols, strategy=strategy)
            .fit_transform(ds, transform_batch_size=args.transform_batch_size)
            .materialize(),
            trials=args.trials,
            warmup=1,
        )[0]

        if args.no_cpu:
            P(
                f"  GPU  fit={gfit:7.2f}s  transform={gtf:7.2f}s  fit_transform={gft:7.2f}s"
            )
            return
        cfit = best_of(
            lambda: SimpleImputer(columns=cols, strategy=strategy).fit(ds),
            trials=args.trials,
            warmup=0 if mf else 1,
        )[0]
        c = SimpleImputer(columns=cols, strategy=strategy)
        c.fit(ds)
        ctf = best_of(
            lambda: c.transform(ds, batch_size=args.transform_batch_size).materialize(),
            trials=args.trials,
            warmup=1,
        )[0]
        cft = best_of(
            lambda: SimpleImputer(columns=cols, strategy=strategy)
            .fit_transform(ds)
            .materialize(),
            trials=args.trials,
            warmup=0 if mf else 1,
        )[0]
        sp = lambda c_, g_: (c_ / g_) if g_ > 0 else 0.0  # noqa: E731
        P(f"  CPU  fit={cfit:8.2f}s  transform={ctf:7.2f}s  fit_transform={cft:8.2f}s")
        P(f"  GPU  fit={gfit:8.2f}s  transform={gtf:7.2f}s  fit_transform={gft:8.2f}s")
        P(
            f"  x    fit={sp(cfit,gfit):7.2f}  transform={sp(ctf,gtf):7.2f}  "
            f"fit_transform={sp(cft,gft):7.2f}"
        )

    if not args.no_e2e:
        P(
            "\n=== E2E CPU vs GPU (cold-inclusive: each run spins up + tears down GPU actors) ==="
        )
        for label, cols, strategy in CASES[args.dataset]:
            e2e(label, cols, strategy)

    # ---- operator-level cold-start floors (fit AND transform) -------------- #
    if not args.no_coldstart:
        P("\n=== operator cold-start floors (single 1M-row block; best of 3) ===")
        tiny = ds.limit(1_000_000).materialize()
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = "1000000"
        fcold = best_of(
            lambda: GpuSimpleImputer(columns=["item_id"], strategy="most_frequent").fit(
                tiny
            ),
            trials=3,
            warmup=0,
        )[0]
        gt = GpuSimpleImputer(columns=["item_id"], strategy="most_frequent")
        gt.fit(tiny)
        tcold = best_of(
            lambda: gt.transform(tiny, batch_size=1_000_000).materialize(),
            trials=3,
            warmup=0,
        )[0]
        P(
            f"  fit       cold-start floor: {fcold:6.2f}s  (actor pool + value_counts pass)"
        )
        P(f"  transform cold-start floor: {tcold:6.2f}s  (actor pool + fillna pass)")

    # ---- batch-size sweep (fit and transform, separately) ------------------ #
    if not args.no_sweep:
        col = "item_id"
        P(f"\n=== batch-size sweep on most_frequent({col}) ===")
        P("  fit-only (RAY_DATA_GPU_PREPROC_BATCH_SIZE):")
        best_fit = None
        for bs in sweep_sizes:
            os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(bs)
            t = best_of(
                lambda: GpuSimpleImputer(columns=[col], strategy="most_frequent").fit(
                    ds
                ),
                trials=args.trials,
                warmup=1,
            )[0]
            P(f"    fit_bs={bs:>10,}  {t:7.2f}s")
            if best_fit is None or t < best_fit[1]:
                best_fit = (bs, t)
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(args.fit_batch_size)
        gsw = GpuSimpleImputer(columns=[col], strategy="most_frequent")
        gsw.fit(ds)
        P("  transform-only (transform batch_size):")
        sel_total = yambda.selected_bytes(pq_schema, [col], n)
        best_tf = None
        for bs in sweep_sizes:
            t = best_of(
                lambda: gsw.transform(ds, batch_size=bs).materialize(),
                trials=args.trials,
                warmup=1,
            )[0]
            gbs = (sel_total / 1e9) / t if t > 0 else 0.0
            P(
                f"    tf_bs={bs:>10,}  {t:7.2f}s  ({n/t/1e6:6.1f} Mrows/s, ~{gbs:5.2f} GB/s eff)"
            )
            if best_tf is None or t < best_tf[1]:
                best_tf = (bs, t)
        P(
            f"  recommended: fit_batch_size={best_fit[0]:,}  transform_batch_size={best_tf[0]:,}"
        )

    if args.gpu_full_fit and args.rows:
        P("\n=== GPU-only fit on FULL dataset (most_frequent item_id) ===")
        full = yambda.inject_nulls(
            yambda.read_ray(args.dataset),
            INJECT[args.dataset],
            frac=args.null_frac,
            seed=0,
        ).materialize()
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(args.fit_batch_size)
        t = best_of(
            lambda: GpuSimpleImputer(columns=["item_id"], strategy="most_frequent").fit(
                full
            ),
            trials=args.trials,
            warmup=1,
        )[0]
        P(f"  full {full.count():,} rows: GPU fit = {t:.2f}s")

    # ---- high-cardinality STRING headline (the GPU-favorable regime) ------- #
    # yambda ids are uint32 (CPU value_counts is vectorized and fast, so the GPU
    # is transfer-bound on them). Casting item_id to string exercises the regime
    # the operator targets: the CPU most_frequent path counts/merges Python
    # objects with `Counter`, which is slow, while cuDF counts strings on device.
    if not args.no_string_id:
        P("\n=== high-cardinality STRING most_frequent (item_id cast to string) ===")
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(args.fit_batch_size)
        sbase = yambda.read_ray(args.dataset)
        if args.rows:
            sbase = sbase.limit(args.rows)
        sbase = yambda.inject_nulls(sbase, ["item_id"], frac=args.null_frac, seed=0)

        def _to_str(b):
            import pyarrow as pa
            import pyarrow.compute as pc

            return pa.table({"item_id_str": pc.cast(b.column("item_id"), pa.string())})

        sds = sbase.map_batches(
            _to_str, batch_format="pyarrow", zero_copy_batch=True
        ).materialize()
        cols = ["item_id_str"]
        P(f"  rows={sds.count():,}  (distinct ids are high-cardinality strings)")
        gfit = best_of(
            lambda: GpuSimpleImputer(columns=cols, strategy="most_frequent").fit(sds),
            trials=args.trials,
            warmup=1,
        )[0]
        gft = best_of(
            lambda: GpuSimpleImputer(columns=cols, strategy="most_frequent")
            .fit_transform(sds, transform_batch_size=args.transform_batch_size)
            .materialize(),
            trials=args.trials,
            warmup=1,
        )[0]
        if args.no_cpu:
            P(f"  GPU  fit={gfit:8.2f}s  fit_transform={gft:8.2f}s")
        else:
            cfit = best_of(
                lambda: SimpleImputer(columns=cols, strategy="most_frequent").fit(sds),
                trials=args.trials,
                warmup=0,
            )[0]
            cft = best_of(
                lambda: SimpleImputer(columns=cols, strategy="most_frequent")
                .fit_transform(sds)
                .materialize(),
                trials=args.trials,
                warmup=0,
            )[0]
            sp = lambda c_, g_: (c_ / g_) if g_ > 0 else 0.0  # noqa: E731
            P(f"  CPU  fit={cfit:8.2f}s  fit_transform={cft:8.2f}s")
            P(f"  GPU  fit={gfit:8.2f}s  fit_transform={gft:8.2f}s")
            P(f"  x    fit={sp(cfit,gfit):7.2f}  fit_transform={sp(cft,gft):7.2f}")

    ray.shutdown()

    # ---- device microbench (separate processes; after Ray shutdown) -------- #
    if not args.no_micro:
        device_microbench(args.dataset, args.null_frac, micro_sizes)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_micro":
        mp = argparse.ArgumentParser()
        mp.add_argument("_micro")
        mp.add_argument("--dataset", required=True)
        mp.add_argument("--phase", choices=["fit", "transform"], required=True)
        mp.add_argument("--strategy", choices=["mean", "most_frequent"], required=True)
        mp.add_argument("--columns", nargs="+", required=True)
        mp.add_argument("--batch-size", type=int, required=True)
        mp.add_argument("--k", type=int, default=6)
        mp.add_argument("--null-frac", type=float, default=0.05)
        mp.add_argument("--inject", action="store_true")
        _micro_main(mp.parse_args())
    else:
        main()
