"""GPU preprocessing pipeline for CriteoPrivateAd -- fused, device-resident.

Same dataset, roles, sort key, verification and manifest as ``cpu_pipeline.py``;
the ONLY difference is the feature-transform stage. The CPU baseline runs
``impute -> encode -> scale`` as three separate Ray Data passes (each crossing
PCIe on its own if it were on a GPU). Here those three steps are composed into a
single :class:`~ray.data.preprocessors.Chain` with ``backend="gpu"`` and run as
ONE device-resident pass: each block crosses PCIe once (H2D), is imputed +
encoded + scaled on the resident cuDF frame, then crosses back once (D2H).

    read   read_parquet(selected day_int folders)        (CPU, load)
    prep   labels + prune + list->len                    (CPU map)
    sort   ds.sort([user_id, (day_int,) display_order])  (CPU -- GPU sort is a
                                                           documented follow-up)
    fused  <col>_isnull indicators (CPU) then
           Chain(SimpleImputer(mean), SimpleImputer(most_frequent),
                 OrdinalEncoder, StandardScaler, backend="gpu").fit_transform
                                                          (ONE GPU pass)
    write  write_parquet(training-ready output)

The fused stage is timed as one unit so it compares directly against the CPU
baseline's ``impute + encode + scale`` subtotal. Impute and scale do not win as
standalone GPU ops (the PCIe round trip exceeds the in-RAM CPU work); here they
ride encode's residency. A numeric column that is mean-imputed and then scaled
resolves a null to ``0`` after the z-score with no imputed intermediate
(impute-mean == scaler-mean), folded analytically in the fused fit.

CPU parity / fallback: the user-facing code is identical to the CPU pipeline
(the same preprocessor classes); ``backend="gpu"`` transparently upgrades them
to their device-fusable GPU equivalents when a GPU is present and falls back to
the CPU path otherwise. Output values and fitted stats match the CPU baseline.

Run (local, single day, on a GPU box):
    .venv/bin/python benchmarks/criteo/gpu_pipeline.py \
        --days 1 \
        --data-root /bobbwang/datasets/CriteoPrivateAd/data \
        --out benchmarks/criteo/data/criteo_days1_gpu \
        --ray-address local

Run (AWS GPU Ray cluster, via ray exec; see cluster/ray-gpu.yaml):
    cd /home/ray/benchmarks/criteo && python gpu_pipeline.py \
        --days 1 \
        --data-root s3://bobbwang-ray-e2e-criteo/criteo-private-ad/data \
        --out s3://bobbwang-ray-e2e-criteo/criteo-private-ad/outputs/gpu_days1 \
        --ray-address auto
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import criteo  # noqa: E402
import ray  # noqa: E402

# Reuse the CPU baseline's stage helpers verbatim so the two pipelines stay in
# lockstep (identical read/prep, verification, sanity checks, manifest IO).
import cpu_pipeline as cpu  # noqa: E402

from ray.data.preprocessors import (  # noqa: E402
    Chain,
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
)

P = lambda *a: print(*a, flush=True)  # noqa: E731


def _build_fused_chain(roles) -> Chain:
    """The CPU pipeline's impute+encode+scale, composed for one fused GPU pass.

    Identical preprocessors / order to ``cpu_pipeline`` (numeric mean impute,
    categorical most_frequent impute, ordinal encode, standard scale);
    ``backend="gpu"`` makes the Chain fuse them into a single device-resident
    pass (falling back to CPU when no GPU is available).
    """
    steps: List[Any] = [SimpleImputer(columns=roles.impute_numeric, strategy="mean")]
    if roles.impute_categorical:
        steps.append(
            SimpleImputer(columns=roles.impute_categorical, strategy="most_frequent")
        )
    steps.append(OrdinalEncoder(columns=roles.categorical))
    steps.append(StandardScaler(columns=roles.numeric_features))
    return Chain(*steps, backend="gpu")


def _find_fitted(chain: Chain):
    """Pull the fitted (upgraded GPU) ops back out of the chain for reporting."""
    ops = list(chain.preprocessors)
    num_imputer = next(
        o for o in ops if isinstance(o, SimpleImputer) and o.strategy == "mean"
    )
    cat_imputer = next(
        (o for o in ops if isinstance(o, SimpleImputer) and o.strategy == "most_frequent"),
        None,
    )
    encoder = next(o for o in ops if isinstance(o, OrdinalEncoder))
    scaler = next(o for o in ops if isinstance(o, StandardScaler))
    return num_imputer, cat_imputer, encoder, scaler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None, help="day(s): '1' | '1-5' | '1-30' | 'all'")
    ap.add_argument("--day", type=int, default=None, help="(legacy) single day")
    ap.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="optional read repartition width. Default (recommended): let Ray "
        "size read/sort blocks automatically. Pass only to override.",
    )
    ap.add_argument("--rows", type=int, default=0, help="row cap for a smoke run (0 = full)")
    ap.add_argument("--null-indicator-threshold", type=float, default=0.01)
    ap.add_argument(
        "--max-cardinality",
        type=int,
        default=criteo.DEFAULT_MAX_CARDINALITY,
        help="integer feature columns whose estimated cardinality exceeds this "
        "are hashed (stateless) instead of OrdinalEncoded. Default: 1,000,000.",
    )
    ap.add_argument(
        "--hash-buckets",
        type=int,
        default=criteo.DEFAULT_HASH_BUCKETS,
        help="buckets for hashed high-cardinality columns. Default: 1<<20.",
    )
    ap.add_argument(
        "--card-sample-rows",
        type=int,
        default=criteo.DEFAULT_CARD_SAMPLE_ROWS,
        help="rows to scan for cardinality (integer candidate cols only). "
        "0 (default) = EXACT full distinct count; positive caps it (approx).",
    )
    ap.add_argument(
        "--high-card-cols",
        default=None,
        help="optional comma-separated explicit high-cardinality columns to hash "
        "(skips cardinality sampling).",
    )
    ap.add_argument(
        "--feature-set",
        choices=["lean", "wide"],
        default="lean",
        help="lean (default): inference-realistic recipe. wide: keep the "
        "features_not_available_* bucket as features (wider fused frame, same "
        "steps as the CPU baseline run with --feature-set wide).",
    )
    ap.add_argument("--out", default=None, help="output dir (default: data/criteo_days<lo>_<hi>_gpu)")
    ap.add_argument("--no-write", action="store_true", help="skip writing parquet + manifest")
    ap.add_argument("--object-store-gb", type=float, default=None, help="Ray object-store GiB (local only)")
    ap.add_argument("--data-root", default=None, help="dataset root (local path or s3:// URI)")
    ap.add_argument("--ray-address", default="local", help="'local' or 'auto' (attach to cluster)")
    ap.add_argument("--skip-saved-sort-verify", action="store_true")
    ap.add_argument(
        "--skip-sort",
        action="store_true",
        help="Skip the global sort stage (CPU in both pipelines, so it cancels "
        "in the fusion comparison; its shuffle is the memory bottleneck at "
        "30-day scale). Isolates the fused impute+encode+scale win. Output is "
        "NOT globally sorted; sortedness verification is skipped. Fitted stats / "
        "parity are unaffected (impute/encode/scale are order-independent).",
    )
    ap.add_argument("--overwrite", action="store_true", help="for s3:// --out: delete existing prefix first")
    ap.add_argument(
        "--gpu-batch-size",
        type=int,
        default=None,
        help="per-worker GPU device batch size (rows). Sets "
        "RAY_DATA_GPU_PREPROC_BATCH_SIZE. Default (recommended): auto-derived "
        "from per-GPU VRAM + row width + GPU count (see "
        "_gpu.auto_gpu_block_rows). Pass only to pin an explicit size.",
    )
    ap.add_argument(
        "--gpu-num-gpus",
        type=int,
        default=None,
        help="number of one-GPU fused workers. Sets RAY_DATA_GPU_PREPROC_NUM_GPUS "
        "(default: the cluster's total GPU count).",
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="log the fused transform's per-worker H2D/compute/D2H wall split "
        "(sets RAY_DATA_GPU_PREPROC_PROFILE).",
    )
    args = ap.parse_args()

    # GPU knobs are read by the preprocessors at call time, so setting them here
    # (before fit_transform) is sufficient.
    if args.gpu_batch_size is not None:
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(args.gpu_batch_size)
    if args.gpu_num_gpus is not None:
        os.environ["RAY_DATA_GPU_PREPROC_NUM_GPUS"] = str(args.gpu_num_gpus)
    if args.profile:
        os.environ["RAY_DATA_GPU_PREPROC_PROFILE"] = "1"

    if args.data_root:
        criteo.set_data_root(args.data_root)
    data_root = criteo.DATA_ROOT

    available = criteo.discover_days()
    if args.days is not None:
        days = criteo.parse_days(args.days, available)
    elif args.day is not None:
        days = criteo.parse_days(str(args.day), available)
    else:
        days = [1]
    days_label = f"{days[0]}" if len(days) == 1 else f"{days[0]}-{days[-1]}"

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(
        here, "data", f"criteo_days{days[0]}_{days[-1]}_gpu"
    )

    import logging

    if args.ray_address == "local":
        osm_bytes = cpu._resolve_object_store_bytes(args.object_store_gb)
        ray.init(logging_level="ERROR", include_dashboard=False, object_store_memory=osm_bytes)
        object_store_setting = f"{osm_bytes / 1024 ** 3:.0f} GiB (local, in-RAM /dev/shm)"
    else:
        osm_bytes = None
        # The GPU-preprocessor TUNING knobs (RMM pool, arrow compaction, pool
        # fractions, VRAM override) are read WORKER-side -- inside the fused
        # actors and fit reductions -- so unlike the driver-side knobs (batch
        # size, num gpus, which are resolved here and passed as map_batches
        # kwargs) they must travel to workers on other nodes via runtime_env.
        # Snapshot whatever RAY_DATA_GPU_PREPROC_* vars the launch environment
        # set and forward them to every worker for this job.
        preproc_env = {
            k: v
            for k, v in os.environ.items()
            if k.startswith("RAY_DATA_GPU_PREPROC_")
        }
        if preproc_env:
            P(f"propagating to workers (runtime_env): "
              + ", ".join(sorted(preproc_env)))
        ray.init(
            address=args.ray_address,
            logging_level="ERROR",
            runtime_env={"env_vars": preproc_env} if preproc_env else None,
        )
        object_store_setting = f"cluster-managed (address={args.ray_address})"
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    ctx.execution_options.preserve_order = True

    n_gpus = int(ray.cluster_resources().get("GPU", 0))
    enabled_stages = ["read", "prep"]
    if not args.skip_sort:
        enabled_stages.append("sort")
    enabled_stages.append("fused")
    if not args.no_write:
        enabled_stages.append("write")
    cpu._print_startup(
        days_label=days_label, days=days, data_root=data_root, out_dir=out_dir,
        ray_address=args.ray_address, object_store_setting=object_store_setting,
        rows=args.rows, no_write=args.no_write, enabled_stages=enabled_stages, here=here,
    )
    P(f"backend        : GPU fused Chain (impute+encode+scale); sort=CPU")
    P(f"cluster GPUs   : {n_gpus}   "
      f"preproc_num_gpus={os.environ.get('RAY_DATA_GPU_PREPROC_NUM_GPUS', 'auto')}  "
      f"preproc_batch_size={os.environ.get('RAY_DATA_GPU_PREPROC_BATCH_SIZE', 'default(1<<20)')}")
    if n_gpus == 0:
        P("WARNING: no GPUs visible to Ray; the fused Chain will fall back to CPU.")

    high_card_cols = (
        [c.strip() for c in args.high_card_cols.split(",") if c.strip()]
        if args.high_card_cols
        else None
    )
    roles = criteo.column_roles_multi(
        days, null_indicator_threshold=args.null_indicator_threshold,
        feature_set=args.feature_set, max_cardinality=args.max_cardinality,
        hash_buckets=args.hash_buckets, card_sample_rows=args.card_sample_rows,
        high_card_cols=high_card_cols,
    )
    P("-" * 78)
    P(f"rows (metadata): {roles.total_rows:,}   feature_set={roles.feature_set}   "
      f"null_indicator_threshold={args.null_indicator_threshold}")
    P(f"feature roles: {len(roles.categorical)} categorical, "
      f"{len(roles.hashed)} hashed, "
      f"{len(roles.numeric_features)} numeric, "
      f"{len(roles.indicator_cols)} missing-indicators")
    cpu._print_high_card(roles)
    P(f"sort key: {roles.sort_key}   (GPU-target stage: fused impute+encode+scale)")

    m = cpu.Metrics()
    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)
    hashed = tuple(roles.hashed)
    hash_buckets = roles.hash_buckets
    indicator_cols = tuple(roles.indicator_cols)

    # ---- read ------------------------------------------------------------- #
    with m.stage("read", 0) as rec:
        ds = criteo.read_ray_days(days)
        if args.rows:
            ds = ds.limit(args.rows)
        if args.blocks is not None:
            ds = ds.repartition(args.blocks, shuffle=False)
        ds = ds.materialize()
        n_blocks = ds.num_blocks()
        rows_in, ncols, gib = cpu.ds_stats(ds)
        rec.update(rows=rows_in, out_cols=ncols, gib=gib)

    # ---- prep ------------------------------------------------------------- #
    with m.stage("prep", ncols) as rec:
        ds = ds.map_batches(
            lambda t: cpu.prep_batch(
                t, numeric_raw, categorical, list_features, hashed, hash_buckets
            ),
            batch_format="pyarrow",
            batch_size=None,
        ).materialize()
        _, ncols, gib = cpu.ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- sort (CPU, same as baseline) ------------------------------------- #
    if args.skip_sort:
        P(f"\n[--skip-sort] skipping the global sort over {roles.sort_key}; "
          "output will NOT be globally sorted (fusion-comparison mode)")
    else:
        with m.stage("sort", ncols) as rec:
            ds = ds.sort(roles.sort_key).materialize()
            _, ncols, gib = cpu.ds_stats(ds)
            rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- size the fused GPU device batch ---------------------------------- #
    # Read/sort block sizes are left to Ray (no manual repartition by default);
    # the ONLY thing we size for the device is the fused op's per-batch row count,
    # auto-derived from per-GPU VRAM + this dataset's row width + GPU count
    # (unless the user pinned --gpu-batch-size). Saved-output global order does
    # NOT depend on this (it is guaranteed by the write path's preserve_order
    # handling), so we are free to pick the throughput-optimal device batch.
    if args.gpu_batch_size is None and n_gpus > 0:
        from ray.data.preprocessors import _gpu as _gpumod

        # num_gpus defaults to env_num_gpus() = the actual fused concurrency
        # (RAY_DATA_GPU_PREPROC_NUM_GPUS / --gpu-num-gpus), which can be < the
        # cluster GPU total, so the load-balance cap matches the real fan-out.
        fused_gpus = _gpumod.env_num_gpus()
        auto_rows = _gpumod.auto_gpu_block_rows(ds, num_gpus=fused_gpus)
        os.environ["RAY_DATA_GPU_PREPROC_BATCH_SIZE"] = str(auto_rows)
        P(f"fused device batch     : {auto_rows:,} rows  (auto: VRAM-aware over "
          f"{fused_gpus} fused GPU worker(s); override with --gpu-batch-size)")

    # ---- fused (GPU): indicators (CPU) then one device-resident pass ------ #
    chain = _build_fused_chain(roles)
    with m.stage("fused", ncols) as rec:
        if indicator_cols:
            ds = ds.map_batches(
                lambda t: cpu.add_indicators(t, indicator_cols),
                batch_format="pyarrow",
                batch_size=None,
            )
        ds = chain.fit_transform(ds).materialize()
        _, ncols, gib = cpu.ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)
    num_imputer, cat_imputer, encoder, scaler = _find_fitted(chain)

    # ---- write ------------------------------------------------------------ #
    if not args.no_write:
        with m.stage("write", ncols) as rec:
            cpu._prepare_output_dir(out_dir, overwrite=args.overwrite)
            ds.write_parquet(out_dir)
            rec.update(rows=rows_in, out_cols=ncols, gib=gib)

    # ---- verify saved output is globally sorted --------------------------- #
    if args.skip_sort:
        verify = None
        P("\nskipping sortedness verification (--skip-sort: output is "
          "intentionally not globally sorted)")
    elif args.skip_saved_sort_verify:
        verify = None
        P("\nskipping saved-output sortedness verification (--skip-saved-sort-verify)")
    elif not args.no_write:
        P(f"\nverifying saved-output global sortedness over {roles.sort_key} ...")
        verify = cpu._verify_saved_sorted(out_dir, roles.sort_key, rows_in)
    else:
        verify = cpu._verify_dataset_sorted(ds, roles.sort_key, rows_in)
    if verify is not None:
        P(f"  verification: sorted={verify['globally_sorted']} "
          f"rows_counted={verify['rows_counted']:,}/{verify['expected_rows']:,} "
          f"blocks={verify['n_blocks']}")

    # ---- sanity checks + report ------------------------------------------- #
    checks = cpu._sanity_checks(ds, roles, rows_in, verify, out_dir, args.no_write)
    _print_report(m, roles, checks, verify, scaler, encoder, num_imputer, n_gpus)

    # ---- manifest --------------------------------------------------------- #
    if not args.no_write:
        manifest = _build_manifest(
            args, days, roles, rows_in, m, scaler, encoder, num_imputer,
            cat_imputer, n_blocks, verify, osm_bytes, n_gpus,
        )
        cpu._write_manifest(out_dir, manifest)
        P(f"\nwrote output parquet + manifest.json -> {out_dir}")

    ray.shutdown()


def _print_report(m, roles, checks, verify, scaler, encoder, num_imputer, n_gpus) -> None:
    P("\n" + "-" * 78)
    P("per-stage wall (RAM -> RAM, each stage materialized):")
    P(f"  {'stage':<8} {'secs':>9} {'Mrows/s':>9} {'cols':>11} {'GiB':>7}   note")
    total = 0.0
    for rec in m.stages:
        total += rec["secs"]
        mrows = (rec.get("rows", 0) / rec["secs"] / 1e6) if rec["secs"] else 0.0
        cols = f"{rec['in_cols']}->{rec.get('out_cols', '?')}"
        note = "GPU FUSED (impute+encode+scale)" if rec["name"] == "fused" else ""
        P(f"  {rec['name']:<8} {rec['secs']:>9.2f} {mrows:>9.2f} {cols:>11} "
          f"{rec.get('gib', 0):>7.2f}   {note}")
    P("  " + "-" * 60)
    P(f"  {'TOTAL':<8} {total:>9.2f}")
    fused = next((rec["secs"] for rec in m.stages if rec["name"] == "fused"), 0.0)
    P(f"  {'fused':<8} {fused:>9.2f}   (impute+encode+scale in one device-resident "
      f"pass; compare vs the CPU baseline's impute+encode+scale subtotal)")

    if verify is not None:
        P("\nsaved-output sortedness verification:")
        P(f"  globally_sorted = {verify['globally_sorted']}  "
          f"(in_block_sorted={verify['all_blocks_internally_sorted']}, "
          f"boundaries_ok={verify['adjacent_boundaries_ok']})")
        P(f"  rows_counted = {verify['rows_counted']:,} / expected {verify['expected_rows']:,}  "
          f"(match={verify['rows_match']}), blocks={verify['n_blocks']}")

    P("\nsanity checks:")
    for name, ok in checks.items():
        P(f"  {'PASS' if ok else 'FAIL'}  {name}")

    P("\nfitted-stat summary:")
    vocabs = sorted(
        ((c, cpu._vocab_size(encoder.stats_[f"unique_values({c})"])) for c in roles.categorical),
        key=lambda x: -x[1],
    )
    max_vocab = max((n for _, n in vocabs), default=0)
    P(f"  ordinal-encoder vocab sizes (top 6 of {len(vocabs)}): "
      + ", ".join(f"{c}={n:,}" for c, n in vocabs[:6]))
    P(f"  max ordinal vocab: {max_vocab:,}  (bounded: high-card cols are hashed)")
    if roles.hashed:
        est = roles.estimated_cardinalities
        P(f"  hashed high-card cols ({roles.hash_buckets:,} buckets each, NO dense "
          f"vocab): " + ", ".join(
              f"{c}(est~{est.get(c, 0):,})" for c in roles.hashed))
    P(f"  scaled numeric columns: {len(roles.numeric_features)}")
    P(f"\nbackend: GPU fused Chain over {n_gpus} GPU(s); sort=CPU")


def _build_manifest(args, days, roles, rows_in, m, scaler, encoder, num_imputer,
                    cat_imputer, n_blocks, verify, osm_bytes, n_gpus) -> Dict[str, Any]:
    scaler_stats = {
        c: {"mean": scaler.stats_[f"mean({c})"], "std": scaler.stats_[f"std({c})"]}
        for c in roles.numeric_features
    }
    return {
        "dataset": "CriteoPrivateAd",
        "pipeline": "gpu_fused",
        "backend": "gpu",
        "fused_stage": "impute+encode+scale (one device-resident Chain pass)",
        "sort_backend": "cpu",
        "n_gpus": n_gpus,
        "preproc_num_gpus": os.environ.get("RAY_DATA_GPU_PREPROC_NUM_GPUS"),
        "preproc_batch_size": os.environ.get("RAY_DATA_GPU_PREPROC_BATCH_SIZE"),
        "days": days,
        "days_spec": args.days if args.days is not None else str(args.day),
        "rows": rows_in,
        "rows_metadata_total": roles.total_rows,
        "feature_set": roles.feature_set,
        "num_blocks_read": n_blocks,
        "ray_address": args.ray_address,
        "data_root": criteo.DATA_ROOT,
        "object_store_gib": round(osm_bytes / 1024 ** 3, 1) if osm_bytes else None,
        "sort_key": roles.sort_key,
        "sort_skipped": bool(args.skip_sort),
        "null_indicator_threshold": args.null_indicator_threshold,
        "max_cardinality": roles.max_cardinality,
        "hash_buckets": roles.hash_buckets,
        "card_sample_rows": roles.card_sample_rows,
        "estimated_cardinalities": roles.estimated_cardinalities,
        "columns": {
            "targets": roles.targets,
            "metadata_sort_keys": roles.metadata_keys,
            "categorical": roles.categorical,
            "hashed": roles.hashed,
            "numeric": roles.numeric_features,
            "missing_indicators": [f"{c}_isnull" for c in roles.indicator_cols],
            "list_len": roles.list_len_cols,
        },
        "dropped_counts": {k: len(v) for k, v in roles.dropped.items()},
        "fitted": {
            "scaler": scaler_stats,
            "encoder_vocab_size": {
                c: cpu._vocab_size(encoder.stats_[f"unique_values({c})"])
                for c in roles.categorical
            },
            "imputer_mean": {
                c: num_imputer.stats_[f"mean({c})"] for c in roles.impute_numeric
            },
            "imputer_most_frequent": (
                {c: cat_imputer.stats_[f"most_frequent({c})"] for c in roles.impute_categorical}
                if cat_imputer is not None else {}
            ),
        },
        "stage_timings_s": {rec["name"]: round(rec["secs"], 3) for rec in m.stages},
        "fused_stage_s": round(
            next((rec["secs"] for rec in m.stages if rec["name"] == "fused"), 0.0), 3
        ),
        "verification": verify if verify is not None else {"skipped": True},
    }


if __name__ == "__main__":
    main()
