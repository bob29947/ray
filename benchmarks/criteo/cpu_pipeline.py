"""CPU-only preprocessing baseline for CriteoPrivateAd (1 day .. all 30 days).

Flow (each stage materialized RAM -> RAM and timed independently):

    read   read_parquet(selected day_int folders)                  (load)
    prep   derive labels + drop leakage/not-available + list->len   (CPU map)
    sort   ds.sort([user_id, (day_int,) display_order])            (GPU TARGET)
    impute add <col>_isnull indicators + SimpleImputer mean/mode    (CPU, not GPU)
    encode OrdinalEncoder(categoricals)                             (GPU TARGET)
    scale  StandardScaler(numerics)                                 (GPU TARGET)
    write  write_parquet(training-ready output)                     (ONLY saved dataset)

Stage order deliberately puts the order-setting ``sort`` BEFORE impute/encode/
scale (all row-order-independent): prep drops the 80 ``features_not_available_*``,
the 3 delay arrays, ``id``, all-null and raw list columns first, so the sort
moves the minimal pruned row -- a clean future GPU-sort comparison. The missing
indicators and imputed/encoded/scaled columns are added AFTER the sort. No
sorted-only intermediate is written: the only saved dataset is the final
processed parquet (after scale) plus ``manifest.json``.

Multi-day: all selected ``day_int=<d>`` folders are read into one Ray Dataset.
``day_int`` (a Hive partition column, exposed by Ray as a string) is cast to int
and kept, with ``user_id`` and ``display_order``, as a raw metadata / sort-key
column -- never encoded or scaled. The global sort key is ``[user_id,
display_order]`` for a single day and ``[user_id, day_int, display_order]`` for
multi-day runs.

The saved parquet's global sortedness is verified with a scalable pass (re-read
the output, keep block order, check each block is internally nondecreasing and
that adjacent blocks do not overlap) -- never by sorting 100M keys in pandas.

ML choices are documented in the decision log printed at the end and written to
``manifest.json`` next to the output (so a later GPU pipeline transforms exactly
the same columns for parity). sort / encode / scale are the GPU-acceleration
targets; impute stays on CPU on purpose (host-staged GPU imputation is not worth
it for this dataset).

The SAME script runs locally (start a local Ray) and on an AWS CPU Ray cluster
(attach with ``--ray-address auto``, read/write ``s3://`` via ``--data-root`` /
``--out``). See ``benchmarks/criteo/README.md`` and ``cluster/ray-cpu.yaml``.

Run (local, single day):
    .venv/bin/python benchmarks/criteo/cpu_pipeline.py \
        --days 1 \
        --data-root /bobbwang/datasets/CriteoPrivateAd/data \
        --out benchmarks/criteo/data/criteo_days1_cpu_baseline \
        --ray-address local

Run (local, multi-day / all 30):
    .venv/bin/python benchmarks/criteo/cpu_pipeline.py --days 1-30 \
        --out benchmarks/criteo/data/criteo_days1_30_cpu_baseline

Run (AWS CPU Ray cluster, via ray exec):
    cd /home/ray/benchmarks/criteo && python cpu_pipeline.py \
        --days 1 \
        --data-root s3://TODO_REQUIRED/criteo-private-ad/data \
        --out s3://TODO_REQUIRED/criteo-private-ad/outputs/cpu_baseline_days1 \
        --ray-address auto
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

import criteo  # noqa: E402
import ray  # noqa: E402

from ray.data.preprocessors import (  # noqa: E402
    OrdinalEncoder,
    SimpleImputer,
    StandardScaler,
)

P = lambda *a: print(*a, flush=True)  # noqa: E731
GPU_TARGET_STAGES = {"sort", "encode", "scale"}


# --------------------------------------------------------------------------- #
# CPU map stages (pyarrow, so block schemas stay identical)
# --------------------------------------------------------------------------- #
def prep_batch(
    table: "pa.Table",
    numeric_raw: Tuple[str, ...],
    categorical: Tuple[str, ...],
    list_features: Tuple[str, ...],
) -> "pa.Table":
    """Label engineering + column pruning. Output keeps only the columns the
    sort and the downstream feature transforms need (drops everything else)."""
    cols: Dict[str, Any] = {}

    # Binary labels -> int8. is_click_landed is stored as 0.0/1.0 double.
    cols["is_clicked"] = pc.cast(table.column("is_clicked"), pa.int8())
    cols["is_visit"] = pc.cast(table.column("is_visit"), pa.int8())
    cols["is_click_landed"] = pc.cast(
        pc.coalesce(table.column("is_click_landed"), pa.scalar(0.0)), pa.int8()
    )

    # Sales targets (target derivation, NOT feature imputation):
    #   sales_count = coalesce(nb_sales, 0);  is_sale = nb_sales > 0
    nb = table.column(criteo.SALES_RAW)
    sales_count = pc.coalesce(nb, pa.scalar(0, nb.type))
    cols["sales_count"] = pc.cast(sales_count, pa.int64())
    cols["is_sale"] = pc.cast(pc.greater(sales_count, pa.scalar(0, nb.type)), pa.int8())

    # Metadata / sort-key columns -- kept RAW through the whole pipeline into the
    # saved output (never encoded or scaled). day_int is the Hive partition
    # column Ray exposes as a string; cast to int64 so it sorts numerically
    # (otherwise "10" < "2" lexicographically would break the day ordering).
    cols[criteo.SORT_USER] = table.column(criteo.SORT_USER)
    cols[criteo.SORT_DAY] = pc.cast(table.column(criteo.SORT_DAY), pa.int64())
    cols[criteo.SORT_DISPLAY] = table.column(criteo.SORT_DISPLAY)

    for c in categorical:
        cols[c] = table.column(c)
    for c in numeric_raw:
        cols[c] = table.column(c)

    # List feature -> length (null/empty list -> 0). v1 baseline keeps the count
    # signal and drops the raw variable-length list (no MultiHot/Hashing yet).
    for c in list_features:
        ln = pc.list_value_length(table.column(c))
        cols[f"{c}_len"] = pc.cast(pc.coalesce(ln, pa.scalar(0, pa.int32())), pa.int32())

    return pa.table(cols)


def add_indicators(table: "pa.Table", indicator_cols: Tuple[str, ...]) -> "pa.Table":
    """Append a 0/1 ``<col>_isnull`` missing-indicator per high-null numeric col.

    Computed from the still-null column BEFORE imputation, so it captures the
    (informative) absence signal. Kept as int8 0/1 and never scaled."""
    out = table
    for c in indicator_cols:
        isnull = pc.cast(pc.is_null(table.column(c)), pa.int8())
        out = out.append_column(f"{c}_isnull", isnull)
    return out


# --------------------------------------------------------------------------- #
# Timing / metrics
# --------------------------------------------------------------------------- #
class Metrics:
    def __init__(self) -> None:
        self.stages: List[Dict[str, Any]] = []

    @contextmanager
    def stage(self, name: str, n_in_cols: int):
        t = time.perf_counter()
        rec: Dict[str, Any] = {"name": name, "in_cols": n_in_cols}
        yield rec
        rec["secs"] = time.perf_counter() - t
        self.stages.append(rec)


def ds_stats(ds) -> Tuple[int, int, float]:
    """rows, n_cols, GiB of a materialized dataset (all metadata-cheap)."""
    rows = ds.count()
    ncols = len(ds.schema().names)
    gib = (ds.size_bytes() or 0) / (1024 ** 3)
    return rows, ncols, gib


def _jsonable(v: Any) -> Any:
    try:
        import numpy as np

        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
    except Exception:
        pass
    return v


def _vocab_size(stat: Any) -> int:
    if isinstance(stat, dict):
        return len(stat)
    keys, _ = stat  # arrow (keys_array, values_array)
    return len(keys)


def _resolve_object_store_bytes(object_store_gb: Any) -> int:
    """Size the Ray object store so the whole pipeline stays in RAM.

    The full 100M-row run materializes ~165 GiB (read) and the CPU sort shuffles
    the pruned rows; with the default (~200 GiB capped) object store this spills
    to local disk, and here ``/tmp`` only has ~120 GB free -> OutOfDiskError. The
    box has ~1.5 TiB RAM and a ~756 GiB ``/dev/shm`` (the plasma backing store),
    so by default we claim ~85% of /dev/shm. That comfortably holds read + prep +
    the sort's map/reduce intermediates with no spill. Stay <= /dev/shm so Ray
    keeps plasma in RAM instead of falling back to the (small) disk."""
    if object_store_gb is not None:
        return int(object_store_gb * 1024 ** 3)
    try:
        st = os.statvfs("/dev/shm")
        shm_bytes = st.f_blocks * st.f_frsize
        return int(shm_bytes * 0.85)
    except Exception:
        return 200 * 1024 ** 3


def _git_commit(here: str) -> str:
    """Short git commit of the worktree, or 'unknown' (e.g. on a mounted copy
    on the cluster, which is not a git repo)."""
    import subprocess

    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=here,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _print_startup(
    *, days_label, days, data_root, out_dir, ray_address,
    object_store_setting, rows, no_write, enabled_stages, here,
) -> None:
    """One banner that makes the local-vs-cloud run fully self-describing."""
    P("=" * 78)
    P("benchmark      : CriteoPrivateAd CPU preprocessing E2E (cpu_pipeline.py)")
    P(f"git commit     : {_git_commit(here)}")
    P(f"ray version    : {ray.__version__}")
    P(f"ray file       : {ray.__file__}")
    P(f"ray_address    : {ray_address}")
    P(f"data_root      : {data_root}")
    P(f"out            : {'(--no-write: not saved)' if no_write else out_dir}")
    P(f"days           : {days_label}  ({len(days)} day_int partition(s))")
    P(f"row limit      : {rows if rows else 'none (full)'}")
    P(f"object store   : {object_store_setting}")
    P(f"stages         : {', '.join(enabled_stages)}  (+ TOTAL)")
    P("=" * 78)


def _prepare_output_dir(out_dir: str, *, overwrite: bool) -> None:
    """Make ``out_dir`` ready for ``ds.write_parquet`` -- local OR ``s3://``.

    Local: remove any existing dir and recreate (preserves current behavior).
    S3: never use os.path / shutil / glob / makedirs; if the prefix already has
    objects, fail clearly unless ``overwrite`` (then delete the prefix contents
    first). ``ds.write_parquet`` creates the prefix, so no mkdir is needed on S3.
    """
    if criteo.is_s3(out_dir):
        import pyarrow.fs as pafs

        fs, path = pafs.FileSystem.from_uri(out_dir)
        existing = fs.get_file_info(
            pafs.FileSelector(path, recursive=False, allow_not_found=True)
        )
        if existing:
            if not overwrite:
                raise RuntimeError(
                    f"S3 output already exists: {out_dir} ({len(existing)} "
                    f"entries). Refusing to overwrite; pass --overwrite or "
                    f"choose a new --out."
                )
            fs.delete_dir_contents(path, missing_dir_ok=True)
        return
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def _write_manifest(out_dir: str, manifest: Dict[str, Any]) -> None:
    """Write ``manifest.json`` next to the output -- local OR ``s3://`` (via the
    pyarrow filesystem, never the builtin ``open`` for S3)."""
    blob = json.dumps(manifest, indent=2, default=_jsonable).encode("utf-8")
    if criteo.is_s3(out_dir):
        import pyarrow.fs as pafs

        fs, path = pafs.FileSystem.from_uri(out_dir)
        with fs.open_output_stream(f"{path}/manifest.json") as fh:
            fh.write(blob)
    else:
        with open(os.path.join(out_dir, "manifest.json"), "wb") as fh:
            fh.write(blob)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--days",
        default=None,
        help="day(s) to process: '1' | '1-5' | '1-25' | '1-30' | 'all'. "
        "Multi-day reads all selected day_int folders into one Ray Dataset.",
    )
    ap.add_argument(
        "--day", type=int, default=None,
        help="(legacy) single day; prefer --days. Ignored if --days is given.",
    )
    ap.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="optional repartition width. Default (recommended): let Ray choose "
        "the read block count (no extra repartition pass). Pass only to override.",
    )
    ap.add_argument("--rows", type=int, default=0, help="row cap for a quick smoke run (0 = full)")
    ap.add_argument("--null-indicator-threshold", type=float, default=0.01)
    ap.add_argument("--out", default=None, help="output dir (default: data/criteo_days<lo>_<hi>_cpu_baseline)")
    ap.add_argument("--no-write", action="store_true", help="skip writing parquet + manifest (RAM->RAM only)")
    ap.add_argument(
        "--object-store-gb",
        type=float,
        default=None,
        help="Ray object-store size in GiB (LOCAL ray-address only). Default: "
        "~85%% of /dev/shm so the whole 100M-row pipeline (incl. the sort "
        "shuffle) stays in RAM and never spills to the small local disk. "
        "Ignored when --ray-address attaches to a cluster.",
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="Dataset root. Local example: /bobbwang/datasets/CriteoPrivateAd/"
        "data ; cloud example: s3://.../CriteoPrivateAd/data. Default: keep "
        "criteo.DATA_ROOT (the local DGX path).",
    )
    ap.add_argument(
        "--ray-address",
        default="local",
        help="'local' -> start a local Ray as today (preserves current "
        "behavior incl. --object-store-gb). 'auto' (or any address) -> attach "
        "to an existing cluster via ray.init(address=...). Default: local.",
    )
    ap.add_argument(
        "--skip-saved-sort-verify",
        action="store_true",
        help="Skip the saved-output global-sortedness verification (escape "
        "hatch for first S3 smoke runs). The write still happens unless "
        "--no-write is also passed.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="For an s3:// --out: delete the existing output prefix before "
        "writing. Local always overwrites; without this an existing S3 prefix "
        "is an error (never a silent overwrite).",
    )
    args = ap.parse_args()

    # Point the loader at the requested dataset root (local path or s3:// URI)
    # BEFORE any discovery. With no --data-root we keep criteo.DATA_ROOT (local),
    # preserving current behavior.
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
        here, "data", f"criteo_days{days[0]}_{days[-1]}_cpu_baseline"
    )

    import logging

    # Ray init: --ray-address local starts a local Ray (current behavior, incl.
    # the in-RAM object-store sizing). Anything else (e.g. "auto") ATTACHES to an
    # already-running cluster and lets the cluster own cpus/gpus/object-store --
    # so we deliberately do NOT pass num_cpus/num_gpus/object_store_memory/
    # include_dashboard in that path.
    if args.ray_address == "local":
        osm_bytes = _resolve_object_store_bytes(args.object_store_gb)
        ray.init(
            logging_level="ERROR",
            include_dashboard=False,
            object_store_memory=osm_bytes,
        )
        object_store_setting = (
            f"{osm_bytes / 1024 ** 3:.0f} GiB (local, in-RAM /dev/shm)"
        )
    else:
        osm_bytes = None
        ray.init(address=args.ray_address, logging_level="ERROR")
        object_store_setting = f"cluster-managed (address={args.ray_address})"
    logging.getLogger("ray.data").setLevel(logging.ERROR)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    # Keep the globally-sorted order through impute/encode/scale/write AND through
    # the read-back verification. Ray Data map operators do NOT preserve block
    # order by default.
    ctx.execution_options.preserve_order = True

    enabled_stages = ["read", "prep", "sort", "impute", "encode", "scale"]
    if not args.no_write:
        enabled_stages.append("write")
    _print_startup(
        days_label=days_label,
        days=days,
        data_root=data_root,
        out_dir=out_dir,
        ray_address=args.ray_address,
        object_store_setting=object_store_setting,
        rows=args.rows,
        no_write=args.no_write,
        enabled_stages=enabled_stages,
        here=here,
    )

    roles = criteo.column_roles_multi(
        days, null_indicator_threshold=args.null_indicator_threshold
    )

    P("-" * 78)
    blocks_label = "auto" if args.blocks is None else str(args.blocks)
    P(f"rows (metadata): {roles.total_rows:,}   blocks={blocks_label}   "
      f"null_indicator_threshold={args.null_indicator_threshold}")
    P(f"feature roles: {len(roles.categorical)} categorical, "
      f"{len(roles.numeric_features)} numeric "
      f"({len(roles.numeric_raw)} raw + {len(roles.list_len_cols)} list-len), "
      f"{len(roles.indicator_cols)} missing-indicators")
    P(f"metadata/sort keys (kept raw, not features): {roles.metadata_keys}")
    P(f"dropped: id, {len(roles.dropped['delay_arrays_leakage'])} delay arrays, "
      f"{len(roles.dropped['not_available_at_inference'])} features_not_available_*, "
      f"{len(roles.dropped['all_null'])} all-null {roles.dropped['all_null']}")
    P(f"sort key: {roles.sort_key}   (GPU-target stages: sort, encode, scale)")

    m = Metrics()
    numeric_raw = tuple(roles.numeric_raw)
    categorical = tuple(roles.categorical)
    list_features = tuple(roles.list_features)
    indicator_cols = tuple(roles.indicator_cols)

    # ---- read ------------------------------------------------------------- #
    # Read all selected day_int folders into ONE dataset. By default we let Ray
    # choose the read block count (no extra pass); only repartition if --blocks
    # is given. Any repartition cost stays inside this read stage.
    with m.stage("read", 0) as rec:
        ds = criteo.read_ray_days(days)
        if args.rows:
            ds = ds.limit(args.rows)
        if args.blocks is not None:
            ds = ds.repartition(args.blocks, shuffle=False)
        ds = ds.materialize()
        n_blocks = ds.num_blocks()
        rows_in, ncols, gib = ds_stats(ds)
        rec.update(rows=rows_in, out_cols=ncols, gib=gib)
    if criteo.SORT_DAY not in ds.schema().names:
        raise RuntimeError(
            f"{criteo.SORT_DAY!r} missing after read; multi-day tagging failed."
        )
    P(f"read produced {n_blocks} blocks "
      f"({'Ray default' if args.blocks is None else f'repartitioned to {args.blocks}'})")

    # ---- prep (label engineering + prune; minimizes the sort payload) ------ #
    with m.stage("prep", ncols) as rec:
        ds = ds.map_batches(
            lambda t: prep_batch(t, numeric_raw, categorical, list_features),
            batch_format="pyarrow",
            batch_size=None,
        ).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- sort (GPU TARGET): sort the pruned raw rows ---------------------- #
    # user_id / day_int / display_order are kept through every later stage and
    # into the saved output as metadata / sort keys; they are NOT dropped here
    # and NOT transformed. No sorted-only intermediate is written -- the sorted
    # rows flow straight into impute/encode/scale and are saved once after scale.
    with m.stage("sort", ncols) as rec:
        # This branch's Dataset.sort has no ``backend=`` arg; CPU is the default.
        ds = ds.sort(roles.sort_key).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- impute (CPU, not GPU): indicators then mean/most_frequent -------- #
    with m.stage("impute", ncols) as rec:
        if indicator_cols:
            ds = ds.map_batches(
                lambda t: add_indicators(t, indicator_cols),
                batch_format="pyarrow",
                batch_size=None,
            )
        num_imputer = SimpleImputer(columns=roles.impute_numeric, strategy="mean")
        ds = num_imputer.fit_transform(ds)
        cat_imputer = None
        if roles.impute_categorical:
            cat_imputer = SimpleImputer(
                columns=roles.impute_categorical, strategy="most_frequent"
            )
            ds = cat_imputer.fit_transform(ds)
        ds = ds.materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- encode (GPU TARGET): OrdinalEncoder ------------------------------ #
    with m.stage("encode", ncols) as rec:
        encoder = OrdinalEncoder(columns=roles.categorical)
        ds = encoder.fit_transform(ds).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- scale (GPU TARGET): StandardScaler ------------------------------- #
    with m.stage("scale", ncols) as rec:
        scaler = StandardScaler(columns=roles.numeric_features)
        ds = scaler.fit_transform(ds).materialize()
        _, ncols, gib = ds_stats(ds)
        rec.update(rows=ds.count(), out_cols=ncols, gib=gib)

    # ---- write (the ONLY saved dataset: final processed parquet) ---------- #
    if not args.no_write:
        with m.stage("write", ncols) as rec:
            _prepare_output_dir(out_dir, overwrite=args.overwrite)
            ds.write_parquet(out_dir)
            rec.update(rows=rows_in, out_cols=ncols, gib=gib)

    # ---- verify SAVED output is globally sorted (scalable, untimed) -------- #
    # Re-read the written parquet and check global sortedness block-by-block
    # without ever loading all keys into local pandas. For --no-write smoke runs
    # we run the same scalable check on the in-memory final dataset instead.
    if args.skip_saved_sort_verify:
        verify = None
        P("\nskipping saved-output sortedness verification "
          "(--skip-saved-sort-verify)")
    elif not args.no_write:
        P(f"\nverifying saved-output global sortedness over {roles.sort_key} ...")
        verify = _verify_saved_sorted(out_dir, roles.sort_key, rows_in)
    else:
        verify = _verify_dataset_sorted(ds, roles.sort_key, rows_in)
    if verify is not None:
        P(f"  verification: sorted={verify['globally_sorted']} "
          f"rows_counted={verify['rows_counted']:,}/{verify['expected_rows']:,} "
          f"blocks={verify['n_blocks']} "
          f"(in_block_sorted={verify['all_blocks_internally_sorted']}, "
          f"boundaries_ok={verify['adjacent_boundaries_ok']})")

    # ---- sanity checks (sampled) ------------------------------------------ #
    checks = _sanity_checks(ds, roles, rows_in, verify, out_dir, args.no_write)

    # ---- report ----------------------------------------------------------- #
    _print_report(m, roles, checks, verify, scaler, encoder, num_imputer)

    # ---- manifest --------------------------------------------------------- #
    if not args.no_write:
        manifest = _build_manifest(
            args, days, roles, rows_in, m, scaler, encoder, num_imputer,
            cat_imputer, n_blocks, verify, osm_bytes,
        )
        _write_manifest(out_dir, manifest)
        P(f"\nwrote output parquet + manifest.json -> {out_dir}")

    ray.shutdown()


# --------------------------------------------------------------------------- #
# Verification / reporting helpers
# --------------------------------------------------------------------------- #
def _block_keys_summary(tbl: "pa.Table", sort_key: List[str]) -> "pa.Table":
    """Reduce ONE block to a 1-row summary: row count, whether the block is
    internally nondecreasing on the FULL key, the block's first/last key, and
    (if a ``path`` column from ``include_paths`` is present) the min/max source
    file path. Heavy work stays distributed in the block; only this tiny summary
    is collected to the driver."""
    n = tbl.num_rows
    has_path = "path" in tbl.schema.names
    rec: Dict[str, Any] = {"n_rows": pa.array([n], pa.int64())}
    if n == 0:
        in_sorted = True
        for c in sort_key:
            ftype = tbl.schema.field(c).type
            rec[f"first__{c}"] = pa.array([None], ftype)
            rec[f"last__{c}"] = pa.array([None], ftype)
        if has_path:
            rec["min_path"] = pa.array([None], pa.string())
            rec["max_path"] = pa.array([None], pa.string())
    else:
        # Stable sort_indices of an already-sorted block (even with ties) is the
        # identity permutation; any disorder makes it differ -> exact, null/tie
        # safe nondecreasing check, all in C++.
        keys_tbl = tbl.select(list(sort_key))
        idx = pc.sort_indices(keys_tbl, sort_keys=[(c, "ascending") for c in sort_key])
        in_sorted = bool(
            np.array_equal(idx.to_numpy(zero_copy_only=False), np.arange(n))
        )
        for c in sort_key:
            ftype = tbl.schema.field(c).type
            col = tbl.column(c)
            rec[f"first__{c}"] = pa.array([col[0].as_py()], ftype)
            rec[f"last__{c}"] = pa.array([col[n - 1].as_py()], ftype)
        if has_path:
            p = tbl.column("path")
            rec["min_path"] = pa.array([pc.min(p).as_py()], pa.string())
            rec["max_path"] = pa.array([pc.max(p).as_py()], pa.string())
    rec["in_block_sorted"] = pa.array([in_sorted], pa.bool_())
    return pa.table(rec)


def _verify_blocks_dataset(ds, sort_key: List[str], expected_rows: int) -> Dict[str, Any]:
    """Scalable global-sortedness check. Each block is reduced (distributed) to a
    tiny first/last-key summary; only those summaries are collected. We then put
    the blocks into canonical order and check that each block is internally
    nondecreasing and that adjacent blocks do not overlap
    (block_i.last <= block_{i+1}.first), plus that rows seen == expected. Never
    loads all keys into local pandas.

    Canonical order: when the blocks carry their source file path (saved-parquet
    case, ``include_paths=True``), we order blocks by ``(min_path, first_key)``
    -- i.e. exactly the filename order a consumer reading the output sees -- which
    is robust to Ray's parallel reader NOT emitting blocks in filename order.
    Without paths (in-memory --no-write case) we order by the key itself."""
    summ = ds.map_batches(
        lambda t: _block_keys_summary(t, sort_key),
        batch_format="pyarrow",
        batch_size=None,
    )
    rows = [r for r in summ.take_all() if int(r["n_rows"]) > 0]
    has_path = bool(rows) and ("min_path" in rows[0])

    def _key(r):
        return tuple(r[f"first__{c}"] for c in sort_key)

    if has_path:
        rows.sort(key=lambda r: (r["min_path"], _key(r)))
    else:
        rows.sort(key=_key)

    total = 0
    n_blocks = 0
    blocks_spanning_files = 0
    all_in_block = True
    boundaries_ok = True
    prev_last = None
    for r in rows:
        n_blocks += 1
        total += int(r["n_rows"])
        if not bool(r["in_block_sorted"]):
            all_in_block = False
        if has_path and r["min_path"] != r["max_path"]:
            blocks_spanning_files += 1
        first = _key(r)
        last = tuple(r[f"last__{c}"] for c in sort_key)
        if prev_last is not None and not (prev_last <= first):
            boundaries_ok = False
        prev_last = last

    rows_match = total == expected_rows
    res = {
        "globally_sorted": bool(all_in_block and boundaries_ok),
        "all_blocks_internally_sorted": bool(all_in_block),
        "adjacent_boundaries_ok": bool(boundaries_ok),
        "rows_counted": int(total),
        "expected_rows": int(expected_rows),
        "rows_match": bool(rows_match),
        "n_blocks": int(n_blocks),
        "sort_key": list(sort_key),
        "ordered_by": "source_file_path" if has_path else "key",
        "method": "blockwise-readback",
    }
    if has_path:
        res["blocks_spanning_multiple_files"] = int(blocks_spanning_files)
    return res


def _verify_saved_sorted(out_dir: str, sort_key: List[str], expected_rows: int) -> Dict[str, Any]:
    """Re-read the SAVED parquet (only the sort-key columns) and verify global
    sortedness with the scalable block-wise pass. This is the authoritative
    check: it validates exactly what a consumer reading the output in canonical
    (filename) order would see.

    Ray's parallel parquet reader does NOT emit blocks in filename order (it
    bin-packs files into size-balanced tasks and parallelises them), so we cannot
    rely on read/emission order. Instead we read with ``include_paths=True`` and
    order the per-block summaries by their source file path -- the output files
    are named ``..._<task_index:06>-...parquet`` in global-sort order, so path
    order == sort order -- making the adjacent-block boundary check exact and
    independent of how Ray schedules the read."""
    if criteo.is_s3(out_dir):
        import pyarrow.fs as pafs

        fs, path = pafs.FileSystem.from_uri(out_dir)
        infos = fs.get_file_info(
            pafs.FileSelector(path, recursive=True, allow_not_found=True)
        )
        files = sorted(
            "s3://" + i.path
            for i in infos
            if i.type == pafs.FileType.File and i.path.endswith(".parquet")
        )
    else:
        import glob

        files = sorted(glob.glob(os.path.join(out_dir, "*.parquet")))
    if not files:
        raise RuntimeError(f"no parquet files found under {out_dir}")
    vds = ray.data.read_parquet(files, columns=list(sort_key), include_paths=True)
    res = _verify_blocks_dataset(vds, sort_key, expected_rows)
    res["method"] = "blockwise-readback-path-ordered"
    res["n_files"] = len(files)
    return res


def _verify_dataset_sorted(ds, sort_key: List[str], expected_rows: int) -> Dict[str, Any]:
    """Same scalable check on an in-memory dataset (used for --no-write runs).
    No source paths exist, so blocks are ordered by key."""
    res = _verify_blocks_dataset(ds.select_columns(sort_key), sort_key, expected_rows)
    res["method"] = "blockwise-in-memory"
    return res


def _sanity_checks(ds, roles, rows_in: int, verify: Optional[Dict[str, Any]],
                   out_dir: str, no_write: bool) -> Dict[str, bool]:
    # Dtype checks use the Arrow schema (Ray returns pyarrow-backed pandas dtypes
    # like "double[pyarrow]", so checking str(pandas dtype) is unreliable).
    sch = ds.schema()
    names = set(sch.names)
    types = dict(zip(sch.names, sch.types))
    num = roles.numeric_features
    cat = roles.categorical
    inds = [f"{c}_isnull" for c in roles.indicator_cols]
    meta = roles.metadata_keys  # user_id, day_int, display_order (kept raw)

    encoded_integer = all(pa.types.is_integer(types[c]) for c in cat)
    scaled_float = all(pa.types.is_floating(types[c]) for c in num)
    indicators_int = all(pa.types.is_integer(types[c]) for c in inds) if inds else True

    # Null + value checks on a sample (pandas isnull works on pyarrow dtypes).
    sample = ds.limit(200_000).to_pandas()
    no_null_features = bool(sample[num + cat].isnull().to_numpy().sum() == 0)
    indicators_binary = (
        all({int(x) for x in sample[c].dropna().unique()} <= {0, 1} for c in inds)
        if inds
        else True
    )
    targets_present = all(c in names for c in roles.targets)

    # Metadata / sort keys: PRESENT, non-null, and NOT transformed features.
    transformed = set(cat) | set(num) | set(inds)
    meta_present = all(c in names for c in meta)
    meta_not_features = all(c not in transformed for c in meta)
    meta_no_null = bool(sample[meta].isnull().to_numpy().sum() == 0)

    # Final parquet reloads with the expected key/target columns.
    reload_ok = True
    if not no_write:
        try:
            rds = ray.data.read_parquet(out_dir)
            reload_ok = (set(roles.targets) | set(meta)).issubset(set(rds.schema().names))
        except Exception:
            reload_ok = False

    checks: Dict[str, bool] = {
        "row_count_preserved": ds.count() == rows_in,
        "final_parquet_reloads": reload_ok,
    }
    # The saved-output sortedness checks only exist when verification ran
    # (--skip-saved-sort-verify omits them rather than reporting a false PASS).
    if verify is not None:
        checks["saved_output_globally_sorted"] = bool(verify["globally_sorted"])
        checks["saved_output_row_count_matches"] = bool(verify["rows_match"])
    checks["no_nulls_in_features"] = no_null_features
    checks["encoded_cols_integer"] = encoded_integer
    checks["scaled_cols_float"] = scaled_float
    checks["indicators_binary"] = indicators_binary and indicators_int
    checks["targets_present"] = targets_present
    checks["metadata_keys_present_not_features"] = bool(
        meta_present and meta_not_features and meta_no_null
    )
    return checks


def _gpu_target_subtotal(m: Metrics) -> float:
    return sum(rec["secs"] for rec in m.stages if rec["name"] in GPU_TARGET_STAGES)


def _print_report(m: Metrics, roles, checks, verify, scaler, encoder, num_imputer) -> None:
    P("\n" + "-" * 78)
    P("per-stage wall (RAM -> RAM, each stage materialized):")
    P(f"  {'stage':<8} {'secs':>9} {'Mrows/s':>9} {'cols':>11} {'GiB':>7}   note")
    total = 0.0
    for rec in m.stages:
        total += rec["secs"]
        mrows = (rec.get("rows", 0) / rec["secs"] / 1e6) if rec["secs"] else 0.0
        cols = f"{rec['in_cols']}->{rec.get('out_cols','?')}"
        note = "GPU TARGET" if rec["name"] in GPU_TARGET_STAGES else ""
        P(f"  {rec['name']:<8} {rec['secs']:>9.2f} {mrows:>9.2f} {cols:>11} "
          f"{rec.get('gib',0):>7.2f}   {note}")
    P("  " + "-" * 60)
    P(f"  {'TOTAL':<8} {total:>9.2f}")
    sub = _gpu_target_subtotal(m)
    pct = (sub / total * 100.0) if total else 0.0
    P(f"  {'GPU-tgt':<8} {sub:>9.2f}   (sort + encode + scale = {pct:.1f}% of TOTAL)")

    if verify is None:
        P("\nsaved-output sortedness verification: SKIPPED "
          "(--skip-saved-sort-verify)")
    else:
        P("\nsaved-output sortedness verification:")
        P(f"  globally_sorted = {verify['globally_sorted']}  "
          f"(in_block_sorted={verify['all_blocks_internally_sorted']}, "
          f"boundaries_ok={verify['adjacent_boundaries_ok']})")
        P(f"  rows_counted = {verify['rows_counted']:,} / expected {verify['expected_rows']:,}  "
          f"(match={verify['rows_match']}), blocks={verify['n_blocks']}, "
          f"key={verify['sort_key']}, method={verify['method']}")

    P("\nsanity checks:")
    for name, ok in checks.items():
        P(f"  {'PASS' if ok else 'FAIL'}  {name}")

    P("\nfitted-stat summary:")
    vocabs = sorted(
        ((c, _vocab_size(encoder.stats_[f"unique_values({c})"])) for c in roles.categorical),
        key=lambda x: -x[1],
    )
    P(f"  ordinal-encoder vocab sizes (top 6 of {len(vocabs)}): "
      + ", ".join(f"{c}={n:,}" for c, n in vocabs[:6]))
    P(f"  total embedding rows (sum of vocab): {sum(n for _, n in vocabs):,}")
    P(f"  scaled numeric columns: {len(roles.numeric_features)} "
      f"(missing-indicators kept unscaled: {len(roles.indicator_cols)})")

    P("\nML decision log:")
    for line in _decision_log(roles):
        P(f"  - {line}")


def _decision_log(roles) -> List[str]:
    return [
        "Stage order sort-before-impute: prep prunes columns first so the "
        "order-setting GPU-target sort moves the smallest row.",
        "Pre-display only: dropped 80 features_not_available_* (not at inference) "
        "+ 3 *_delay_after_display_array (post-display label leakage) + id.",
        "nb_sales: sales_count = coalesce(nb_sales, 0); is_sale = nb_sales > 0 "
        "(target derivation, not feature imputation).",
        f"Missing-not-at-random: added {len(roles.indicator_cols)} <col>_isnull "
        f"indicators (null_frac > {roles.null_indicator_threshold}); kept 0/1, never scaled.",
        "Impute on CPU (not GpuSimpleImputer): numeric mean, categorical "
        "most_frequent only for the cols that actually have nulls.",
        f"Sort key {roles.sort_key}: sessionizes each user's impressions in "
        "(day_int,) display order. user_id, day_int and display_order are kept "
        "RAW in the output as metadata / sort keys (never encoded or scaled), so "
        "the saved parquet stays globally sortable and directly verifiable. "
        "Encoding user_id as a feature is a separate high-cardinality stress mode.",
        "Encode: OrdinalEncoder integer codes for embedding tables, not one-hot.",
        "Scale: StandardScaler (z-score) on numeric features only -- never on the "
        "category codes, the 0/1 indicators, or the sort keys (display_order is a "
        "raw sort key here, not a scaled feature).",
    ]


def _build_manifest(args, days, roles, rows_in, m, scaler, encoder, num_imputer,
                    cat_imputer, n_blocks, verify, osm_bytes):
    scaler_stats = {
        c: {
            "mean": scaler.stats_[f"mean({c})"],
            "std": scaler.stats_[f"std({c})"],
        }
        for c in roles.numeric_features
    }
    return {
        "dataset": "CriteoPrivateAd",
        "days": days,
        "days_spec": args.days if args.days is not None else str(args.day),
        "rows": rows_in,
        "rows_metadata_total": roles.total_rows,
        "blocks_setting": "auto" if args.blocks is None else args.blocks,
        "num_blocks_read": n_blocks,
        "ray_address": args.ray_address,
        "data_root": criteo.DATA_ROOT,
        "object_store_gib": (
            round(osm_bytes / 1024 ** 3, 1) if osm_bytes else None
        ),
        "sort_key": roles.sort_key,
        "null_indicator_threshold": args.null_indicator_threshold,
        "columns": {
            "targets": roles.targets,
            "metadata_sort_keys": roles.metadata_keys,
            "categorical": roles.categorical,
            "numeric": roles.numeric_features,
            "missing_indicators": [f"{c}_isnull" for c in roles.indicator_cols],
            "list_len": roles.list_len_cols,
        },
        "dropped": {k: v for k, v in roles.dropped.items()},
        "dropped_counts": {k: len(v) for k, v in roles.dropped.items()},
        "fitted": {
            "scaler": scaler_stats,
            "encoder_vocab_size": {
                c: _vocab_size(encoder.stats_[f"unique_values({c})"])
                for c in roles.categorical
            },
            "imputer_mean": {
                c: num_imputer.stats_[f"mean({c})"] for c in roles.impute_numeric
            },
            "imputer_most_frequent": (
                {
                    c: cat_imputer.stats_[f"most_frequent({c})"]
                    for c in roles.impute_categorical
                }
                if cat_imputer is not None
                else {}
            ),
        },
        "stage_timings_s": {rec["name"]: round(rec["secs"], 3) for rec in m.stages},
        "gpu_target_stages": sorted(GPU_TARGET_STAGES),
        "gpu_target_subtotal_s": round(_gpu_target_subtotal(m), 3),
        "verification": verify if verify is not None else {"skipped": True},
    }


if __name__ == "__main__":
    main()
