"""CPU-only diagnostic to localize the fused-stage order-preservation bug.

The 30-day GPU run saved output where 53/442 files were internally unsorted on
the sort key, while the CPU baseline was perfect (0/447). The fused stage is the
only structural difference: an **actor-pool** ``map_batches`` with a large
``batch_size`` (2,000,000) that **coalesces** many small post-sort blocks into
each task and re-splits the (wide) output, run over ``concurrency`` GPU actors
with ``preserve_order=True``.

Order preservation is a property of Ray Data's block plumbing, NOT of cuDF, so
it reproduces on CPU with an identity transform and no GPU. This script mimics
the real shape as closely as possible:

* ``--upstream sort``  : shuffle then ``ds.sort("id")`` so the map's input comes
  from an all-to-all op (like the pipeline), not a pre-sorted ``range``.
* ``--width-cols N``   : the identity transform appends N float64 columns so the
  output block is **wide** and gets split by ``target_max_block_size`` (the real
  run splits each 2M batch into ~8 files), exercising the output-split path.
* ``--target-block-mb``: shrink ``target_max_block_size`` to force that split
  without needing 60+ columns.
* ``--write-dir PATH`` : write parquet and verify with the SAME path-ordered
  block readback the pipeline uses (``_verify_saved_sorted``), instead of the
  in-memory ``iter_batches`` order. This is the authoritative check.

Sweep to isolate the cause (actor vs task pool, coalescing vs per-block,
concurrency, with/without the sort upstream and the output split). Exit code is
non-zero when order is broken, so the sweep can be scripted.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
import time
from typing import List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import ray

P = lambda *a: print(*a, flush=True)  # noqa: E731


def _fatten(batch: "pa.Table", width_cols: int) -> "pa.Table":
    """Append ``width_cols`` float64 payload columns derived from ``id``.

    Makes the output block wide (like the fused stage's 59 columns) so it is
    split by ``target_max_block_size`` -- the path that produced 442 files from
    52 batches in the real run.
    """
    cols = {name: batch.column(name) for name in batch.column_names}
    if width_cols > 0:
        idf = pc.cast(batch.column("id"), pa.float64())
        for j in range(width_cols):
            cols[f"f{j}"] = pc.add(idf, float(j))
    return pa.table(cols)


class _IdentityActor:
    """Stateful (actor-pool) identity+fatten UDF -- mirrors ``_GpuBatchActor``."""

    def __init__(self, width_cols: int = 0):
        self._width_cols = width_cols

    def __call__(self, batch: "pa.Table") -> "pa.Table":
        return _fatten(batch, self._width_cols)


def _make_identity_fn(width_cols: int):
    def _identity_fn(batch: "pa.Table") -> "pa.Table":
        return _fatten(batch, width_cols)

    return _identity_fn


def _is_sorted(col) -> bool:
    """True if ``col`` is nondecreasing (null/tie safe), all in C++."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    if n <= 1:
        return True
    idx = pc.sort_indices(col)
    return bool(np.array_equal(idx.to_numpy(zero_copy_only=False), np.arange(n)))


def _check_in_memory(mapped) -> dict:
    """Read output blocks in preserve_order and check per-block monotonicity."""
    n_blocks = n_rows = unsorted = boundary_viol = 0
    prev_last = None
    for batch in mapped.iter_batches(batch_size=None, batch_format="pyarrow"):
        if batch.num_rows == 0:
            continue
        n_blocks += 1
        n_rows += batch.num_rows
        col = batch.column("id")
        if not _is_sorted(col):
            unsorted += 1
        first, last = col[0].as_py(), col[batch.num_rows - 1].as_py()
        if prev_last is not None and not (prev_last <= first):
            boundary_viol += 1
        prev_last = last
    return {
        "method": "in-memory-iter_batches",
        "n_blocks": n_blocks,
        "rows": n_rows,
        "unsorted_blocks": unsorted,
        "boundary_violations": boundary_viol,
    }


def _check_per_file(write_dir: str) -> dict:
    """Ground-truth check: read each parquet file INDIVIDUALLY with pyarrow (no
    Ray reader bin-packing) in filename order, and verify (a) each file's ``id``
    is internally sorted and (b) files in filename order have non-overlapping
    increasing ranges. This disambiguates a real on-disk reorder from a Ray
    read-back / bin-packing artifact in the pipeline's verifier."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(write_dir, "*.parquet")))
    n_files = n_rows = unsorted = boundary_viol = 0
    prev_last = None
    for f in files:
        col = pq.read_table(f, columns=["id"]).column("id")
        n = len(col)
        if n == 0:
            continue
        n_files += 1
        n_rows += n
        if not _is_sorted(col):
            unsorted += 1
        first, last = col[0].as_py(), col[n - 1].as_py()
        if prev_last is not None and not (prev_last <= first):
            boundary_viol += 1
        prev_last = last
    return {
        "method": "per-file-pyarrow-filename-ordered (ground truth)",
        "n_blocks": n_files,
        "rows": n_rows,
        "unsorted_blocks": unsorted,
        "boundary_violations": boundary_viol,
        "n_files": len(files),
    }


def _check_saved(write_dir: str) -> dict:
    """Write-readback path-ordered check, mirroring ``_verify_saved_sorted``."""
    files = sorted(glob.glob(os.path.join(write_dir, "*.parquet")))
    vds = ray.data.read_parquet(files, columns=["id"], include_paths=True)

    def _summary(t: "pa.Table") -> "pa.Table":
        n = t.num_rows
        col = t.column("id")
        return pa.table(
            {
                "n_rows": pa.array([n], pa.int64()),
                "first": pa.array([col[0].as_py() if n else None], pa.int64()),
                "last": pa.array([col[n - 1].as_py() if n else None], pa.int64()),
                "in_sorted": pa.array([_is_sorted(col) if n else True], pa.bool_()),
                "min_path": pa.array(
                    [pc.min(t.column("path")).as_py() if n else None], pa.string()
                ),
            }
        )

    summ = vds.map_batches(_summary, batch_format="pyarrow", batch_size=None)
    rows = [r for r in summ.take_all() if int(r["n_rows"]) > 0]
    rows.sort(key=lambda r: (r["min_path"], r["first"]))
    n_blocks = n_rows = unsorted = boundary_viol = 0
    prev_last = None
    for r in rows:
        n_blocks += 1
        n_rows += int(r["n_rows"])
        if not bool(r["in_sorted"]):
            unsorted += 1
        if prev_last is not None and not (prev_last <= r["first"]):
            boundary_viol += 1
        prev_last = r["last"]
    return {
        "method": "saved-readback-path-ordered",
        "n_blocks": n_blocks,
        "rows": n_rows,
        "unsorted_blocks": unsorted,
        "boundary_violations": boundary_viol,
        "n_files": len(files),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=40_000_000)
    ap.add_argument("--blocks", type=int, default=384)
    ap.add_argument(
        "--batch-size", type=int, default=2_000_000,
        help="map_batches batch_size. 0 => None (per-block, no coalescing).",
    )
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--compute", choices=["actor", "task"], default="actor")
    ap.add_argument("--upstream", choices=["range", "sort"], default="sort")
    ap.add_argument(
        "--width-cols", type=int, default=12,
        help="extra float64 cols the map appends (fattens output -> split).",
    )
    ap.add_argument(
        "--target-block-mb", type=float, default=16.0,
        help="target_max_block_size in MiB (forces output split). 0 => Ray default.",
    )
    ap.add_argument(
        "--write-dir", default=None,
        help="if set, write parquet here and run the path-ordered readback check.",
    )
    ap.add_argument(
        "--pre-repartition-rows", type=int, default=0,
        help="if >0, repartition(target_num_rows_per_block=R) after the upstream "
        "and before the map (the proposed fix shape). 0 => no repartition.",
    )
    ap.add_argument(
        "--strict", action="store_true",
        help="pass strict=True to the pre-repartition (exact R-row blocks).",
    )
    ap.add_argument(
        "--no-map", action="store_true",
        help="skip the map entirely (write the sorted/repartitioned input "
        "directly) to test write_parquet order preservation in isolation.",
    )
    ap.add_argument(
        "--force-no-threads", action="store_true",
        help="pass arrow_parquet_args={'use_threads': False} to write_parquet "
        "(verifies the use_threads root cause / fix path).",
    )
    args = ap.parse_args()

    batch_size = None if args.batch_size == 0 else args.batch_size

    ray.init(logging_level="ERROR", include_dashboard=False)
    ctx = ray.data.DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False
    ctx.execution_options.preserve_order = True
    if args.target_block_mb and args.target_block_mb > 0:
        ctx.target_max_block_size = int(args.target_block_mb * 1024 * 1024)

    P("=" * 78)
    P("fused-stage order-preservation diagnostic (CPU, identity transform)")
    P(f"  rows={args.rows:,}  input_blocks={args.blocks}  upstream={args.upstream}")
    P(f"  batch_size={'None (per-block)' if batch_size is None else f'{batch_size:,}'}"
      f"  concurrency={args.concurrency}  compute={args.compute}")
    P(f"  width_cols={args.width_cols}  target_block_mb={args.target_block_mb}  "
      f"preserve_order=True  ray={ray.__version__}")
    P("=" * 78)

    # Build a globally-sorted input over a monotonic int64 ``id``.
    if args.upstream == "range":
        ds = ray.data.range(args.rows, override_num_blocks=args.blocks)
    else:
        # Shuffle then sort so the map's input is produced by an all-to-all sort
        # (like the pipeline), not a pre-sorted range.
        ds = ray.data.range(args.rows, override_num_blocks=args.blocks)
        ds = ds.random_shuffle()
        ds = ds.sort("id")
    if args.pre_repartition_rows > 0:
        # The proposed fix: order-preserving sizing of blocks to ~R rows so the
        # GPU map runs per-block (no cross-block coalescing).
        ds = ds.repartition(
            target_num_rows_per_block=args.pre_repartition_rows,
            shuffle=False,
            strict=args.strict,
        )
    ds = ds.materialize()
    P(f"input blocks (after upstream+materialize): {ds.num_blocks()}")

    kwargs = dict(
        batch_format="pyarrow",
        zero_copy_batch=True,
        batch_size=batch_size,
        concurrency=args.concurrency,
    )
    t0 = time.perf_counter()
    if args.no_map:
        mapped = ds
        P("(--no-map: writing the sorted/repartitioned input directly)")
    elif args.compute == "actor":
        mapped = ds.map_batches(
            _IdentityActor, fn_constructor_kwargs={"width_cols": args.width_cols}, **kwargs
        )
    else:
        mapped = ds.map_batches(_make_identity_fn(args.width_cols), **kwargs)
    mapped = mapped.materialize()
    dt = time.perf_counter() - t0
    P(f"map output blocks: {mapped.num_blocks()}   (built in {dt:.1f}s)")

    if args.write_dir:
        # Decisive isolation: check the SAME materialized dataset both in-memory
        # (block order, preserve_order) and on-disk (per-file ground truth).
        in_mem = _check_in_memory(mapped)
        P("-" * 78)
        P(f"IN-MEMORY (same materialized dataset, before write):")
        P(f"  blocks={in_mem['n_blocks']}  rows={in_mem['rows']:,}  "
          f"internally-unsorted-blocks={in_mem['unsorted_blocks']}  "
          f"boundary_violations={in_mem['boundary_violations']}")
        if os.path.isdir(args.write_dir):
            shutil.rmtree(args.write_dir)
        os.makedirs(args.write_dir, exist_ok=True)
        write_args = {}
        if args.force_no_threads:
            write_args["arrow_parquet_args"] = {"use_threads": False}
        mapped.write_parquet(args.write_dir, **write_args)
        gt = _check_per_file(args.write_dir)
        P(f"ON-DISK GROUND-TRUTH (per-file pyarrow, filename order):")
        P(f"  files={gt['n_files']}  rows={gt['rows']:,}  "
          f"internally-unsorted-files={gt['unsorted_blocks']}  "
          f"boundary_violations={gt['boundary_violations']}")
        P(f"  => on-disk {'CORRECT' if gt['unsorted_blocks'] == 0 and gt['boundary_violations'] == 0 else 'GENUINELY REORDERED'}")
        res = _check_saved(args.write_dir)
    else:
        res = _check_in_memory(mapped)

    ok = res["unsorted_blocks"] == 0 and res["boundary_violations"] == 0 and res[
        "rows"
    ] == args.rows
    P("-" * 78)
    P(f"check method              : {res['method']}")
    if "n_files" in res:
        P(f"output files              : {res['n_files']}")
    P(f"output blocks checked     : {res['n_blocks']}")
    P(f"rows checked              : {res['rows']:,} / {args.rows:,} "
      f"(match={res['rows'] == args.rows})")
    P(f"internally-unsorted blocks: {res['unsorted_blocks']}  <-- the reported symptom")
    P(f"boundary violations       : {res['boundary_violations']}")
    P("-" * 78)
    P(f"RESULT: {'PASS (order preserved)' if ok else 'FAIL (order broken)'}")

    ray.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
