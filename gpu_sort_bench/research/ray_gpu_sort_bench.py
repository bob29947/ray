"""Benchmark the end-to-end GPU sort wired into Ray Data's ds.sort().

Same dataset as sort.py / the microbenchmarks: 1Gi rows x 16 int32 cols (64 GiB),
sort key c0 (random int32), other columns zero.

Reports two timings the user asked for:
  * FULL    : RAM -> VRAM -> GPU sort -> VRAM -> RAM (H2D + sort + D2H), measured
              inside the GPU actor as it consumes object-store blocks and writes
              sorted object-store blocks back.
  * GPU-ONLY: in-VRAM -> sorted-in-VRAM (same contract as the microbenchmarks).

Correctness is checked on the actual sorted Ray result (row count + global order
+ key sum/min/max vs the input).
"""

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa

# Enable the experimental GPU sort path in Ray Data BEFORE planning ds.sort().
os.environ["RAY_DATA_GPU_SORT"] = "1"

import ray
from ray.data import DataContext
from ray.data._internal.planner.gpu_sort import _SORTER_NAME


def gib(n):
    return f"{n / 2**30:.2f} GiB"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024 * 1024 * 1024)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--blocks", type=int, default=256)
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--baseline-seconds", type=float, default=45.691)
    p.add_argument("--check", action="store_true", help="small-scale correctness run")
    args = p.parse_args()

    if args.check:
        args.rows = 1 << 24
        args.blocks = 64
        args.trials = 1

    os.environ["RAY_DATA_GPU_SORT_NUM_GPUS"] = str(args.gpus)
    assert args.rows % args.blocks == 0
    rows_per_block = args.rows // args.blocks
    raw_bytes = args.rows * args.cols * 4

    ray.init(object_store_memory=512 * 2**30)
    ctx = DataContext.get_current()
    ctx.enable_rich_progress_bars = False
    ctx.use_ray_tqdm = False

    print("=== ray gpu sort benchmark ===")
    print(f"rows: {args.rows:,}  cols: {args.cols}  blocks: {args.blocks}  gpus: {args.gpus}")
    print(f"rows/block: {rows_per_block:,}  dataset: {gib(raw_bytes)}")
    print(f"ray GPUs: {ray.cluster_resources().get('GPU', 0)}")

    # Build the same dataset as sort.py (identical RNG seed) and accumulate the
    # input key checksum for correctness.
    cols = [f"c{i}" for i in range(args.cols)]
    rng = np.random.default_rng(0)
    in_sum = 0
    in_count = 0
    in_min = None
    in_max = None
    refs = []
    print("\nbuilding input blocks in object store...")
    for i in range(args.blocks):
        c0 = rng.integers(0, 2**31 - 1, rows_per_block, dtype=np.int32)
        in_sum += int(c0.astype(np.int64).sum())
        in_count += c0.shape[0]
        in_min = int(c0.min()) if in_min is None else min(in_min, int(c0.min()))
        in_max = int(c0.max()) if in_max is None else max(in_max, int(c0.max()))
        data = {"c0": c0}
        for c in cols[1:]:
            data[c] = np.zeros(rows_per_block, dtype=np.int32)
        refs.append(ray.put(pa.table(data)))
    ds = ray.data.from_arrow_refs(refs)
    print(f"input built. in_count={in_count:,} in_sum={in_sum} in_min={in_min} in_max={in_max}")

    def kill_sorter():
        try:
            ray.kill(ray.get_actor(_SORTER_NAME))
        except Exception:
            pass

    kill_sorter()  # drop any stale detached actor from a previous run
    sorter_stats = lambda: ray.get(ray.get_actor(_SORTER_NAME).get_last_stats.remote())

    # Warmup (primes actor, CUDA context, cupy pool, cub workspaces).
    print("\nwarmup sort...")
    sorted_ds = ds.sort("c0").materialize()
    n_out = sorted_ds.count()
    st = sorter_stats()
    v = st["verify"]
    ok = (
        v["rows"] == args.rows
        and v["globally_sorted"]
        and v["key_sum"] == in_sum
        and v["key_min"] == in_min
        and v["key_max"] == in_max
        and n_out == args.rows
    )
    print("\n=== correctness (on the sorted Ray result) ===")
    print(f"rows out: {v['rows']:,} / materialized {n_out:,} (expected {args.rows:,}) -> "
          f"{'OK' if v['rows'] == args.rows == n_out else 'MISMATCH'}")
    print(f"globally sorted across blocks: {'OK' if v['globally_sorted'] else 'FAIL'}")
    print(f"key sum match: {'OK' if v['key_sum'] == in_sum else 'FAIL'} "
          f"(out={v['key_sum']} in={in_sum})")
    print(f"key min/max match: {'OK' if v['key_min'] == in_min and v['key_max'] == in_max else 'FAIL'} "
          f"(out=[{v['key_min']},{v['key_max']}] in=[{in_min},{in_max}])")
    print(f"output block rows: {v['block_rows']}")
    if not ok:
        print("\nCORRECTNESS FAILED")
        sys.exit(1)
    del sorted_ds

    # Timed trials.
    print("\n=== timed trials ===")
    fulls, gpus, h2ds, d2hs, walls = [], [], [], [], []
    for t in range(args.trials):
        w0 = time.perf_counter()
        sorted_ds = ds.sort("c0").materialize()
        w1 = time.perf_counter()
        st = sorter_stats()
        fulls.append(st["full_s"])
        gpus.append(st["gpu_sort_s"])
        h2ds.append(st["h2d_s"])
        d2hs.append(st["d2h_s"])
        walls.append(w1 - w0)
        print(
            f"trial {t}: full={st['full_s']:.3f}s "
            f"(h2d={st['h2d_s']:.3f} sort={st['gpu_sort_s']:.3f} d2h={st['d2h_s']:.3f}) "
            f"| gpu-only={st['gpu_sort_s']:.4f}s | ds.sort().materialize() wall={w1 - w0:.3f}s"
        )
        del sorted_ds

    def best_mean(x):
        return min(x), sum(x) / len(x)

    full_b, full_m = best_mean(fulls)
    gpu_b, gpu_m = best_mean(gpus)
    h2d_b, h2d_m = best_mean(h2ds)
    d2h_b, d2h_m = best_mean(d2hs)
    gib_total = raw_bytes / 2 ** 30

    print("\n=== RESULT ===")
    print(f"dataset: {gib(raw_bytes)} ({args.rows:,} rows x {args.cols} int32 cols)")
    print(f"FULL  RAM->VRAM->sort->VRAM->RAM (best): {full_b:.3f} s  "
          f"({gib_total / full_b:.1f} GiB/s, {args.rows / full_b:,.0f} rows/s)")
    print(f"   phase means: h2d={h2d_m:.3f}s  gpu_sort={gpu_m:.3f}s  d2h={d2h_m:.3f}s")
    print(f"   H2D (best): {h2d_b:.3f} s -> {gib_total / h2d_b:.1f} GiB/s   "
          f"(mean {gib_total / h2d_m:.1f} GiB/s)")
    print(f"   D2H (best): {d2h_b:.3f} s -> {gib_total / d2h_b:.1f} GiB/s   "
          f"(mean {gib_total / d2h_m:.1f} GiB/s)")
    print(f"GPU-ONLY in-VRAM->sorted-in-VRAM (best): {gpu_b:.4f} s  "
          f"({gib_total / gpu_b:.1f} GiB/s, {args.rows / gpu_b:,.0f} rows/s)")
    print(f"ds.sort().materialize() wall (best): {min(walls):.3f} s")
    if args.baseline_seconds > 0:
        print(f"\nray cpu baseline: {args.baseline_seconds:.3f} s")
        print(f"speedup vs cpu (FULL incl. transfers): {args.baseline_seconds / full_b:.1f}x")
        print(f"speedup vs cpu (GPU-only): {args.baseline_seconds / gpu_b:.1f}x")

    kill_sorter()
    ray.shutdown()


if __name__ == "__main__":
    main()
