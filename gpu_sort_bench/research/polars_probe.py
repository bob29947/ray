"""Isolate where the polars time goes, single process, no Ray.

  1) standalone polars   df.sort("c0")                  <- polars at its best
  2) standalone pyarrow  table.sort_by("c0")            <- the Ray CPU kernel
  3) polars round-trip   pl.from_arrow(t).sort().to_arrow()   <- what Ray does per block

This tells us whether we're "using polars wrong" or whether the Ray integration
(per-block Arrow<->polars conversion + the object-store shuffle) is the cost.
"""
import argparse
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl


def build(rows, cols):
    rng = np.random.default_rng(0)
    data = {"c0": rng.integers(0, 2**31 - 1, rows, dtype=np.int32)}
    for j in range(1, cols):
        data[f"c{j}"] = np.zeros(rows, dtype=np.int32)
    return data


def best_of(fn, n=3):
    fn()  # warmup
    b = min(fn() for _ in range(n))
    return b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=64 * 1024 * 1024)  # ~4 GiB
    p.add_argument("--cols", type=int, default=16)
    args = p.parse_args()
    gib = args.rows * args.cols * 4 / 2**30
    print(f"rows={args.rows:,} cols={args.cols} = {gib:.1f} GiB   "
          f"polars threads={pl.thread_pool_size()}")

    data = build(args.rows, args.cols)
    tbl = pa.table(data)
    df = pl.DataFrame(data)

    def arrow_sort():
        t0 = time.perf_counter()
        s = tbl.sort_by("c0")
        dt = time.perf_counter() - t0
        assert s.num_rows == args.rows
        return dt

    def polars_sort():
        t0 = time.perf_counter()
        s = df.sort("c0")
        dt = time.perf_counter() - t0
        assert s.height == args.rows
        return dt

    def polars_roundtrip():
        t0 = time.perf_counter()
        s = pl.from_arrow(tbl).sort("c0").to_arrow()
        dt = time.perf_counter() - t0
        assert s.num_rows == args.rows
        return dt

    a = best_of(arrow_sort)
    pol = best_of(polars_sort)
    rt = best_of(polars_roundtrip)

    print(f"\nstandalone pyarrow  table.sort_by : {a*1e3:8.1f} ms  ({gib/a:6.2f} GiB/s)")
    print(f"standalone polars   df.sort        : {pol*1e3:8.1f} ms  ({gib/pol:6.2f} GiB/s)")
    print(f"polars from_arrow->sort->to_arrow  : {rt*1e3:8.1f} ms  ({gib/rt:6.2f} GiB/s)")
    print(f"\npolars vs pyarrow (pure kernel): {a/pol:.2f}x")
    print(f"arrow<->polars conversion tax  : {rt - pol:.3f} s  ({100*(rt-pol)/rt:.0f}% of the round-trip)")


if __name__ == "__main__":
    main()
