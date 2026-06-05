"""Standalone benchmark: CPU OrdinalEncoder vs GpuOrdinalEncoder.

RAM in -> RAM out, warm, best-of-N. Encodes the categorical columns of a
recommender-style event table. Run directly:

    RAY_DATA_GPU_PREPROC_NUM_GPUS=8 .venv/bin/python \
        gpu_preproc_bench/bench_ordinal_encoder.py --rows 50000000 --blocks 64
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ray  # noqa: E402
from common import best_of, gib, make_recsys_dataset, nbytes  # noqa: E402

from ray.data.preprocessors import GpuOrdinalEncoder, OrdinalEncoder  # noqa: E402

CATEGORICAL = ["item_id", "user_id", "event_type", "device_type", "country"]
P = lambda *a: print(*a, flush=True)  # noqa: E731


def _aligned(df):
    return df.sort_values("row_id").reset_index(drop=True)


def correctness(ds) -> bool:
    sample = ds.limit(200_000)
    cpu = _aligned(
        OrdinalEncoder(columns=CATEGORICAL).fit_transform(sample).to_pandas()
    )
    gpu = _aligned(
        GpuOrdinalEncoder(columns=CATEGORICAL).fit_transform(sample).to_pandas()
    )
    return all((cpu[c].fillna(-1) == gpu[c].fillna(-1)).all() for c in CATEGORICAL)


def split(EncCls, ds):
    enc = EncCls(columns=CATEGORICAL)
    t = time.perf_counter()
    enc.fit(ds)
    ft = time.perf_counter() - t
    t = time.perf_counter()
    enc.transform(ds).materialize()
    tt = time.perf_counter() - t
    return ft, tt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    ray.init(logging_level="ERROR", include_dashboard=False)
    ngpus = int(os.environ.get("RAY_DATA_GPU_PREPROC_NUM_GPUS", "1"))
    # OrdinalEncoder (CPU and GPU) requires non-null categoricals; imputation is
    # the imputer's job, so encode a clean table here.
    ds = make_recsys_dataset(args.rows, args.blocks, null_frac=0.0)
    P(f"dataset: {args.rows:,} rows x {args.blocks} blocks "
      f"({gib(nbytes(ds)):.1f} GiB), columns={CATEGORICAL}, GPUs={ngpus}")

    P(f"correctness vs CPU OrdinalEncoder (200k sample): "
      f"{'PASS' if correctness(ds) else 'FAIL'}")

    cpu_b, cpu_m, _ = best_of(
        lambda: OrdinalEncoder(columns=CATEGORICAL).fit_transform(ds).materialize(),
        trials=args.trials,
    )
    gpu_b, gpu_m, _ = best_of(
        lambda: GpuOrdinalEncoder(columns=CATEGORICAL).fit_transform(ds).materialize(),
        trials=args.trials,
    )
    cf, ct = split(OrdinalEncoder, ds)
    gf, gt = split(GpuOrdinalEncoder, ds)

    P("")
    P("fit_transform  RAM -> RAM (best / median):")
    P(f"  CPU OrdinalEncoder    : {cpu_b:7.2f}s / {cpu_m:7.2f}s")
    P(f"  GpuOrdinalEncoder     : {gpu_b:7.2f}s / {gpu_m:7.2f}s")
    P(f"  SPEEDUP (best)        : {cpu_b / gpu_b:6.2f}x")
    P("phase split (one run):")
    P(f"  CPU  fit={cf:7.2f}s  transform={ct:7.2f}s")
    P(f"  GPU  fit={gf:7.2f}s  transform={gt:7.2f}s   "
      f"(fit {cf / gf:.1f}x, transform {ct / gt:.2f}x)")
    ray.shutdown()


if __name__ == "__main__":
    main()
