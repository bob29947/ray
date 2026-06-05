"""Standalone benchmark: CPU SimpleImputer vs GpuSimpleImputer.

RAM in -> RAM out, warm, best-of-N. Headline is ``most_frequent`` over string
categoricals (where the CPU path counts with Python ``Counter``); ``mean`` over
a numeric column is reported as the secondary, transfer-bound case. Run:

    RAY_DATA_GPU_PREPROC_NUM_GPUS=8 .venv/bin/python \
        gpu_preproc_bench/bench_simple_imputer.py --rows 50000000 --blocks 64
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import ray  # noqa: E402
from common import best_of, gib, make_recsys_dataset, nbytes  # noqa: E402

from ray.data.preprocessors import GpuSimpleImputer, SimpleImputer  # noqa: E402

# High-cardinality nullable categorical: the case where the CPU most_frequent
# (Python Counter merge) is slow and the GPU wins.
MF_HIGH_CARD = ["last_item_id"]
# Low-cardinality categoricals + numeric mean: CPU territory (trivial value_counts
# / mean), reported honestly as the transfer/startup-bound secondary case.
MF_LOW_CARD = ["event_type", "device_type", "country"]
MEAN_COLUMNS = ["price"]
P = lambda *a: print(*a, flush=True)  # noqa: E731


def _aligned(df):
    return df.sort_values("row_id").reset_index(drop=True)


def correctness(ds) -> bool:
    sample = ds.limit(200_000)
    mf_cols = MF_HIGH_CARD + MF_LOW_CARD
    cpu = _aligned(
        SimpleImputer(columns=mf_cols, strategy="most_frequent")
        .fit_transform(sample)
        .to_pandas()
    )
    gpu = _aligned(
        GpuSimpleImputer(columns=mf_cols, strategy="most_frequent")
        .fit_transform(sample)
        .to_pandas()
    )
    mf_ok = all((cpu[c] == gpu[c]).all() for c in mf_cols)

    cpu_m = _aligned(
        SimpleImputer(columns=MEAN_COLUMNS, strategy="mean")
        .fit_transform(sample)
        .to_pandas()
    )
    gpu_m = _aligned(
        GpuSimpleImputer(columns=MEAN_COLUMNS, strategy="mean")
        .fit_transform(sample)
        .to_pandas()
    )
    mean_ok = np.allclose(cpu_m["price"].values, gpu_m["price"].values, rtol=1e-6)
    return mf_ok and mean_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50_000_000)
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    ray.init(logging_level="ERROR", include_dashboard=False)
    ngpus = int(os.environ.get("RAY_DATA_GPU_PREPROC_NUM_GPUS", "1"))
    ds = make_recsys_dataset(args.rows, args.blocks)
    P(f"dataset: {args.rows:,} rows x {args.blocks} blocks "
      f"({gib(nbytes(ds)):.1f} GiB), GPUs={ngpus}")
    P(f"correctness vs CPU SimpleImputer (200k sample): "
      f"{'PASS' if correctness(ds) else 'FAIL'}")

    def run(label, cols, strategy):
        cpu_b, cpu_m, _ = best_of(
            lambda: SimpleImputer(columns=cols, strategy=strategy)
            .fit_transform(ds)
            .materialize(),
            trials=args.trials,
        )
        gpu_b, gpu_m, _ = best_of(
            lambda: GpuSimpleImputer(columns=cols, strategy=strategy)
            .fit_transform(ds)
            .materialize(),
            trials=args.trials,
        )
        P("")
        P(f"{label}  fit_transform RAM -> RAM (best / median):")
        P(f"  CPU SimpleImputer  : {cpu_b:7.2f}s / {cpu_m:7.2f}s")
        P(f"  GpuSimpleImputer   : {gpu_b:7.2f}s / {gpu_m:7.2f}s")
        P(f"  SPEEDUP (best)     : {cpu_b / gpu_b:6.2f}x")

    P("\n--- headline: most_frequent on a high-cardinality column ---")
    run(f"most_frequent {MF_HIGH_CARD} (high-card)", MF_HIGH_CARD, "most_frequent")
    P("\n--- secondary (CPU territory): low-card most_frequent + numeric mean ---")
    run(f"most_frequent {MF_LOW_CARD} (low-card)", MF_LOW_CARD, "most_frequent")
    run(f"mean {MEAN_COLUMNS}", MEAN_COLUMNS, "mean")
    ray.shutdown()


if __name__ == "__main__":
    main()
