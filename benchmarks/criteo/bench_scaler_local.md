# Standalone StandardScaler benchmark (GPU vs CPU)

Driver: [`bench_scaler_local.py`](bench_scaler_local.py).

This isolates **just the scaler** from the CriteoPrivateAd pipeline: it
materializes `read -> prep` into RAM once (untimed), then times **only**
`StandardScaler` (fit + transform + materialize back to RAM) on
`roles.numeric_features`. It models an isolated GPU StandardScaler dropped into a
real pipeline -- everything upstream is already materialized, the scaler runs,
and the clock stops when all data is scaled and back in RAM. **All scaler
overhead is inside the timer**: the mean/std fit reduction, the GPU actor-pool /
CUDA-context / cuDF startup, H2D/D2H, compute, and the final materialize
(`--warmup 0`, fresh actor pools per repeat).

## TL;DR

On this box (16x V100-SXM3-32GB), pinned to **4 GPUs + 64 CPUs**, lean recipe,
scaling **44 columns** (43 float features + 1 list-length feature):

- **CPU wins at every scale tested.** Even fully tuned, the GPU scaler is
  **~0.6-0.8x** the 64-CPU baseline.
- **Large device batches are the dominant GPU lever** (the fit reduction is the
  bottleneck): batch 2M -> 4M cut the GPU fit ~30-40%.
- **Fractional GPU packing helps the transfer-bound transform** (~27s -> ~18.5s
  at 100M rows by running 2-4 actors/GPU), and helps the fit at large batch.
  Sweet spot is ~2 actors/GPU; 16 actors barely beats 8. Tuning nearly doubled
  GPU throughput (1.38 -> 2.61 Mrows/s) but not enough to overtake CPU.
- Every GPU variant's fitted mean/std **matched the CPU scaler** (parity OK).

The standalone scaler is trivially light compute (one subtract + divide per
value) and is **memory-bandwidth / PCIe-transfer bound**. 64 CPU cores doing an
in-RAM Arrow pass beat shipping 44 columns to only 4 GPUs and back -- especially
the fit reduction, where 64-core aggregation beats a 4-GPU transfer-bound scan.
This is consistent with the repo's existing result that the GPU advantage comes
from the **fused** impute+encode+scale stage (the heavy ordinal-encode amortizes
the PCIe round trip; see README section 4), not from scaling alone.

> The verdict is specific to the requested **4 GPU / 64 CPU** split. With more
> GPUs the balance shifts toward the GPU; `--gpus 16` would find the crossover.

## Setup

- Hardware: 16x Tesla V100-SXM3-32GB, 96 CPUs, 756 GiB `/dev/shm`. Pinned via a
  single `ray.init(num_cpus=64, num_gpus=4)`; the CPU baseline uses only CPUs
  (no GPU code path), the GPU sweep uses the 4 GPUs.
- Recipe: `lean` (drops the 80 `features_not_available_*` and the all-null
  `features_kv_bits_constrained_5`).
- Columns scaled: **44** = `roles.numeric_features` (43 raw float features + 1
  list-length feature `features_ctx_not_constrained_2_len`). Raw numerics may
  contain nulls (the full pipeline imputes upstream); StandardScaler ignores
  nulls in fit and transform on both CPU and GPU, so this is a valid throughput
  benchmark for the scaler in isolation.
- Timed region = full scaler op (`fit` + `transform` + `materialize`); `fit` and
  `transform` are also reported separately. Headline is `total_s`.

## The GPU fraction knob

The documented `RAY_DATA_GPU_PREPROC_GPU_FRACTION` was previously not
implemented (every GPU `map_batches` hardcoded `num_gpus=1`). It is now real:
`env_gpu_fraction()` in
[`python/ray/data/preprocessors/_gpu.py`](../../python/ray/data/preprocessors/_gpu.py)
sets the `num_gpus` each actor requests, wired into all 6 GPU `map_batches`
calls (fit reductions + transform). Default `1.0` is byte-for-byte the old
`num_gpus=1` behavior. A fraction `< 1` packs multiple actors per GPU, so
`concurrency` can exceed the physical GPU count (e.g. `0.25` -> 4 actors/GPU).

## Results -- all 30 days (103,862,032 rows, 49.91 GiB)

`repeats=2, warmup=0`. **CPU baseline: 25.39s (4.09 Mrows/s).**

| GPU config | actors/GPU | batch | fit_s | xform_s | total_s | Mrows/s | vs CPU |
|---|---|---|---|---|---|---|---|
| frac=1.0, 4 actors | 1 | 2.0M | 48.24 | 27.27 | 75.52 | 1.38 | 0.34x |
| frac=1.0, 4 actors | 1 | 4.0M | 30.38 | 25.98 | 56.36 | 1.84 | 0.45x |
| frac=0.5, 4 actors | 2 | 2.0M | 48.09 | 27.38 | 75.47 | 1.38 | 0.34x |
| frac=0.5, 4 actors | 2 | 4.0M | 30.80 | 27.68 | 58.47 | 1.78 | 0.43x |
| frac=0.5, 8 actors | 2 | 2.0M | 33.59 | 18.81 | 52.40 | 1.98 | 0.48x |
| frac=0.5, 8 actors | 2 | 4.0M | 22.21 | 18.52 | 40.72 | 2.55 | 0.62x |
| frac=0.25, 4 actors | 4 | 2.0M | 50.70 | 27.55 | 78.24 | 1.33 | 0.32x |
| frac=0.25, 4 actors | 4 | 4.0M | 31.61 | 27.89 | 59.50 | 1.75 | 0.43x |
| frac=0.25, 8 actors | 4 | 2.0M | 34.72 | 19.24 | 53.96 | 1.92 | 0.47x |
| frac=0.25, 8 actors | 4 | 4.0M | 22.77 | 18.89 | 41.67 | 2.49 | 0.61x |
| frac=0.25, 16 actors | 4 | 2.0M | 32.60 | 18.45 | 51.05 | 2.03 | 0.50x |
| **frac=0.25, 16 actors** | **4** | **4.0M** | **21.04** | **18.75** | **39.80** | **2.61** | **0.64x** |

Fastest GPU = `frac=0.25, 16 actors (4/GPU), batch 4M` -> 39.80s (0.64x), with
`frac=0.5, 8 actors (2/GPU), batch 4M` -> 40.72s essentially tied.

## Results -- days 1-3 (14,727,673 rows, 7.05 GiB)

`repeats=3, warmup=0`. **CPU baseline: 10.23s (1.44 Mrows/s).** Selected rows
(full sweep = fractions {1.0, 0.5, 0.25} x actors {4, 8, 16} x batch
{auto, 1M, 2M, 4M}):

| GPU config | actors/GPU | batch | total_s | vs CPU |
|---|---|---|---|---|
| frac=1.0, 4 actors | 1 | 1.0M | 20.11 | 0.51x |
| frac=1.0, 4 actors | 1 | 4.0M | 13.88 | 0.74x |
| frac=0.5, 8 actors | 2 | 2.0M | 13.24 | 0.77x |
| frac=0.25, 8 actors | 4 | 2.0M | 12.98 | 0.79x |
| **frac=0.25, 8 actors** | **4** | **4.0M** | **12.83** | **0.80x** |

Same shape as the 100M run: batch 1M is consistently worst (fit-bound),
2-4M best; packing to ~2-4 actors/GPU helps; GPU stays below the CPU baseline.

## Tuning findings

1. **Batch size is the biggest GPU lever.** The fit (mean/std reduction) cost
   scales with the number of device blocks, so bigger batches = fewer blocks =
   far less per-block `from_arrow` / reduction overhead. At 100M: fit 48s (2M) ->
   22s (4M) at 2 actors/GPU.
2. **Fractional packing helps the transform.** The transform is transfer-bound
   (trivial compute), so overlapping 2-4 actors per GPU hides H2D/D2H latency:
   transform 27s (1/GPU) -> ~18.5s (2-4/GPU) at 100M. It also lifts the fit at
   large batch. Returns flatten past ~2 actors/GPU (8 vs 16 actors are close).
3. **It is not enough.** The best tuned GPU config is still ~0.6x the CPU
   baseline at 100M (and ~0.8x at 15M), because the op is bandwidth/transfer
   bound and 4 GPUs have far less aggregate bandwidth for this than 64 CPU cores
   doing an in-RAM Arrow pass.

## How to run

GPU runs need a real CUDA device, so run OUTSIDE any sandbox.

```bash
# quick smoke (row-capped)
.venv/bin/python benchmarks/criteo/bench_scaler_local.py --days 1 --rows 2000000

# full dataset, focused large-batch sweep (what produced the 100M table above)
.venv/bin/python benchmarks/criteo/bench_scaler_local.py \
  --days all --gpu-batch-sizes 2000000,4000000 --repeats 2 \
  --out benchmarks/criteo/data/bench_scaler_all30.json

# default multi-day sweep (fractions x actors x batch incl. auto/1M)
.venv/bin/python benchmarks/criteo/bench_scaler_local.py --days 1-3
```

Key flags: `--cpus 64 --gpus 4` (resource pin), `--gpu-fractions`,
`--gpu-actors`, `--gpu-batch-sizes` (`auto` = VRAM-aware sizer), `--feature-set
lean|wide`, `--warmup` (default 0 keeps cold-start overhead in the number).
Results JSON is written under `benchmarks/criteo/data/` (git-ignored).
