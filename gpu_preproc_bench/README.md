# GPU preprocessor benchmarks

Standalone benchmarks for the experimental, opt-in GPU Ray Data preprocessors,
mirroring the methodology of `gpu_sort_bench`:

- **RAM in -> RAM out**: the input dataset is materialized in the object store
  (RAM); we time `preprocessor.fit_transform(ds).materialize()` (output back in
  RAM). Each GPU operator pulls Arrow blocks, moves the needed columns to a GPU
  as cuDF, computes, and writes Arrow blocks back. No device-resident state is
  carried between operators.
- **Warm, best-of-N**: one uncounted warmup, then N timed trials; we report
  best and median.
- **Faithful drop-in**: every run first checks the GPU output equals the CPU
  preprocessor output on a sample (`PASS`/`FAIL`).

Each operator is benchmarked on its own and can run by itself.

## Operators

- `bench_ordinal_encoder.py` — `OrdinalEncoder` vs `GpuOrdinalEncoder`
- `bench_simple_imputer.py` — `SimpleImputer` vs `GpuSimpleImputer`
  (`most_frequent` headline, `mean` secondary), synthetic data
- `bench_imputer_yambda.py` — `SimpleImputer` vs `GpuSimpleImputer` on the **real**
  `yandex/yambda` `flat/500m` dataset (`yambda.py` downloads it from the Hugging
  Face Hub). See `RESULTS_yambda.md`.

## Real-data benchmark (yandex/yambda)

`bench_imputer_yambda.py` runs the imputer on the real recommender event log and
reports four things the synthetic bench does not:

- **E2E (cold-inclusive)** CPU vs GPU, split into `fit` / `transform` /
  `fit_transform`. Every GPU run builds and tears down a fresh actor pool, so
  *each* trial includes cold start (there is no resident pool).
- **Operator cold-start floors** for `fit` and `transform` separately (1-block
  slice).
- **Device microbench**: `cudf.from_arrow -> value_counts/sum_count | fillna ->
  to_arrow` replayed on real batches, **each phase in its own subprocess** so
  both the fit and transform cold starts pay a true CUDA init; steady batches
  give the H2D / compute / D2H split. This is a *device floor* and excludes Ray
  scheduling, the object store, actor lifecycle, and block formation.
- **Batch-size sweep** for `fit` and `transform`, independently.

Inputs (downloaded to `gpu_preproc_bench/data/` by `yambda.py`; needs network):

- `flat/500m/multi_event.parquet` (~480M rows) — main input. `played_ratio_pct`
  and `track_length_seconds` are **naturally null** for non-listen events
  (~2.86%). `item_id`/`event_type` get injected nulls so `most_frequent`
  transform has something to fill.
- `flat/500m/likes.parquet` (~9M rows, no native nulls) — targeted, injected.

Requires the extra deps in `requirements.txt` (`huggingface_hub`, `hf_xet`):

```bash
.venv/bin/pip install -r gpu_preproc_bench/requirements.txt
RAY_enable_open_telemetry=0 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
    .venv/bin/python gpu_preproc_bench/bench_imputer_yambda.py \
        --dataset multi_event --rows 50000000 --gpus 16 --gpu-full-fit
```

## Running

```bash
# 8 of the 16 local GPUs, ~2M-row batches
RAY_DATA_GPU_PREPROC_NUM_GPUS=8 RAY_DATA_GPU_PREPROC_BATCH_SIZE=2000000 \
    .venv/bin/python gpu_preproc_bench/bench_ordinal_encoder.py --rows 20000000 --blocks 32

RAY_DATA_GPU_PREPROC_NUM_GPUS=8 RAY_DATA_GPU_PREPROC_BATCH_SIZE=2000000 \
    .venv/bin/python gpu_preproc_bench/bench_simple_imputer.py --rows 20000000 --blocks 32
```

Environment knobs (read by the operators, see
`python/ray/data/preprocessors/_gpu.py`):

- `RAY_DATA_GPU_PREPROC_NUM_GPUS` — number of one-GPU workers (actor pool size).
- `RAY_DATA_GPU_PREPROC_BATCH_SIZE` — per-worker batch size (required because
  `map_batches` mandates a batch size whenever `num_gpus` is set).

## What to expect

The win is concentrated in **`fit`** over string columns, where the CPU path
counts values with Python `Counter` objects while cuDF counts on the GPU:

- `GpuOrdinalEncoder.fit_transform` is dominated by the vocabulary build (fit);
  the per-row encode (transform) is a host-staged wash because the CPU encoder's
  Arrow `pc.index_in` is already fast C++.
- `GpuSimpleImputer(strategy="most_frequent")` wins for the same reason; the
  `mean` strategy is transfer-bound and roughly break-even.

See `RESULTS.md` for measured numbers and the bottleneck analysis.
