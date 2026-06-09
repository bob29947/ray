# GpuSimpleImputer on real data: yandex/yambda (flat/500m)

End-to-end CPU vs GPU imputation on the real **yandex/yambda** recommender event
log, plus cold-start, host<->device transfer, and batch-size characterization.
Companion to `RESULTS.md` (synthetic) using the real dataset requested for the
imputer's own PR worktree.

## Environment
- 16x Tesla V100-SXM3-32GB, cuDF/RMM + CUDA 12.x, Ray (editable, this worktree).
- `RAY_DATA_GPU_PREPROC_NUM_GPUS=16`, `RAY_enable_open_telemetry=0`.
- E2E numbers are **cold-inclusive** best-of-2 (each run builds + tears down a
  fresh GPU actor pool — there is no resident pool). Device-microbench numbers
  are a single-GPU device floor (separate process per phase).

## Dataset status (is there anything to impute?)

Measured from parquet row-group statistics (no scan):

| file | rows | columns with nulls |
|---|---:|---|
| `flat/500m/multi_event.parquet` | 480,255,564 | `played_ratio_pct` **2.86%**, `track_length_seconds` **2.86%** (null for non-listen events) |
| `flat/500m/likes.parquet` | 9,033,960 | none (4 non-nullable uint columns) |

So `multi_event` has **real** missing values in the playback fields; the id /
flag / `event_type` columns are dense, so we inject ~5% nulls there (deterministic,
seeded, materialized once, reused for CPU and GPU) to exercise `most_frequent`
transform. `event_type` is a `dictionary<string>` (5 values) decoded to plain
strings on load.

## Headline

- **GPU wins the work the CPU is bad at: counting many distinct strings.**
  `most_frequent` over a high-cardinality **string** column (item_id cast to
  string, 10M rows): **CPU fit 38.1s -> GPU fit 2.0s = 19.2x**; fit_transform
  8.65x.
- **On yambda's native columns the GPU is transfer/startup-bound** (so the CPU
  is faster at these scales): they are numeric (`item_id` uint32, playback
  ints) or low-cardinality (`event_type`, `is_organic`), where pandas
  `value_counts` / `mean` are already fast vectorized C.
- **Recommended batch size: 16,000,000 rows for both fit and transform** (the
  curve plateaus from ~8M; narrow columns make large batches cheap).

## E2E CPU vs GPU, multi_event @ 50,000,000 rows (best of 2, seconds)

| case | CPU fit | CPU tf | CPU f+t | GPU fit | GPU tf | GPU f+t | f+t x |
|---|---:|---:|---:|---:|---:|---:|---:|
| mean `played_ratio_pct` (natural nulls) | 1.17 | 1.51 | 2.50 | 3.29 | 3.14 | 7.36 | 0.34 |
| mean `track_length_seconds` (natural nulls) | 1.16 | 1.54 | 2.54 | 3.64 | 3.50 | 7.39 | 0.34 |
| most_frequent `item_id` (high-card **uint32**) | 3.78 | 1.56 | 5.08 | 4.19 | 3.49 | 7.78 | 0.65 |
| most_frequent `event_type` (low-card string) | 0.90 | 1.62 | 2.28 | 3.81 | 3.50 | 7.55 | 0.30 |

## High-cardinality STRING (item_id cast to string) @ 10,000,000 rows (best of 1)

| case | CPU fit | CPU f+t | GPU fit | GPU f+t | fit x | f+t x |
|---|---:|---:|---:|---:|---:|---:|
| most_frequent `item_id_str` | 38.08 | 37.74 | 1.99 | 4.36 | **19.18** | **8.65** |

The only difference from the `item_id` row above is the **dtype**: the GPU path
is identical, but the CPU `most_frequent` path merges per-block `value_counts`
with Python `Counter` over string objects, which is ~20x slower than cuDF
counting on device. This is the same effect the synthetic `RESULTS.md` reports.

## Scale: GPU fit on the FULL dataset

| op | rows | GPU |
|---|---:|---:|
| most_frequent `item_id` fit | 480,255,564 | 18.17 s |

The fixed actor-pool start (~2 s) amortizes; fit scales smoothly to the full
480M rows across 16 GPUs.

## Cold start (fit and transform, separately)

**Operator level** (single 1M-row block; actor pool + one pass):

| pass | cold-start floor |
|---|---:|
| fit (value_counts pass) | 2.17 s |
| transform (fillna pass) | 2.40 s |

**Device level** (microbench, first batch vs steady, ms; bs = 2,000,000):

| phase / strategy | cold total | cold H2D | steady total | steady H2D | steady compute | steady D2H |
|---|---:|---:|---:|---:|---:|---:|
| fit / mean `played_ratio_pct` | 305 | 263 | 3.9 | 1.8 | 2.1 | 0.0 |
| transform / mean `played_ratio_pct` | 342 | 277 | 8.1 | 1.5 | 2.8 | 3.4 |
| fit / most_frequent `item_id` | 364 | 268 | 10.3 | 2.4 | 7.0 | 0.7 |
| transform / most_frequent `item_id` | 293 | 264 | 5.5 | 2.4 | 1.5 | 1.7 |

The first batch is ~300-360 ms (dominated by the first H2D, which pays CUDA
context creation + first-kernel JIT); steady batches are ~4-10 ms. Each phase is
timed in its **own process**, so both the fit and transform cold starts are real.

## Host<->device transfer

H2D / compute / D2H are broken out above. On these narrow columns (4-32 MB per
batch) the transfers are overhead-bound: steady H2D is ~3-4 GB/s and grows with
batch size (2M-row item_id batch ~3.4 GB/s; 8M-row ~4.1 GB/s). Payload columns
never cross the bus (only the imputed column is moved to cuDF and the result
re-attached to the host Arrow table).

## Batch-size sweep (multi_event @ 50M, most_frequent item_id)

| batch size (rows) | fit-only (s) | transform-only (s) | transform Mrows/s |
|---:|---:|---:|---:|
| 1,048,576 | 5.62 | 4.44 | 11.3 |
| 2,000,000 | 5.14 | 3.74 | 13.4 |
| 4,000,000 | 5.11 | 4.15 | 12.0 |
| 8,000,000 | 4.23 | 3.45 | 14.5 |
| 16,000,000 | 3.95 | 3.40 | 14.7 |

**Recommended: `fit_batch_size = transform_batch_size = 16,000,000`** (set fit
size via `RAY_DATA_GPU_PREPROC_BATCH_SIZE`, transform size via
`transform(batch_size=)` / `fit_transform(transform_batch_size=)`).

## Where the win comes from (and where it doesn't)

Same rule as the sort and the synthetic preprocessor study: **the GPU wins the
work the CPU is bad at** — here, counting a high-cardinality **string** column
(~19x). It does **not** win numeric `mean`, low-cardinality `most_frequent`, or
high-cardinality **integer** `most_frequent`, because pandas does those with fast
vectorized C and the host-staged GPU path then just pays its fixed actor-pool
start + H2D/D2H. yambda's native schema is mostly that CPU-friendly kind, so the
honest end-to-end result on the raw columns is "CPU territory"; the GPU value
shows up on string categoricals and as the dataset scales.

This operator is host-staged (RAM in -> GPU -> RAM out), so every op pays H2D +
D2H. A device-resident mode that keeps blocks in VRAM across chained operators
(impute -> encode -> sort) would amortize the transfers, as quantified for the
sort in `../gpu_sort_bench/research/MERGE_JUSTIFICATION.md`. That is the natural
follow-up; v1 starts and ends in RAM.

## Reproduce

```bash
.venv/bin/pip install -r gpu_preproc_bench/requirements.txt   # huggingface_hub, hf_xet

# Main: native columns + sweep + cold-start + device microbench + full-480M fit
RAY_enable_open_telemetry=0 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
    .venv/bin/python gpu_preproc_bench/bench_imputer_yambda.py \
        --dataset multi_event --rows 50000000 --gpus 16 --gpu-full-fit

# High-cardinality string headline only (item_id cast to string)
RAY_enable_open_telemetry=0 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
    .venv/bin/python gpu_preproc_bench/bench_imputer_yambda.py \
        --dataset multi_event --rows 10000000 --trials 1 \
        --no-e2e --no-coldstart --no-sweep --no-micro

# Targeted likes stress (injected nulls)
RAY_enable_open_telemetry=0 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
    .venv/bin/python gpu_preproc_bench/bench_imputer_yambda.py --dataset likes --rows 9000000
```
