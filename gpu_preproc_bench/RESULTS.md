# GPU preprocessor results: `GpuOrdinalEncoder` and `GpuSimpleImputer`

Goal: show that the experimental, opt-in GPU preprocessors are faithful drop-ins
for their CPU counterparts and deliver a real end-to-end win, host-staged
(RAM in -> RAM out), the same way `gpu_sort_bench` did for `ds.sort(gpu=True)`.

## Environment
- 16x Tesla V100-SXM3-32GB, 96 CPU cores, ~1.5 TB RAM.
- Ray 3.0.0.dev (editable), cuDF/RMM 26.02, CUDA 12.x, Python 3.10.
- 8 of 16 GPUs used (`RAY_DATA_GPU_PREPROC_NUM_GPUS=8`), batch size 2,000,000.

## Methodology (mirrors the sort)
- **RAM in -> RAM out**: input materialized in the object store; we time
  `preprocessor.fit_transform(ds).materialize()`, warm, best-of-2 (median shown
  too). Each GPU op pulls Arrow blocks, moves only its input columns to a GPU as
  cuDF, computes, and writes Arrow back.
- **Faithful drop-in**: every run first verifies the GPU output equals the CPU
  preprocessor output on a 200k sample (`PASS`).
- Dataset: a recommender-style event table, 20,000,000 rows x 32 blocks
  (~1.2 GiB). High-cardinality string ids (`item_id` ~200k, `user_id` ~50k,
  `last_item_id` ~200k with nulls), low-cardinality categoricals
  (`event_type`/`device_type`/`country`), and a numeric `price` with nulls.

## Results

### `GpuOrdinalEncoder` (encode item_id, user_id, event_type, device_type, country)
Correctness vs CPU `OrdinalEncoder`: **PASS**.

| stage | CPU | GPU | speedup |
|---|---:|---:|---:|
| **fit_transform** (best) | 244.00 s | **9.83 s** | **24.8x** |
| fit_transform (median) | 246.15 s | 9.88 s | 24.9x |
| fit (vocabulary) | 240.14 s | 5.14 s | **46.7x** |
| transform (encode) | 1.64 s | 5.39 s | 0.31x |

### `GpuSimpleImputer`
Correctness vs CPU `SimpleImputer`: **PASS**.

| case | CPU | GPU | speedup |
|---|---:|---:|---:|
| **most_frequent, high-card** (`last_item_id`) | 190.93 s | **7.89 s** | **24.2x** |
| most_frequent, low-card (`event_type`/`device`/`country`) | 2.47 s | 8.60 s | 0.29x |
| mean (`price`) | 2.57 s | 8.10 s | 0.32x |

## Where the win comes from (and where it doesn't)

The win is concentrated in **`fit` over high-cardinality string columns**. The
CPU path computes the vocabulary / mode by pulling per-block `value_counts` to
the driver and merging them with Python `Counter` objects over object arrays --
~240 s for the encoder's five columns, ~191 s for the imputer's one high-card
column. cuDF does the same counting on the GPU; the per-block partials are tiny
and merged with pyarrow on the driver. That is the entire ~24x.

The parts that **don't** win are honest and expected, and match what the sort
study found (host-staged work is a transfer/overhead contest):

- **`transform` is a wash** (encoder 0.31x). The CPU encode is a vectorized
  Arrow `pc.index_in` + `pc.take` in C++ that finishes in ~1.6 s; the GPU pays
  actor-pool startup + H2D/D2H for a cheaper-than-fit op. Encoding is genuinely
  CPU-cheap.
- **Low-card imputation is CPU territory** (0.29x / 0.32x). With 4-8 distinct
  values (or a single numeric mean) the CPU `value_counts` / `mean` is ~2.5 s;
  the GPU's fixed per-pass startup can't be amortized over so little work.

So the rule mirrors the sort: **GPU wins the work the CPU is bad at (counting
many distinct strings); it does not win trivial vectorized work.** The benchmark
reports both honestly.

## Bottleneck analysis (what we fixed)

Following the sort's host-staging playbook, we phase-instrumented the operators
and removed the dominant overheads:

1. **Per-column fit passes -> one fused pass (the big one).** The fit originally
   ran one `map_batches` pass *per column*, re-paying GPU actor-pool / CUDA
   startup each time. Fusing same-dtype columns into a single pass (emit the
   per-block counts in long `(__col, value, count)` form, split per column on the
   driver) cut the encoder's GPU fit substantially:

   | encoder GPU fit, 10M-row probe | before | after |
   |---|---:|---:|
   | fit (3 string columns) | 10.95 s | **3.81 s** |
   | fit_transform | 13.91 s | **6.95 s** |

   This roughly doubled the end-to-end speedup (~7x -> ~14x at 10M; ~25x at 20M
   with five columns).

2. **Only the operator's columns cross the bus.** Transforms convert just the
   input columns to cuDF and re-attach outputs to the original Arrow table, so
   payload columns (`price`, ids, timestamps) never round-trip host<->device.

3. **One CUDA context per worker, not per batch.** Transforms run through a
   stateful `_GpuBatchActor` so the context + any device-resident state (the
   fitted vocabulary) are built once per worker.

The remaining GPU floor (~5-8 s) is fixed actor-pool startup for the two passes
(fit + transform) -- the same kind of fixed cost the sort sees; it amortizes as
data grows and is why low-card cases (little work) stay CPU territory.

## Device-resident projection (not implemented here)

These operators are host-staged: each pays H2D + D2H. As the sort study
quantified (`gpu_sort_bench/research/MERGE_JUSTIFICATION.md`, Section 2), a
device-resident execution mode that keeps blocks in VRAM across chained
operators (sort -> impute -> encode) would pay the transfers once and amortize
them, converging toward the GPU-only regime (the sort measured up to ~11x over
per-op host staging). That is the natural follow-up; v1 deliberately starts and
ends in RAM, like the first GPU sort.

## Reproduce

```bash
RAY_DATA_GPU_PREPROC_NUM_GPUS=8 RAY_DATA_GPU_PREPROC_BATCH_SIZE=2000000 \
    .venv/bin/python gpu_preproc_bench/bench_ordinal_encoder.py --rows 20000000 --blocks 32
RAY_DATA_GPU_PREPROC_NUM_GPUS=8 RAY_DATA_GPU_PREPROC_BATCH_SIZE=2000000 \
    .venv/bin/python gpu_preproc_bench/bench_simple_imputer.py --rows 20000000 --blocks 32
```

## Scope
Two experimental, opt-in GPU preprocessors, both faithful drop-ins (CPU remains
the default; they fall back to CPU with no GPU). The win is on high-cardinality
string `fit`; transform and low-card stats are host-staged CPU territory, which
a future device-resident mode would address.
