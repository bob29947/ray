# CriteoPrivateAd CPU preprocessing baseline (full dataset, days 1–30)

A full **CPU-only** preprocessing pass over the **entire** CriteoPrivateAd
ad-ranking dataset (all 30 `day_int` partitions, **103,862,032 rows**), producing
one training-ready, globally-sorted parquet output and a per-stage timing
baseline. This is the reference the later **mixed CPU/GPU** pipeline (GPU sort +
GPU encode + GPU scale) is measured against, so the three GPU-acceleration-target
stages are timed in isolation. Companion to `RESULTS.md` / `RESULTS_yambda.md`
(the per-operator GPU drop-in studies).

Produced by [`bench_criteo_cpu_baseline.py`](bench_criteo_cpu_baseline.py) +
[`criteo.py`](criteo.py).

## Command used
```bash
# medium sanity pass first (days 1–5)
RAY_enable_open_telemetry=0 .venv/bin/python \
    gpu_preproc_bench/bench_criteo_cpu_baseline.py \
    --days 1-5 \
    --out gpu_preproc_bench/data/criteo_days1_5_cpu_baseline

# full dataset (days 1–30)  -> the result documented here
RAY_enable_open_telemetry=0 .venv/bin/python \
    gpu_preproc_bench/bench_criteo_cpu_baseline.py \
    --days 1-30 \
    --out gpu_preproc_bench/data/criteo_days1_30_cpu_baseline
```
`--days` accepts `1`, `1-5`, `1-25`, `1-30`, or `all`. Block count is **left to
Ray** (no forced repartition).

## Environment
- 96 CPU cores, ~1.5 TiB RAM, ~756 GiB `/dev/shm`. **CPU only** — no GPU is used
  for this baseline.
- Ray 3.0.0.dev0 (editable, this worktree), pyarrow 17.0.0, pandas 2.3.3,
  Python 3.10. (cuDF is installed but unused here; it is the GPU stack the
  follow-up pipeline will use.)
- One end-to-end pass; each stage is `.materialize()`-d (RAM in → RAM out) and
  timed on its own. `read` includes gzip-parquet decode from disk; `write`
  includes parquet encode to disk. `preserve_order=True` so the sorted row order
  survives impute/encode/scale/write.
- **No spilling.** The default Ray object store is capped (~200 GiB) and the
  100M-row sort shuffle overflowed it, spilling to `/tmp` — which only has ~120 GB
  free — and the run died with `OutOfDiskError`. Fix: the script now sizes the
  object store to **~85% of `/dev/shm` (642 GiB, in RAM)** via
  `ray.init(object_store_memory=…)`, so read + prep + the sort's map/reduce
  intermediates all stay in RAM. The full run completed with **zero spilled
  bytes** (see *Spilling / memory / Ray warnings*). Override with
  `--object-store-gb`.

## Input (days 1–30)
- `/bobbwang/datasets/CriteoPrivateAd/data/day_int={1..30}/` — 288 gzip-parquet
  shards (Hive-partitioned by `day_int`), **103,862,032 rows**, ~28 GB on disk,
  150 in-file columns.
- `day_int` is **not** stored inside the parquet files; it comes from the Hive
  partition path. Ray exposes it as a **string** (`"1"`, `"2"`, …), so the prep
  stage casts it to `int64` — otherwise `"10" < "2"` lexicographically would
  corrupt the day ordering. (If a future Ray ever fails to surface it, the reader
  falls back to reading each folder and tagging its rows.)
- One row = one ad impression. Labels: `is_clicked`, `is_click_landed`,
  `is_visit` (dense binary) and `nb_sales` (mostly null = no attributed sale).

## Pipeline (sort before impute; one saved dataset)
```
read → prep/drop-leakage/derive-labels → sort → impute(+indicators) → encode → scale → write
```
The order-setting **sort** runs *before* impute/encode/scale (all row-order-
independent). `prep` first prunes 151 → 67 columns (163.7 → 51.3 GiB) — dropping
the 80 `features_not_available_*`, the 3 delay arrays, `id`, the all-null column,
and the raw list column — so the sort (a GPU target) moves the **minimal** pruned
row. **No sorted-only intermediate is written:** the sorted rows flow straight
into impute/encode/scale and the *only* saved dataset is the final processed
parquet (after scale) plus `manifest.json`.

## Metadata / sort keys (kept raw, never transformed)
`user_id` (`string`), `day_int` (`int64`) and `display_order` (`int32`) are
carried through every stage into the saved output as **metadata / sort-key**
columns. They are **never encoded or scaled** (`user_id` would only become a
feature in a separate high-cardinality encoder stress mode). Keeping them raw is
what makes the saved parquet directly verifiable for global sortedness.
- Single-day sort key: `[user_id, display_order]`.
- **Multi-day sort key: `[user_id, day_int, display_order]`** — `day_int` in the
  middle keeps each user's days contiguous and in order; it is *not* treated as a
  constant. (Note: this differs from the earlier day-1-only baseline, which
  scaled `display_order` as a numeric feature; here `display_order` is a raw sort
  key, so the saved key is exactly the sort order.)

## ML decision log (what each stage does and why)
- **Pre-display features only.** Drop the 80 `features_not_available_*` (the
  dataset's cross-device bucket, *not available at inference*), the 3
  `*_delay_after_display_array` columns (post-display conversion delays = label
  leakage), and `id`.
- **Targets.** `is_clicked`, `is_click_landed`, `is_visit` cast to int8; sales
  handled as **target derivation, not feature imputation**:
  `sales_count = coalesce(nb_sales, 0)`, `is_sale = nb_sales > 0`. Labels are
  never fed through the feature transforms.
- **Missing-not-at-random.** Add a 0/1 `<col>_isnull` indicator for each numeric
  column whose null fraction (aggregated across all selected days) exceeds the
  threshold (default 1% → **30 indicators**), then mean-impute to keep the tensor
  dense. Indicators are kept **0/1 and never scaled**.
- **Impute (CPU, deliberately not `GpuSimpleImputer`).** Numeric `mean`;
  categorical `most_frequent` only for columns that actually have nulls. Impute
  must precede encode (OrdinalEncoder requires non-null input).
- **Encode: `OrdinalEncoder`** (integer codes for embedding tables), not one-hot.
  The 15 categoricals are `campaign_id`, `publisher_id`, and 13 int hash columns.
- **Scale: `StandardScaler`** (z-score) on the **44 numeric features only** —
  never on the category codes, the 0/1 indicators, or the sort keys.

## Output
- `data/criteo_days1_30_cpu_baseline/` (gitignored): **1,054 parquet files,
  103,862,032 rows, 97 columns, ~16 GB on disk**, globally sorted by
  `[user_id, day_int, display_order]`.
- `data/criteo_days1_30_cpu_baseline/manifest.json`: selected days, row count,
  sort key, feature columns, target columns, **metadata/sort-key columns**,
  dropped columns + counts, fitted imputer stats, encoder vocab sizes, scaler
  means/stds, per-stage timings, the GPU-target subtotal, and the verification
  result.

### Resulting schema (97 columns written)
| group | count | dtype | notes |
|---|---:|---|---|
| metadata / sort keys | 3 | `user_id` string, `day_int` int64, `display_order` int32 | kept raw; never encoded/scaled |
| binary labels | 3 | int8 | `is_clicked`, `is_click_landed`, `is_visit` |
| sales targets | 2 | int64 / int8 | `sales_count`, `is_sale` |
| categorical (encoded) | 15 | int64 | ordinal codes for embeddings |
| numeric (scaled) | 44 | double | 43 raw doubles + 1 list-len (`features_ctx_not_constrained_2_len`) |
| missing indicators | 30 | int8 | `<col>_isnull`, unscaled |

Dropped: `id` (1), 3 delay arrays, `features_not_available_*` (80), all-null
`features_kv_bits_constrained_5` (1). (`user_id`/`day_int`/`display_order` are
**kept**, as metadata, not dropped.)

## CPU baseline timings (full days 1–30, Ray-default 1,054 blocks, one pass)
| stage | sec | Mrows/s | cols (in→out) | RAM GiB | GPU target |
|---|---:|---:|---:|---:|:---:|
| read | 50.45 | 2.06 | 0→151 | 163.72 | |
| prep | 17.34 | 5.99 | 151→67 | 51.26 | |
| **sort** | **865.22** | 0.12 | 67→67 | 51.19 | **yes** |
| impute | 165.37 | 0.63 | 67→97 | 54.45 | |
| **encode** | **723.19** | 0.14 | 97→97 | 53.97 | **yes** |
| **scale** | **39.47** | 2.63 | 97→97 | 54.89 | **yes** |
| write | 15.97 | 6.51 | 97→97 | 54.89 | |
| **TOTAL** | **1876.99** | | | | |

- **GPU-target subtotal (sort + encode + scale): 1,627.88 s = 86.7% of the run.**
  This is the slice the mixed CPU/GPU pipeline will attack.
- **`sort` (865 s, 46% of the run)** is the global shuffle of 103.9M rows over the
  high-cardinality `user_id` string key — a strong GPU-sort target. Ray's default
  ~1,054 blocks make the CPU pull-based shuffle an all-to-all of ~1.1M
  intermediate objects, which is exactly the kind of work a GPU sort collapses.
- **`encode` (723 s, 39%)** is the high-cardinality vocabulary build:
  `features_ctx_not_constrained_5` has **12,162,598** distinct values, `_4`
  **8,844,728** (sum of all 15 vocabularies = **21,828,806** embedding rows).
  This is the work the CPU is bad at and the GPU encoder targets.
- `scale` is cheap on CPU (39 s, vectorized Arrow); on GPU it would be
  transfer-bound. `impute` (165 s, CPU only by design) is the missing-indicator
  pass + mean/most_frequent. `prep` prunes 163.7 → 51.3 GiB before the sort.

## Saved-output sortedness verification (scalable; PASS)
The old full-pandas sortedness check does not scale to 100M keys, so the saved
parquet is verified with a distributed, block-wise pass that never loads all keys
locally:
1. re-read the written parquet with Ray, selecting **only the sort-key columns**
   (`include_paths=True`);
2. reduce each block (distributed) to a tiny summary: row count, whether the
   block is internally nondecreasing on the full key, and its first/last key +
   source-file path;
3. put the blocks in **canonical filename order** and assert every block is
   internally sorted and `block_i.last ≤ block_{i+1}.first` for all adjacent
   blocks;
4. count rows during the pass and assert the total equals the expected
   103,862,032.

```
globally_sorted = True   (in_block_sorted=True, boundaries_ok=True)
rows_counted    = 103,862,032 / 103,862,032   (match=True)
blocks/files    = 1,054      ordered_by = source_file_path
method          = blockwise-readback-path-ordered
```
> **Why order blocks by source-file path?** Ray's parallel parquet reader
> bin-packs files into size-balanced tasks and does **not** emit blocks in
> filename order, even with `preserve_order=True` (it happened to line up at 228
> files but not at 1,054). The output files are named `…_<task_index:06>-….parquet`
> in global-sort order, so ordering the per-block summaries by their source path
> reconstructs the exact order a consumer reading the output sees — making the
> boundary check correct and independent of how Ray schedules the read.

## Sanity checks (all PASS)
`row_count_preserved` (103,862,032), `final_parquet_reloads`,
`saved_output_globally_sorted`, `saved_output_row_count_matches`,
`no_nulls_in_features`, `encoded_cols_integer`, `scaled_cols_float`,
`indicators_binary`, `targets_present`, and
`metadata_keys_present_not_features` (`user_id`/`day_int`/`display_order` are
present as metadata/sort keys and are **not** in the encoded/scaled/indicator
feature sets).

## Spilling / memory pressure / Ray warnings
- **No spilling in the final run** (0 spilled bytes; `/tmp` untouched). Peak
  resident data: the 163.7 GiB raw read, then ~51–55 GiB per pruned stage; the
  sort's pull-based shuffle adds map+reduce intermediates on top. All of it fit
  inside the 642 GiB in-RAM object store.
- **First full-run attempt failed with `OutOfDiskError`** using the default
  (~200 GiB) object store: the sort shuffle spilled to `/tmp` (only ~120 GB free)
  and filled the disk. Resolved by sizing the object store to ~85% of `/dev/shm`
  (642 GiB) so nothing spills. There is no large local scratch disk, so keeping
  the pipeline in RAM is required at this scale.
- **Benign Ray warning** during the imputer/encoder fit:
  `Failed to convert column '…' into pyarrow array (ArrowConversionError);
  falling back to serialize as pickled python objects` — emitted by the
  `get_pd_value_counts` aggregation for a few high-cardinality columns. It is a
  slower serialization fallback, not an error; the fitted vocabularies and
  imputer stats are correct (verified in `manifest.json`).
- Cosmetic: a `DeprecationWarning` for `read_parquet(columns=…)` (kept because it
  guarantees the verification reads only the 3 sort-key columns from disk), and
  network/`http_connect` proxy log lines from the sandbox (unrelated to the run).

## Medium sanity pass (days 1–5)
Run first, per the plan: **22,138,676 rows**, 228 output files, all sanity checks
pass and the saved-output sortedness verification passes
(`globally_sorted=True`, rows match). End-to-end ≈ 176 s, GPU-target subtotal
≈ 111 s (63%). Output at `data/criteo_days1_5_cpu_baseline/`.

## Reproduce
```bash
# full dataset (writes data/criteo_days1_30_cpu_baseline/ + manifest.json)
RAY_enable_open_telemetry=0 .venv/bin/python \
    gpu_preproc_bench/bench_criteo_cpu_baseline.py --days 1-30 \
    --out gpu_preproc_bench/data/criteo_days1_30_cpu_baseline

# quick smoke run (RAM → RAM, no files written)
RAY_enable_open_telemetry=0 .venv/bin/python \
    gpu_preproc_bench/bench_criteo_cpu_baseline.py --days 1-5 --rows 200000 --no-write
```
Flags: `--days` (`1` | `1-5` | `1-30` | `all`), `--object-store-gb` (default:
~85% of /dev/shm, to avoid spilling), `--blocks` (default: let Ray choose),
`--rows` (cap; 0 = full), `--null-indicator-threshold` (default 0.01), `--out`,
`--no-write`.

## What the mixed CPU/GPU pipeline will compare against
- **CPU baseline (this doc):** CPU sort + CPU OrdinalEncoder + CPU StandardScaler
  (with cheap CPU prep/impute around them) = **1,877 s** end-to-end on 103.9M
  rows, **1,628 s (86.7%)** in the three GPU-target stages.
- **Mixed GPU target:** GPU sort + CPU cleanup/impute + GPU encode + GPU scale.
  The win concentrates in **sort** (global shuffle over the high-cardinality
  `user_id` key) and **encode** (the 21.8M-row vocabulary build) — the same "GPU
  wins the work the CPU is bad at" pattern documented in `RESULTS.md` /
  `RESULTS_yambda.md`; `scale` is transfer-bound and closer to break-even.
