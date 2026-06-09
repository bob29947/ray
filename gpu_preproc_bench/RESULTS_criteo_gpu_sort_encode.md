# CriteoPrivateAd 30-day: GPU sort + GPU OrdinalEncoder (CPU everything else)

Status: **MEASURED** (16x Tesla V100-SXM3-32GB, single node). The OOM stress demo
was skipped by request; its analysis is summarized from the engine design at the
end.

## Goal & scope
Take the CPU baseline pipeline on the full 30-day CriteoPrivateAd dataset and
swap **exactly two** operators to GPU, changing nothing else:

- `CPU ds.sort(...)`  ->  `ds.sort(..., backend="gpu")`  (cuDF + rapidsmpf)
- `CPU OrdinalEncoder`  ->  `GpuOrdinalEncoder`  (cuDF, host-staged)

**Scale stays on CPU** (`StandardScaler`). This run is **not** "GPU scale"; the
GPU-accelerated stage set is `{sort, encode}` only. Verified in the manifest:
`gpu_stages = ["encode", "sort"]`, `scale_backend = "cpu"`.

Sort and encode are **sized independently** (different optimal block sizes).

## Headline result
- **End-to-end: 1876.99 s (CPU) -> 303.52 s (GPU) = 6.18x**, output verified
  globally sorted, all sanity checks PASS, encoder vocab parity vs CPU = PASS.
- **Pure replaced operators: CPU sort+encode 1588.41 s -> GPU 108.09 s = 14.69x.**
- **Honest pipeline replacement (incl. both repartitions): 1588.41 s -> 149.98 s
  = 10.59x.**
- GPU **sort 28.13x** (865.22 -> 30.76 s), GPU **encode 9.35x** (723.19 -> 77.34 s).
- 16-GPU sort fit comfortably: peak 9.57 GiB/rank vs 25.39 GiB pool cap
  (**15.82 GiB headroom**); **Ray spilled = 0, GPU/RMM device spill disabled,
  no OOM**.

## Comparability contract (held)
Same days `1-30`; same row count **103,862,032**; same prep/drop/leakage and
target derivation (`prep_batch`/`add_indicators` reused verbatim); same CPU
imputers and CPU `StandardScaler`; same output schema/roles; same raw metadata
keys `user_id, day_int, display_order`; same final sort key
`[user_id, day_int, display_order]`; same saved-output sortedness verification;
same object-store policy (642 GiB, ~85% of /dev/shm). Encoded values come from
`GpuOrdinalEncoder` and pass parity: **all 15 categorical vocab sizes match the
CPU baseline manifest**.

## 1. Encoder batch-size scan (independent of sort sizing)
Built once: `read -> prep -> GPU sort -> post-sort repartition -> CPU impute`
(103,862,032 rows). Bigger batch = fewer/larger map batches = less per-batch
overhead = faster, at the cost of more peak GPU memory. Monotonic; **8,000,000
is the fastest in the requested range** and was used for the main run.

| batch_size | fit (s) | transform (s) | total (s) | Mrows/s | out_blocks | peak GPU (GiB) | result |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 250,000   | 264.28 | 165.61 | 429.89 | 0.24 | 703 | 0.98 | ok |
| 500,000   | 160.64 |  93.88 | 254.51 | 0.41 | 527 | 0.94 | ok |
| 1,000,000 |  98.94 |  58.17 | 157.11 | 0.66 | 445 | 1.02 | ok |
| 2,000,000 |  64.35 |  39.21 | 103.56 | 1.00 | 434 | 1.40 | ok |
| 4,000,000 |  51.25 |  38.16 |  89.41 | 1.16 | 428 | 1.73 | ok |
| **8,000,000** | **37.22** | **43.72** | **80.94** | **1.28** | 437 | 2.63 | **ok (fastest)** |

Note: every batch size produces ~428-703 output blocks (< target/2 = 527), so the
main run always triggers the conditional post-encode repartition.

## 2. CPU vs GPU per-stage side-by-side (RAM -> RAM)
CPU column = baseline manifest; GPU column = this run. Repartition rows are
GPU-only overhead with no CPU counterpart and are **timed separately, never
folded into sort/encode**.

| stage | CPU (s) | GPU (s) | speedup | note |
|---|---:|---:|---:|:--|
| read                    |   50.45 |  47.62 |  1.06x | CPU (both) |
| prep                    |   17.34 |  15.43 |  1.12x | CPU (both) |
| **sort**                |  865.22 |  30.76 | **28.13x** | **GPU** |
| post_sort_repartition   |     n/a |  23.87 |    n/a | GPU-overhead [16->1054 blocks] |
| impute                  |  165.37 |  56.41 |  2.93x | CPU (both) -- see note |
| **encode**              |  723.19 |  77.34 | **9.35x** | **GPU** |
| post_encode_repartition |     n/a |  18.02 |    n/a | GPU-overhead [437->1054 blocks] |
| scale                   |   39.47 |  22.93 |  1.72x | CPU (both, NOT GPU scale) |
| write                   |   15.97 |  11.15 |  1.43x | CPU (both) |
| **TOTAL**               | **1876.99** | **303.52** | **6.18x** | end-to-end |

Honesty note on CPU stages: `read/prep/impute/scale/write` are CPU in **both**
runs, so their differences are **not** GPU effects -- they are run-to-run /
block-layout / scheduling variance. The largest, `impute` (165.37 -> 56.41), is
incidental: the GPU run feeds impute a freshly even 1054-block layout (from the
post-sort repartition), which parallelizes the CPU imputer better than the
baseline's post-CPU-sort layout. The durable GPU contribution is isolated in the
replaced-stage subtotals below.

## 3. Replaced-stage subtotals (the honest accounting)
1. **PURE replaced operators:** CPU `sort+encode` **1588.41 s** vs GPU
   `sort+encode` **108.09 s** = **14.69x**.
2. **HONEST pipeline replacement (incl. repartitions):** CPU `sort+encode`
   **1588.41 s** vs GPU `sort + post_sort_repartition + encode +
   post_encode_repartition` **149.98 s** = **10.59x**
   (post_sort_repartition = 23.87 s, post_encode_repartition = 18.02 s --
   shown, not hidden).
3. **Total end-to-end:** **1876.99 s -> 303.52 s = 6.18x**.

## 4. GPU sort block/partition sizing + VRAM pressure (biggest dataset)
| quantity | value |
|---|---:|
| input rows after prep | 103,862,032 |
| input bytes after prep | **51.26 GiB** |
| input block count | 1054 |
| num_gpus (ranks) | 16 |
| per-rank ESTIMATED (= pruned / 16) | 6,491,377 rows / **3.20 GiB** |
| per-rank MEASURED resident | 6.25-6.79 M rows / **3.13-3.40 GiB** (balanced) |
| RMM pool cap / GPU (0.80 x 31.74) | **25.39 GiB** |
| peak VRAM / rank (RMM allocated) | 6.66-**9.57** GiB |
| peak VRAM max (RMM) | **9.57 GiB** |
| peak VRAM (nvidia-smi, 1 GPU) | 16.22 GiB (see note) |
| headroom vs pool cap | **15.82 GiB** |
| headroom vs total VRAM | 22.17 GiB |

Note: `nvidia-smi` shows ~16 GiB/GPU because RMM pre-reserves a 50%-of-VRAM pool
**arena** at startup; the figure that decides "did it fit" is the **RMM allocated
peak (9.57 GiB)** vs the pool cap (25.39 GiB), not the reserved arena. The
all-to-all range partition is well balanced (per-rank rows within +-4%).

## 5. Spill / OOM -- EXPECTED vs MEASURED (not claimed from expectation)
- **Expected:** pruned sort input **51.26 GiB** over 16 GPUs = **~3.20 GiB/GPU**
  before overhead; RMM pool cap ~25.39 GiB ⇒ fits with spill **off**.
- **Measured (16-GPU run):**
  - `RAY_DATA_GPU_SORT_SPILL_FRAC` = **disabled (None)**
  - GPU/RMM device spill engaged = **no (path disabled)**
  - peak VRAM < RMM pool cap (no OOM) = **True (9.57 < 25.39 GiB)**
  - Ray object-store spilled bytes = **0 MiB**
  - Ray object-store restored bytes = **0 MiB**
  - `/tmp` spill dir bytes = **0**
  - Explicit: **`Ray object-store spilled bytes = 0`**, **`GPU/RMM device spill =
    disabled / not engaged`**, **`No OOM`**.

## 6. Current (cold start + transfers) vs theoretical device-resident
Sort phase timers are per-rank walls (max across ranks). Encode phase timers are
**summed GPU-seconds across all transform batches and all 16 workers**, so the
wall-equivalent is `/16`.

- **GPU sort (transfer-bound):** stage wall **30.76 s**; in-fn FULL **21.57 s** =
  H2D 3.34 + gpu_only **3.88** + shuffle 3.56 + D2H 1.78 + emit(D2H->object
  store + wait) **11.86**. Transfer/move (H2D+D2H+emit) **16.98 s** vs compute
  (gpu_only) **3.88 s** -> the sort is dominated by moving data, not sorting it.
  Theoretical resident sort (gpu_only, paid once) = **3.88 s**.
- **GPU encode (compute-bound):** stage wall **77.34 s**. Transform aggregate
  GPU-seconds: H2D 39.92, compute 88.53, D2H 5.60. Wall-equivalent (/16): H2D
  **2.50**, compute **5.53**, D2H **0.35** -> transfers are small vs compute; the
  77 s stage wall is dominated by **fixed overhead** (two actor-pool cold starts
  for fit + transform, and the driver-side vocabulary merge for the 12.16M /
  8.84M-cardinality columns), not data movement.
- **Chain sort->encode device-resident projection (wall-equivalent):** current
  two host-staged GPU stages **108.09 s** vs resident lower bound
  `H2D_once 3.34 + sort_gpu_only 3.88 + encode_compute 5.53 + D2H_once 1.78` =
  **~14.5 s** -> **~7.4x**. A device-resident execution mode would remove the
  sort's emit/D2H (~13.6 s) and keep the encoder context/vocabulary warm on
  device, converging the chain toward GPU-only compute -- the concrete payoff
  argument for an Arrow-CUDA resident path.

## 7. Repartitioning after GPU stages
- **post_sort_repartition** (kept): GPU sort emits `num_gpus = 16` blocks ->
  repartitioned to **1054** (the CPU-baseline block count) before CPU impute.
  **23.87 s**, `preserve_order=True`.
- **post_encode_repartition** (conditional, triggered): `GpuOrdinalEncoder` at
  batch 8M emitted **437** blocks (< 1054/2) -> repartitioned to **1054** before
  CPU scale + write. **18.02 s**, `preserve_order=True`.
- Final saved output (1054 files) verified **globally sorted** by
  `[user_id, day_int, display_order]`: `globally_sorted=True`,
  `rows 103,862,032/103,862,032 match`.

## 8. OOM analysis (empirical demo skipped by request)
With all 16 GPUs the per-rank partition (~3.2 GiB) fits with ~15.8 GiB headroom,
so no spill/OOM occurs (section 5). If a per-rank partition does **not** fit
(too few GPUs and/or a low `RAY_DATA_GPU_SORT_POOL_FRAC`), then on a single node
with device spill **off by default** the sort hits a hard `cudaMalloc`/RMM OOM
when a rank's allocation exceeds the pool cap. Mitigations: (a) add GPUs so
per-rank = pruned/num_gpus shrinks; (b) enable `RAY_DATA_GPU_SORT_SPILL_FRAC`
device->host spill (OOM-safe, slower); (c) raise `RAY_DATA_GPU_SORT_POOL_FRAC` if
VRAM allows. The `--demo-oom` mode in `bench_criteo_gpu_sort_encode.py`
demonstrates the failure and the spill mitigation when run.

## Artifacts
- Output dataset: `data/criteo_days1_30_gpu_sort_encode/` (1054 parquet files,
  16 GiB) + `manifest.json` (stage timings, GPU sort sizing, per-rank VRAM,
  spill status, encode phase split, fitted stats, verification).
- Encoder scan: `data/encoder_scan_days1_30.json` (2M),
  `data/encoder_scan_days1_30_c2.json` (4M/8M/1M/500k); logs
  `data/encoder_scan_c1.log`, `data/encoder_scan_c2.log`.
- Main run log: `data/gpu_sort_encode_main.log`.

## Reproduce
```bash
# 1) encoder block-size scan (cacheable, resumable)
RAY_DATA_GPU_SORT_NUM_GPUS=16 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
  .venv/bin/python gpu_preproc_bench/scan_encoder_blocksize.py --days 1-30 \
  --cache-dir /dev/shm/criteo_postimpute_days1_30 \
  --out gpu_preproc_bench/data/encoder_scan_days1_30.json

# 2) main 16-GPU run (GPU sort + GPU encode; scale CPU), batch size = scan winner
RAY_DATA_GPU_SORT_NUM_GPUS=16 RAY_DATA_GPU_PREPROC_NUM_GPUS=16 \
  .venv/bin/python gpu_preproc_bench/bench_criteo_gpu_sort_encode.py --days 1-30 \
  --gpus 16 --batch-size 8000000 \
  --out gpu_preproc_bench/data/criteo_days1_30_gpu_sort_encode

# 3) (optional) partition-OOM demo + mitigation
.venv/bin/python gpu_preproc_bench/bench_criteo_gpu_sort_encode.py --days 1-30 \
  --gpus 2 --pool-frac 0.5 --demo-oom
```
