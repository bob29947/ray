# Should the general multi-GPU sort be merged into Ray Data? — the evidence

**Subject of review:** the `general` GPU sort backend (cuDF + rapidsmpf),
selected by the user-facing `ds.sort(..., gpu=True)` flag, implemented in
`ray/data/_internal/planner/gpu_sort_general.py` and routed by
`ray/data/_internal/planner/sort.py` (commit `9dcf2c4`, branch
`gpu-sort-general`).

**Claim under test:** this backend is a *faithful, general drop-in* for Ray
Data's CPU sort that delivers a large end-to-end win on GPU hardware, is bounded
by NVLink (not by anything we add), and exposes a device-resident sort fast
enough to justify a future device-resident execution mode in Ray Data.

**TL;DR recommendation:** **Merge as an experimental, opt-in backend.** On a
64 GiB sort it is **6.1× faster than pyarrow and 4.9× faster than polars**
end-to-end, it is a correct drop-in across dtypes / multi-key / nulls
(including a 17 GiB datetime sort that matches pyarrow element-for-element), its
all-to-all is NVLink-bound (**818 GiB/s**, 2.2× the hand-rolled P2P loop), and
its in-VRAM sort (**0.37 s, ~93× the CPU wall**) shows the prize that a
device-resident mode would unlock. The honest blockers to *default-on* are the
pageable `cudf.to_arrow` D2H (the general path's largest single cost), the
single-node / all-GPUs-local assumption, and the dependency weight of
cuDF + rapidsmpf + UCXX. Details, caveats, and exact repro commands below.

> All numbers below were produced **this session** on the box described in
> [Environment](#environment), with a fresh process + fresh `ray.init()` per
> backend, a warmup that is measured but **not** counted, and ≥3 timed trials.
> We report **best** and **median** because a one-off run-2 object-store
> high-water-mark spike hits *every* backend (visible in the raw trials), so
> mean is not representative. Raw logs are committed alongside this file.

---

## 0. Correction & update (later session): CPU-baseline fix + non-int32 at scale

Two **measurement** issues were found and fixed after the rest of this doc was
written (they do **not** affect the GPU engine itself, which is unchanged):

1. **`gpu=False` does not force the CPU sort.** `Dataset.sort` maps
   `gpu=False → op_gpu=None`, which — when `RAY_DATA_GPU_SORT_IMPL=general` is in
   the environment — resolves to the **GPU** engine, not pyarrow. The only
   reliable CPU opt-out is **`backend="cpu"`**. Any script that took a CPU
   baseline with a plain `ds.sort()` was therefore *not* timing pyarrow.
   - Fixed in `large_nonint32_sort.py` and `test_general_sort.py`.
   - **The §1 64 GiB numbers are unaffected** — `cpu_vs_gpu_general.py` runs the
     CPU backend in a *separate process that never sets that env*, so pyarrow vs
     GPU there was always a real comparison. The **6.1×** stands.

2. **`test_general_sort.py`'s "oracle" was the GPU engine** (same bug) — it
   compared the engine to itself (vacuous PASS). It now validates against an
   **independent pandas** oracle. (Ray's pyarrow CPU sort *cannot* be the oracle
   for that input: it **raises** on null string keys, see below.) Re-run: **PASS**.

**Non-int32 results at scale — real pyarrow CPU, warm, fixed harness**
(`cpu_vs_gpu_dtypes.py`; 16-column tables to match the §1 shape so the CPU
object-store shuffle is the bottleneck, the regime where the GPU win is real):

| dataset | size | GPU best | CPU best | speedup | sorted |
|---|--:|--:|--:|--:|:--:|
| **datetime** key (1Gi × 16) | 68.00 GiB | **5.90 s** | 37.56 s | **6.4×** | PASS |
| datetime key (256Mi × 16) | 17.00 GiB | 1.83 s | 12.52 s | 6.8× | PASS |
| **strings** 3-key asc/desc (1Gi × 16, no-null) | 81.25 GiB | **10.84 s** | 625.3 s¹ | **~58×** | PASS² |
| strings 3-key asc/desc, **null keys** | 81 GiB | works | **raises**³ | — | GPU-only |

¹ one warm CPU sort (each CPU string sort ≈ 10 min; GPU is best-of-3).
² order validated GPU == independent pandas/pyarrow at smaller scale + datetime
at 68 GiB; the 81 GiB GPU output is globally sorted, all 1,073,741,824 rows.
³ Ray's CPU sort: `TypeError: '<' not supported between instances of 'NoneType'
and 'str'` (it does `np.searchsorted` on the null-containing string key). The
GPU engine sorts null string keys correctly — a capability the CPU path lacks.

**Why the strings win is so much larger than datetime's:** Ray's CPU sort is
pathologically slow on string multi-key (per-boundary `np.searchsorted` over
string object arrays + a string multi-key merge), and outright unsupported with
null string keys. The datetime **6.4×** is the cleaner hardware-bound number
(both engines efficient); the strings **~58×** reflects the GPU engine's added
value where the CPU path is weak or broken.

---

## 1. Four-way end-to-end sort on 64 GiB

Dataset (identical for all four): **1Gi rows (1,073,741,824) × 16 int32 columns
= 64 GiB**, sort key `c0` = random int32, other 15 columns zero,
`numpy.random.default_rng(0)`. Timed region is the warm, in-memory→in-memory
`ds.sort("c0").materialize()` (GPU backends use `ds.sort("c0", gpu=True)`),
excluding process/cluster startup and data generation. Each result is
independently full-scan verified (row count / key sum / min / max / global
monotonic order across every output block).

| backend | wall best | wall median | FULL | H2D | shuffle | GPU-only | D2H | vs pyarrow (best) | vs polars (best) | sorted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| **pyarrow** (CPU default) | 34.752 s | 34.811 s | — | — | — | — | — | 1.00× | 0.79× | **PASS** |
| **polars** (Ray CPU, polars kernels) | 27.540 s | 28.736 s | — | — | — | — | — | 1.26× | 1.00× | **PASS** |
| **gpu_tuned** (hand-tuned int32) | 6.566 s | 6.979 s | 3.587 s | 1.926 s | (P2P inline) | **0.266 s** | 1.395 s | 5.29× | 4.19× | **PASS** |
| **gpu_general** (cuDF + rapidsmpf) | **5.668 s** | **5.795 s** | 5.462 s | 1.592 s | 0.110 s | **0.372 s** | 3.470 s | **6.13×** | **4.86×** | **PASS** |

GPU phase seconds are the in-actor timers for the **best** trial. `FULL` =
RAM→VRAM→sort→VRAM→RAM; `GPU-only` = in-VRAM→sorted-in-VRAM (incl. the shuffle).

**Raw trials (showing the known run-2 spike — why best/median, not mean):**

```
pyarrow   : 34.811, 42.678, 34.752   (warmup 48.186 s)
polars    : 28.736, 34.939, 27.540   (warmup 40.343 s)
gpu_tuned :  6.566, 17.509,  6.979   (warmup 137.826 s)
gpu_general: 5.795, 19.020,  5.668   (warmup 46.302 s)
```

The run-2 spike is a transient object-store high-water-mark event that hits all
four backends; it is excluded from best and does not move the median.

**Reading the table:**

- **The general engine is the fastest end-to-end backend**, beating both CPU
  kernels *and* the hand-tuned GPU engine on wall time (5.668 s vs 6.566 s):
  **6.1× vs pyarrow, 4.9× vs polars** (best), 6.0× / 5.0× on median. Against the
  separately-documented 45.691 s pyarrow baseline it is **8.1×**.
- polars is a real CPU win (1.26× over pyarrow) and is the **stronger CPU
  baseline** — the GPU argument is made against polars too, not just pyarrow.
- **FULL is ~93% data movement.** H2D + D2H = 5.06 s of the general engine's
  5.46 s FULL (92.7%); the sort compute is 0.37 s (6.8%). The tuned engine is
  the same shape: 3.32 s of 3.59 s (92.6%) is transfer. *The whole benchmark is
  a host↔device transfer contest* — which is exactly the setup for §2.
- The general engine's cold start is also far cheaper than the tuned engine's
  (46 s vs 138 s warmup), because it doesn't pre-build large pinned bounce
  buffers.

---

## 2. GPU-only sort, and why Ray should keep blocks device-resident

This is the core of the merge argument for a **device-resident execution mode**.

The device-resident sort — *in-VRAM → sorted-in-VRAM*, including the NVLink
all-to-all — costs, on 64 GiB:

| | seconds | vs pyarrow (34.752 s) | vs polars (27.540 s) | vs documented 45.691 s |
|---|---:|---:|---:|---:|
| **gpu_general GPU-only** | **0.372 s** | **93.4×** | **74.0×** | 122.8× |
| gpu_tuned GPU-only | 0.266 s | 130.6× | 103.5× | 171.8× |

So the *sort itself* is ~2 orders of magnitude faster than either CPU kernel.
But end-to-end we only see 6.1× — because, as §1 showed, **H2D + D2H dominate
FULL**:

```
FULL  =  H2D  +  GPU-only  +  D2H        (general, 64 GiB, best trial)
5.462 = 1.592 +   0.372    + 3.470       H2D+D2H = 92.7% of FULL
```

If Ray kept blocks **device-resident across operators**, H2D would be paid once
at ingest and D2H once at output, and every chained GPU op in between would run
at ~GPU-only cost. For a pipeline of **N** GPU ops on resident data:

```
amortized wall per op  ≈  (H2D + N·GPU_only + D2H) / N   →   GPU_only  as N grows
```

### Resident multi-op microbench (measured)

`resident_multiop_bench.py` demonstrates this on the **same 64 GiB**, 16 one-GPU
rapidsmpf actors (4 GiB/GPU). It runs K=8 sorts two ways and verifies the
resident result is globally sorted (**PASS**, all 1,073,741,824 rows preserved):

- **resident:** H2D **once** → 8 device-resident sorts back-to-back (no host
  round-trip; key direction alternated each op so every op is a real re-sort) →
  D2H **once**.
- **host_staged:** 8 sorts where **each** op pays H2D + sort + D2H.

Measured (best/steady-state): **H2D 1.409 s (once), D2H 2.119 s (once),
GPU-only 0.343 s/op, host-staged FULL 3.832 s/op**. (The host-staged FULL of
3.83 s here is the engine's FULL minus Ray's object-store `ray.put`
serialization, which the microbench omits to isolate the transfer; GPU-only
0.343 s matches the §1 engine's 0.372 s.)

| N (chained ops) | resident /op = (1.409 + N·0.343 + 2.119)/N | host-staged /op | speedup |
|---:|---:|---:|---:|
| 1 | 3.870 s | 3.832 s | 0.99× |
| 2 | 2.107 s | 3.832 s | 1.82× |
| 3 | 1.519 s | 3.832 s | 2.52× |
| 4 | 1.225 s | 3.832 s | 3.13× |
| 6 | 0.931 s | 3.832 s | 4.12× |
| 8 | 0.784 s | 3.832 s | 4.89× |
| **N → ∞** | **→ 0.343 s** | 3.832 s | **11.2×** |

Per-op resident times (ms), showing convergence after a cold first op (pool
growth): `5661, 978, 642, 570, 376, 579, 343, 349` — steady state ~0.34–0.58 s,
i.e. **GPU-only**. Host-staged is flat at ~3.83–3.86 s because it repays the
round-trip every op.

**The takeaway for Ray:** a single GPU sort can only ever be ~6× (transfer
bound). But a *chain* of GPU operators on device-resident blocks amortizes the
two transfers and converges to the GPU-only regime — **up to ~11× faster than
re-staging through host per op**, and ~93× faster than the CPU sort, for the
sort step. This is the concrete payoff that a device-resident (Arrow-CUDA)
execution mode in Ray Data would unlock; this backend is a working first
operator for it.

---

## 3. Shuffle isolation — the distributed all-to-all is NVLink-bound

A distributed sort's hard part is the all-to-all. `shuffle_bench.py` strips away
the sort and moves the same 64 GiB (each GPU holds 1/16, pre-split into 16
chunks) two ways:

| all-to-all backend | best | fabric throughput | dataset throughput |
|---|---:|---:|---:|
| tuned P2P `cudaMemcpyPeerAsync` | 158.398 ms | 378.8 GiB/s | 404.0 GiB/s |
| **rapidsmpf Shuffler (UCXX)** | **73.369 ms** | **817.8 GiB/s** | 872.3 GiB/s |

- The rapidsmpf shuffle is **2.16× faster** than the hand-rolled P2P loop — UCXX
  pipelines the all-pairs sends/receives better than a per-rank sequential
  `memcpyPeerAsync` loop. **Adopting the general engine is *not* a shuffle
  regression; it is an improvement.**
- Both run **8–18× above the box's ~46 GiB/s host↔device ceiling**, which is the
  proof the data rides **NVLink/NVSwitch**, not a PCIe/host-staging fallback (a
  host fallback would be bounded by ~46 GiB/s ≈ 1.3 s for 60 GiB, ~17× slower).
- Inside the full sort the shuffle is only ~0.11 s (§1) — a small fraction of
  FULL. The distributed exchange is **not** the bottleneck; the host transfers
  are.

---

## 4. Generality / drop-in correctness — "a faithful replacement for Ray's CPU sort"

The oracle for all correctness is **pyarrow** (`ds.sort()` default), because the
general engine deliberately matches pyarrow's `null_placement="at_end"` (Ray's
default). polars/pandas place nulls differently for *descending* keys, so where
we mention polars we treat null-placement differences as expected, not failures.

### 4a. `test_general_sort.py` — strings + multi-key (asc/desc) + nulls (PASS)

```
ds.sort(["s", "f", "g"], descending=[False, True, False], gpu=True)
```

4M rows; key `s` = strings (~12% null), `f` = float64 (~12% null), `g` = int32.
Oracle = an **independent pandas sort** with `na_position="last"` (see §0:
Ray's pyarrow CPU sort **raises** on null string keys, so it cannot be the
oracle here; and a plain `ds.sort()` would route to the GPU engine, which would
make the test compare the engine to itself).

- global key order **== independent pandas oracle (nulls last): PASS**
- payload / row integrity (re-keyed by unique id, every column matches input):
  **PASS** → **RESULT: PASS**

This proves `backend="gpu"` is a faithful drop-in across dtypes, multiple keys
with mixed directions, and nulls — with pandas/pyarrow-consistent null placement.
(Float nulls are real Arrow nulls, not NaN: cuDF treats NaN as the largest float,
which would sort *first* under a descending key, so a true null is used to keep
"nulls last" unambiguous and engine-vs-oracle identical.)

### 4b. Large (≥16 GiB) NON-int32 sort at scale — datetime "sort by time" (PASS)

`large_nonint32_sort.py --key datetime` builds a **~17 GiB** dataset whose sort
key `c0` is a **timestamp** (random calendar dates centered on **2026-01-03**,
spanning ~2016–2032), sorted ascending end-to-end with `ds.sort("c0",
backend="gpu")`, and verified **element-for-element against Ray's pyarrow order**
(oracle = `numpy.sort` of the input keys; both the GPU result and a real pyarrow
`backend="cpu"` sort are checked to equal it exactly).

```
sample sorted dates: 2016-01-06, 2019-05-06, 2022-09-04, 2026-01-02, 2029-05-03, 2032-08-31
GPU  ds.sort('c0', backend="gpu"): 1.831 s  (h2d 0.584, gpu_only 0.094, shuffle 0.024, d2h 0.904)
                                   rows ok, monotonic, ==oracle(Ray-pyarrow order)=True  -> PASS
CPU  ds.sort('c0', backend="cpu"): 12.521 s   ==oracle -> PASS  (real pyarrow; backend="cpu" required)
RESULT: GPU == Ray-pyarrow sort -> PASS   (GPU 1.83 s vs CPU 12.52 s = 6.8×)
```

Generality holds **at scale**, on a non-int32 (datetime) key, not just in the
small unit test — and now against a *real* pyarrow CPU baseline (§0). The
larger **68 GiB** datetime run (§0) gives **6.4×** on the same shape as §1; this
17 GiB run gives **6.8×**. The no-null int32 64 GiB case (§1) is unambiguous
against both pyarrow and polars; these datetime cases are unambiguous against
pyarrow.

---

## 5. Should this be merged?

**Recommendation: yes — merge it as an experimental, opt-in backend
(`ds.sort(..., gpu=True)`), with the CPU sort remaining the default.** The
evidence clears every bar set for it, and the residual weaknesses are about
*production hardening*, not correctness or approach.

### Strengths (the case for)

1. **General drop-in, proven.** Columnar cuDF tables + per-column
   `(Order, NullOrder)` chosen to match pyarrow → arbitrary dtypes,
   strings, datetime, multi-key asc/desc, and nulls-last placement.
   Verified against an independent pandas oracle in the unit test **and**
   against real pyarrow at 17 GiB / 68 GiB on a datetime key (§0, §4).
   **It also sorts inputs the CPU path cannot:** Ray's pyarrow sort *raises*
   on null string keys (`None < str` in its boundary search); the GPU engine
   sorts them correctly — a robustness win, not just a speed win (§0).
2. **Big end-to-end win vs *both* CPU kernels.** 5.67 s on 64 GiB = **6.1× vs
   pyarrow, 4.9× vs polars** (best), and it even beats the hand-tuned int32
   engine on wall time while being fully general (§1).
3. **The distributed all-to-all is NVLink-bound and *better* than the tuned
   path** — 818 GiB/s, 2.16× the hand-rolled P2P loop, 18× over the host ceiling
   (§3). The hard part of a distributed sort is not a regression.
4. **Standard RAPIDS components.** It is `pylibcudf` sort kernels + the official
   `rapidsmpf` Ray integration (`BaseShufflingActor`, `setup_ray_ucxx_cluster`,
   `Shuffler`, `split_and_pack`/`unpack_and_concat`) over UCXX — not a bespoke
   transport. Lower long-term maintenance than the hand-tuned engine.
5. **GPU-only justifies a device-resident mode.** The sort is 0.37 s (~93× the
   CPU wall); FULL is ~93% host transfer; a resident multi-op chain amortizes
   that and converges to GPU-only, up to ~11× faster than per-op host staging
   (§2). This backend is a concrete first operator for a future Arrow-CUDA
   device-resident execution path in Ray Data.

### Weaknesses / risks (the honest case against default-on)

1. **Pageable `cudf.to_arrow` D2H is the main remaining cost.** D2H is 3.47 s of
   the general engine's 5.46 s FULL — it is the single largest phase and the
   reason the tuned engine's FULL (3.59 s) still beats the general engine's
   (5.46 s) even though the general engine wins the *wall* and the *shuffle*.
   The tuned path's pinned, reused D2H buffers do the same move in 1.40 s
   (~2.5×). A production version should give the general path a pinned/pooled
   D2H (or, better, skip D2H via device-resident output) — this is the highest-
   value follow-up and would push the general engine well past the tuned one
   end-to-end.
2. **Single-node, all-GPUs-local assumptions.** One UCXX communicator over local
   NVLink, detached actors named per-process, `num_gpus` read from env. No
   multi-node UCXX bootstrap, no rail-aware/NIC topology, no fault tolerance if a
   rank dies mid-shuffle. Fine for one DGX-class box; not yet for a multi-node
   Ray cluster.
3. **Dependency weight.** cuDF + rmm + rapidsmpf + UCXX (+ pylibcudf) is a heavy,
   CUDA-version-pinned stack (here CUDA 12.9, RAPIDS 26.02). It must stay opt-in
   and import-lazy (it already is — heavy imports are deferred until the GPU path
   runs) so CPU-only Ray installs are unaffected.
4. **Cluster-lifecycle sharp edges.** The engine reuses detached actors across
   trials for speed; stale detached actors from a crashed run will OOM the next
   run unless torn down (`kill_actor_pool`). A production version needs robust
   actor lifecycle tied to the Ray Data execution, not module globals
   (`LAST_RUN_STATS`, `_OP_COUNTER`).
5. **Range-partition balance depends on the key sample.** Boundaries come from a
   strided key sample sorted on one rank; pathologically skewed keys could
   produce unbalanced partitions / a straggler rank. The CPU sort samples too,
   but this path should be stress-tested on skewed and heavy-duplicate keys.

### What a production version would need
Pinned/pooled (or device-resident) D2H; multi-node UCXX bootstrap + topology
awareness; shuffle fault tolerance; actor lifecycle owned by the executor;
skew/duplicate-key stress tests; and ideally the device-resident execution mode
that §2 shows is the real prize.

### Bottom line
Merge **behind the existing `gpu=True` opt-in** (default stays CPU). It is
correct, general, materially faster than the best CPU kernel, NVLink-bound on
the hard part, and built from standard RAPIDS pieces. Land it as experimental,
then invest in pinned/device-resident D2H and a device-resident execution mode —
the path to the ~11×/93× regime §2 quantifies.

---

## Environment

- **16× Tesla V100-SXM3-32GB** (512 GB total VRAM), fully connected via NVSwitch
  (`NV6` all-to-all NVLink); 96 CPU cores; ~1.5 TB RAM.
- **CUDA 12.9**, Python 3.10. **Ray 2.55.1**, **cudf 26.02.01**, **rmm 26.02.00**,
  **rapidsmpf 26.02.000**.
- Box host↔device aggregate ceiling (measured previously, see `RESULTS.md`):
  ~46 GiB/s H2D, ~49 GiB/s D2H under 16-way contention.

## Exact commands (reproducible)

All runs use the repo venv. **Between every run** the GPUs/actors were released
and verified idle:

```bash
.venv/bin/ray stop --force && sleep 3 && \
  nvidia-smi --query-gpu=memory.used --format=csv,noheader   # expect 16× "0 MiB"
```

```bash
# 1) Four-way end-to-end sort, 64 GiB, 3 trials each (pyarrow | polars | gpu_tuned | gpu_general)
.venv/bin/python cpu_vs_gpu_general.py --trials 3                       # -> bench_4way_64gib.log

# 2) Resident multi-op VRAM-amortization microbench, 64 GiB, K=8
.venv/bin/python resident_multiop_bench.py --k 8 --gpus 16 --json       # -> resident_multiop.log

# 3) Shuffle isolation (run separately, cleanup between)
.venv/bin/python shuffle_bench.py --backend rapidsmpf --trials 5        # -> shuffle_rapidsmpf.log
.venv/bin/python shuffle_bench.py --backend gpu       --trials 5        # -> shuffle_gpu.log

# 4a) Generality: strings + multi-key asc/desc + nulls vs pyarrow oracle
.venv/bin/python test_general_sort.py --rows 4000000 --blocks 32 --gpus 16   # -> test_general_sort.log

# 4b) Large (~17 GiB) NON-int32 datetime "sort by time" vs pyarrow oracle (element-wise)
.venv/bin/python large_nonint32_sort.py --key datetime \
    --rows $((256*1024*1024)) --blocks 256 --gpus 16                    # -> large_datetime_sort.log
```

Backend selection (all four measured in their own fresh process):
`pyarrow` = default; `polars` = `RAY_DATA_USE_POLARS_SORT=1` +
`POLARS_MAX_THREADS=1`; `gpu_tuned` = `ds.sort(gpu=True)` +
`RAY_DATA_GPU_SORT_IMPL=tuned`; `gpu_general` = `ds.sort(gpu=True)` +
`RAY_DATA_GPU_SORT_IMPL=general` (the merge candidate; the default of
`gpu=True`). `cpu_vs_gpu_general.py` sets these per subprocess.
