# BTS distributed GPU sort: historical AWS results

These are the final measurements from the instrumented Ray 2.55.1 candidate
used to develop the production implementation.  They are retained separately
from the production PR: the cleaned port removes benchmark telemetry and is not
claimed to be byte-for-byte identical to this measured overlay.

## Method and hardware

- 16 retained AWS `g6.4xlarge` nodes, one NVIDIA L4 24 GB GPU per node.
- 256 aggregate vCPUs and 289.04 GiB aggregate default-Ray Plasma.
- Local NVMe Ray spill; zero swap; no artificial GPU residency cap.
- Ray 2.55.1 (`237c2455eb`), PyArrow 24.0.0, cuDF 26.2.1, RMM and
  RAPIDS-MPF 26.2.0, UCXX 0.48.0.
- CPU denominators are archived one-observation results from 16 retained
  `m5dn.4xlarge` nodes using default Ray settings.  They are cross-campaign,
  directional comparisons.
- Deterministic stratified CPU planning sample, 512 MiB bounded input batches,
  0.375 wave fraction, fan-in-four GPU merge.
- Input was materialized before timing.  The headline is unmodified
  `Dataset.sort(..., backend="gpu").materialize()` cold wall time.
- Ray restarted between observations while the same 16 instances and staged
  local dataset were retained.

The immutable measured bundle digest was
`203ae1f8853a8044b489f6a575fb14a7ab5e7fe60daa02b72210607f7b9dab33`.

## Payload and key trends

| Workload | Columns | Sort keys | GPU observations | GPU median | Archived default-Ray CPU | Directional speedup |
|---|---:|---|---:|---:|---:|---:|
| Narrow | 5 | Four-key baseline | 5.52 / 5.37s | **5.44s** | 69.06s | **12.69x** |
| Core | 57 | Four-key baseline | 19.00 / 17.49s | **18.25s** | 115.69s | **6.34x** |
| Full | 110 | Four-key baseline | 31.53 / 29.96s | **30.74s** | 118.93s | **3.87x** |
| Origin string | 110 | `Origin` | 32.20 / 32.68s | **32.44s** | 82.64s | **2.55x** |
| Origin integer | 110 | `OriginAirportID` | 32.48 / 32.16s | **32.32s** | 59.01s | **1.83x** |
| Route | 110 | `Origin, Dest` | 30.26 / 29.58s | **29.92s** | 122.76s | **4.10x** |

Payload movement is the largest GPU factor: narrow, core, and full take 5.44,
18.25, and 30.74 seconds.  String versus integer keys barely changes GPU time,
while the CPU benefits materially from the integer key.  At full payload GPU
time is nearly flat for one, two, and four natural keys; the CPU changes much
more, so key choice primarily changes the relative speedup.

## Natural spill trend

| Decoded input | GPU observations | GPU median | Archived CPU | Directional speedup | GPU externalized | Ray NVMe write / restore | Run geometry |
|---|---:|---:|---:|---:|---:|---:|---|
| 63.9 GiB | 31.53 / 29.96s | **30.74s** | 118.93s | **3.87x** | 0 GiB | 0 / 0 GiB | resident, 16 outputs |
| 127.8 GiB | 98.82 / 96.94s | **97.88s** | 499.95s | **5.11x** | 127.77 GiB | 383.35 / 14.45 GiB | 49 runs, one merge pass |
| 156.5 GiB | 96.82 / 82.75s | **89.78s** | did not complete | - | 156.52 GiB | 316.87 / 103.20 GiB | 50-51 runs, one merge pass |

Externalized bytes are cumulative GPU-sorted run output, not simultaneous VRAM
excess.  At 2x and 2.45x every row is externalized once, then read back in
bounded groups and merged on GPU.  Ray may independently evict some Plasma run
objects to NVMe.  No dataset row is sorted or merged on CPU.

Rows double from 1x to 2x, while GPU time grows 3.18x because run output and
merge add a second H2D/D2H pass plus Plasma and NVMe traffic.  Archived CPU time
grows more sharply, so the directional speedup increases to 5.11x.  The 2.45x
observations have material variability; their median does not prove 2.45x is
intrinsically faster than 2x.  No 2.45x speedup is reported because the archived
CPU observation did not complete.

## Pipeline attribution

| Size | Plan | Partition + MPF | Sort + merge | Arrow + seal | Orchestration | Unattributed | Official cold |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1x | 0.63s | 20.55s | 0.17s | 4.46s | 3.69s | 1.24s | **30.74s** |
| 2x | 1.17s | 39.50s | 1.15s | 23.44s | 31.14s | 1.49s | **97.88s** |
| 2.45x | 1.36s | 44.15s | 1.48s | 27.94s | 13.32s | 1.53s | **89.78s** |

These are diagnostic median inner intervals.  No interval or Ray bookkeeping
delay is subtracted from the official cold wall time.

## Compact optimization ledger

| Candidate | Net production lines | Official 2x before -> after | Change | Decision |
|---|---:|---:|---:|---|
| Direct final merge | +64 | 133.40 -> 116.29s | -12.82% | accepted |
| 512 MiB adjacent-block batching | +64 | 116.29 -> 96.03s | -17.42% | accepted |
| 1 GiB batch target | 0 | 96.03 -> 91.14s | -5.09% | rejected: repeatable 2.45x RMM/MPF OOM |
| Batched run restore | +8 | 91.14 -> 95.65s | +4.94% | rejected |

Direct final merge removed replacement-run materialization.  Bounded batching
reduced 1,254 block operations to 289 at 2x without materially changing H2D,
network, or spill bytes.  The final 0.375 wave setting was 4.65% faster than
0.50 in the 2x screen.

## Validation

The 160,000-row smoke matched PyArrow schema and every row/value exactly.  All
16 authoritative GPU observations had correct rows, schema, global order, and
`row_id` checksum; used 16 ranks on distinct nodes; produced locatable output;
and reported zero CPU dataset sort, CPU merge, fallback, and MPF host spill.
Input materialization had zero pre-timer spill in every accepted observation.
