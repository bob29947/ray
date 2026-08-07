# DGX BTS unified spillable GPU sort

## Result

One adaptive Ray Data algorithm now covers resident and external execution.
It range-partitions over all 16 GPUs with RAPIDS-MPF, retains a destination
when it fits, and performs one final pylibcudf sort. Before a constrained
destination crosses its residency watermark, it GPU-sorts the accumulated
rows into an immutable Arrow run, seals that run in Plasma, and releases its
VRAM. Fan-in-four passes merge external runs on GPU. No GPU-backend row was
sorted or merged on CPU.

On the 63.881-GiB full BTS workload, the resident GPU times were 54.471 and
53.337 seconds (53.904-second median), or 5.00x the corrected 269.752-second
default-Ray/PyArrow observation. The 191.642-GiB proof completed in 507.708
seconds with two GPU merge passes, bounded measured GPU use, global order,
and zero CPU fallback. These are DGX-local, directional results: each CPU cell
has one observation and each GPU trend point has two.

## Implementation

- Public entry point: `Dataset.sort(..., backend="gpu")`; the default CPU path
  is unchanged.
- Sixteen one-GPU actors use 65,536 deterministic byte-weighted samples,
  ordered ranges, GPU partitioning, and the NVSwitch RAPIDS-MPF all-to-all.
  Repeated boundaries use a deterministic internal row token to spread equal
  keys; the token is removed from the output.
- Resident and external execution are states of the same
  `partition_then_sort` algorithm, not separate algorithms. External runs are
  sorted before leaving VRAM and all run merges use pylibcudf.
- GPU-to-host Arrow output uses pinned allocation when supported. Ray objects
  are released only after replacement runs are sealed.
- RMM starts at 50% and is capped at 85% of VRAM. cuDF/MPF host spill and CPU
  fallback are disabled; Ray Core independently manages Plasma-to-disk spill.

The artifact-generation diff recorded 9 production files and +2,394/-2
production lines, plus one 332-line focused test file. All 20 source,
test, and harness paths were within `python/ray/data/**` or
`release/benchmarks/ray_data_gpu_sort/**`; Ray Core, dependencies, build, CI,
and the host were unchanged.

## Dataset and method

- Hardware: one DGX, 16 V100 32-GiB GPUs with NVSwitch, and 96 logical CPU
  threads.
- Ray: 2.55.1 at `237c2455ebb1ea15a32dd9e1fdeb2d617badc37f`; wheel SHA-256
  `bb49fbbe53a1d931e1f92d17f9271338f0b738885f8f70b7f531aa33f019d8af`.
- Data: normalized public BTS On-Time Performance data from April 2013 through
  December 2025 at `/raid/spark-team/bobbwang/datasets/bts-airline-on-time`.
  It has 109 native categorical, numeric, boolean, and time columns plus a
  deterministic `row_id`.
- Fixed trend cohort: 80,738,761 rows and 627 input blocks. Projection happens
  before materialization, so the 5-, 57-, and 110-column cells contain the
  same rows but decode to 2.857, 35.108, and 63.881 GiB.
- The four-key workload orders flights by route, date, and scheduled departure
  time: `Origin, Dest, FlightDate, CRSDepTime`. This is useful clustering for
  route/time grouping, joins, and downstream flight analysis rather than a
  synthetic sort key.
- Timed boundary: fully materialized Plasma input through sorted output sealed
  in Plasma. Reading/materialization and Ray startup are recorded separately
  and excluded. GPU actor startup, CUDA/RMM/MPF initialization, transfer,
  sorting, conversion, and output sealing are included.
- Each observation starts a fresh Ray runtime. CPU is one default Ray sort
  with no forced CPU count, object-store size, or shuffle settings. GPU is two
  fresh-runtime observations with all 16 GPUs.

### Exact smoke

One 160,000-row, 16-block, 110-column smoke sorted by the mixed
`Origin, OriginAirportID, FlightDate, row_id` key. PyArrow, resident GPU, and
forced-external GPU took 1.089, 11.951, and 13.238 seconds. Schema and every
row/value matched exactly. The 16-MiB forced run externalized all rows; the
small timings are correctness evidence, not performance points.

## Final six-cell trends

All rows, inputs, outputs, and schemas passed the benchmark gates. Every GPU
trend run used all 16 GPUs with zero external runs, MPF host spill, Ray disk
spill, or fallback.

| Cell | Cols / GiB | Keys | Read CPU/GPU s | CPU sort s | CPU RAID write/restore GiB | GPU r1/r2 s | GPU median | Speedup |
|:--|--:|:--|--:|--:|--:|:--|--:|--:|
| Narrow | 5 / 2.857 | four | 3.427/3.985 | 203.946 | 0/0 | 16.058/15.804 | 15.931s | 12.80x |
| Core | 57 / 35.108 | four | 12.361/12.392 | 243.555 | 0/0 | 33.188/34.147 | 33.667s | 7.23x |
| Full | 110 / 63.881 | four | 18.737/19.591 | 269.752 | 70.94/0.22 | 54.471/53.337 | 53.904s | 5.00x |
| Origin string | 110 / 63.881 | `Origin` | 18.578/19.142 | 223.844 | 122.94/36.46 | 51.321/50.881 | 51.101s | 4.38x |
| Origin integer | 110 / 63.881 | `OriginAirportID` | 19.073/19.202 | 156.026 | 127.43/0 | 50.980/50.377 | 50.678s | 3.08x |
| Route | 110 / 63.881 | `Origin, Dest` | 18.773/19.141 | 273.418 | 90.09/7.54 | 53.114/52.438 | 52.776s | 5.18x |

The corrected CPU observations above use default Ray startup. Ray reported a
200,000,000,000-byte (186.3-GiB) default object store; every input and final
output ref was resident at its boundary. Ray nevertheless spilled transient
shuffle objects in four wide cells, and those normal RAID writes/restores are
inside the CPU times. Rejected earlier artifacts with forced CPU or object
store settings are not used. The speedups therefore compare the tuned GPU
path with the intentionally requested default-Ray CPU behavior, including its
automatic disk spill where shown.

Directional findings:

- Payload is the clearest GPU cost. GPU time rises 15.931 -> 33.667 -> 53.904
  seconds as projected data rises 2.857 -> 35.108 -> 63.881 GiB. CPU rises
  203.946 -> 243.555 -> 269.752 seconds because all cells still sort 80.7
  million rows and perform the same range shuffle. Speedup consequently
  narrows from 12.80x to 7.23x to 5.00x as GPU movement and sealing grow.
- Integer keys help CPU much more than GPU here. Replacing `Origin` with its
  integer airport ID lowers CPU time 30.3%, while GPU medians differ by less
  than 1%. The integer cell therefore has the smallest GPU speedup, 3.08x.
- With the full payload, GPU medians are close for one, two, and four natural
  keys: 51.101, 52.776, and 53.904 seconds. CPU is 223.844, 273.418, and
  269.752 seconds: two keys cost more than one, while this single repetition
  does not resolve an additional two-to-four-key cost. CPU disk traffic also
  differs between cells, so this is not a statistical comparator claim.
- Single-airport ranges are less balanced: 2.60-4.93 GiB/rank, or 1.89x.
  Route and four-key outputs are about 1.06x balanced. Higher-cardinality
  compound keys improve range balance, but do not materially lower GPU time.

Historical GPU medians for these cells were 22.263, 36.775, 51.245, 49.951,
50.000, and 50.698 seconds respectively. The unified implementation improves
the narrow/core cells and is 5.2% slower on the full cell; historical
cross-campaign values are context, not regression statistics.

## Resident full-cell phase attribution

These are medians of the two additive controller wall decompositions. The
operator phases sum to 52.553 seconds; the remaining 1.350 seconds is the
outer Dataset/materialization boundary.

| Phase | Median s |
|:--|--:|
| Sampling and boundaries | 5.186 |
| GPU partitioning | 5.283 |
| RAPIDS-MPF exchange | 5.787 |
| Run/final GPU sort | 0.340 |
| GPU merge | 0.000 |
| GPU-to-Arrow conversion | 4.954 |
| Plasma sealing | 15.716 |
| Startup and orchestration | 15.287 |
| Cold total | 53.904 |

Peak device allocation was 16.236 GiB/rank and peak NVML usage was 16.496
GiB, leaving at least 15.50 GiB of physical headroom. Output balance was
1.06x. The final sort kernel is small; output sealing and startup/orchestration
are the largest measured components.

## Four-point externalization trend

`B` was the measured 16.236-GiB resident peak. The nominal 0.75B trial used a
12.177-GiB budget but produced no GPU external run, so it was diagnostic and
the one allowed replacement used 9.741 GiB. The diagnostic took 233.837
seconds and independently caused 127.80 GiB of Ray disk writes despite zero
GPU externalization; it is excluded from the four-point curve.

| Point | Budget/GPU | GPU r1/r2 s | Median / slowdown | First externalization | Initial/replacement runs; passes | Ray RAID write/restore median GiB |
|:--|--:|:--|:--|:--|:--|--:|
| Resident | default | 54.471/53.337 | 53.904s / 1.00x | none | 0/0; 0 | 0/0 |
| Light | 9.741 GiB | 144.151/163.099 | 153.625s / 2.85x | 39.47s, wave 0 | 32/16; 1 | 210.18/25.09 |
| Medium | 8.118 GiB | 139.637/142.363 | 141.000s / 2.62x | 36.46s, wave 0 | 32/16; 1 | 188.67/18.23 |
| Heavy | 4.059 GiB | 201.457/203.113 | 202.285s / 3.75x | 24.88s, wave 0 | 48/16; 1 | 255.58/68.69 |

Every retained constrained point externalized all 80,738,761 rows, about
63.89 GiB. H2D/D2H traffic rose from 127.8/63.9 GiB resident to about
191.7/127.8 GiB, and Plasma read/write traffic rose from 0/63.9 GiB to about
127.8/191.7 GiB. Light and medium have the same run geometry and their timing
order is noise at two observations; heavy creates 50% more initial runs and
is directionally slower. Across the external points, the dominant added costs
are Plasma sealing, RAID spill/restore, and orchestration, not the sub-second
run-sort or roughly 0.4-1.3-second GPU-merge phases.

## 191.642-GiB large/skew proof

This proof repeats the full cohort three times with uniquely offset `row_id`,
retains all 110 columns, sorts by `Origin`, and uses the heavy 4.059-GiB
budget.

| Metric | Final observation |
|:--|:--|
| Input | 191.642 GiB; 242,216,283 rows; 1,881 blocks |
| Cold sort / throughput | 507.708s; 0.377 GiB/s; 477,078 rows/s |
| Input materialization, excluded | 52.359s; zero input spill |
| GPU externalization | 191.651 GiB; all rows |
| Initial / replacement runs; merge passes | 136 / 51; 2 |
| Ray RAID write / restore | 781.553 / 526.241 GiB |
| Plasma read / write | 568.180 / 759.829 GiB |
| H2D / D2H | 759.816 / 568.180 GiB |
| Peak device / NVML / headroom | 16.527 / 17.096 / 14.904 GiB |
| Output range balance | 7.804-15.589 GiB/rank; 2.00x |
| CPU sort / CPU merge / fallback | 0 / 0 / 0 |
| Validation | globally ordered; exact rows and `row_id` checksum |

Its measured phases were 13.946 seconds sampling, 12.013 partitioning, 23.277
MPF exchange, 0.475 run sorting, 3.153 GPU merge, 45.758 Arrow conversion,
147.690 Plasma sealing, and 243.078 orchestration. The result proves this BTS
case completes beyond aggregate VRAM; it does not prove an adversarial memory
bound.

## Spill placement

`/dev/shm/rgs/<trial>/plasma` contains only the Plasma object store. Actual Ray
filesystem spill is configured at
`.venv/gpu-sort-external-runtime/<trial>/ray-spill` on `/raid`, matching the
local-disk/NVMe tier normally used on a cluster. Workers reject any filesystem
spill directory below `/dev/shm`. The constrained 64-GiB trials use 132 GiB
of Plasma and the large proof uses 256 GiB, so intermediate runs exercise
real RAID spill. Cumulative Ray Core counters retain write/restore traffic
after restored spill files are deleted. CPU default-Ray spill uses the same
RAID location. MPF host-spill bytes were zero throughout.

## Important receiver-credit limitation

The implementation proactively bounds tables retained after receive, but it
does not yet bound transient inbound MPF bytes before receive. Each source
computes local partition row offsets, immediately packs/sends them, and the
destination extracts its completed partition before `_accept_received()` can
externalize it. The controller has no pre-shuffle destination-byte admission
step.

With an explicit residency budget `B`, a normal wave may contain `B/2` per
source when that exceeds the 256-MiB floor, as it did here. Sixteen
adversarially skewed sources could therefore direct roughly `8B` of raw
payload to one receiver before its retained-payload limit of `B/2.7` is
applied. An oversized input block is not split and can exceed that estimate.
With the default budget the resident fast path uses one whole-input wave.

The 65,536-row sample and repeated-boundary routing handled natural BTS skew:
the large proof's hottest output was 15.589 GiB and the run completed. That is
empirical coverage, not a guarantee for an unseen all-to-one distribution.
A production follow-up should add exact per-destination preflight/receiver
credits and bounded micro-waves while retaining the one-collective balanced
fast path.

## Validation scope and artifacts

Focused tests cover API/schema planning, inverse-inclusion sample weights,
PyArrow-compatible null/NaN comparison, typed all-null output, equal-key
distribution, spill transition, and bounded fan-in merge progress. The exact
live resident/external smoke is the integration check. No broad Ray suite,
failure injection, cloud run, or repeated correctness matrix was performed.

Machine-readable evidence is in `.venv/gpu-sort-external-artifacts/study.json`
and the unchanged raw JSON files under
`.venv/gpu-sort-external-artifacts/trials/`. Historical provenance remains in
each trial; current environment identity is recorded separately in the study
artifact.
