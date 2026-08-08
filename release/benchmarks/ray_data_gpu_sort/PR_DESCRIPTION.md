# Ray Data: experimental spillable distributed GPU sort

## Summary

This change adds an opt-in `Dataset.sort(..., backend="gpu")` path. The
existing CPU backend and its default behavior are unchanged.

The GPU backend implements one adaptive distributed `partition_then_sort`
algorithm:

- Resident destinations remain in VRAM and receive one final pylibcudf sort.
- A destination approaching its payload watermark is GPU-sorted into an
  immutable Arrow run, sealed as a Ray ObjectRef, and released from VRAM.
- External runs are merged on GPU in bounded fan-in-four passes.
- Ray Core remains responsible for evicting Plasma run objects to the normal
  cluster filesystem spill tier.
- No row is sorted or merged on CPU.

The implementation and benchmarks are confined to `python/ray/data/**` and
`release/benchmarks/ray_data_gpu_sort/**`. There are no Ray Core, build, CI,
global dependency, or host changes. AWS lifecycle code is benchmark-only.

## Why `partition_then_sort`

The preceding BTS work selected range-partition-first sorting for wide,
variable-width rows. It avoids a full local presort whose ordering is mostly
discarded by the all-to-all, uses one direct RAPIDS-MPF range exchange, and
leaves each destination with one globally ordered range. The same path now
becomes external when its retained payload reaches the configured watermark;
there is no second algorithm switch.

## Ordering and supported data

- Multiple keys and directions.
- PyArrow-compatible null and floating-point NaN placement.
- Nullable flat Arrow boolean, string, integer, floating, date, time, and
  timestamp sort keys.
- Flat cuDF-compatible payload columns, including typed all-null columns.
- Deterministic byte-weighted sampling with at least 65,536 samples.
- Repeated boundaries and a hidden deterministic row token divide equal keys
  across adjacent ranges; the token is removed before output.

Nested, union, dictionary, extension columns and explicit user boundaries are
rejected in this first change.

## Memory and spill semantics

Each GPU actor uses an RMM pool with 0.50 initial and 0.85 maximum VRAM
fractions. The residency budget limits live row payload while preserving
workspace for sorting and merging. GPU-externalized Arrow runs live in Plasma;
if Plasma is pressured, Ray automatically writes them to the configured local
disk/NVMe spill directory and restores them for merge. GPU-to-Plasma
externalization, Ray-to-disk spill, and RAPIDS-MPF spill are separately
reported.

For multi-node execution, actors report their real MPF rank, Ray node ID, and
usable RMM budget. Input ObjectRefs are assigned to the GPU actor on their
Plasma node and balanced by decoded bytes. Shuffle waves are bounded from
measured device memory, while allocator-headroom-aware concatenation and run
slicing reserve the workspace needed by the final GPU sort. These are portable
production behaviors rather than a separate cloud algorithm.

The DGX harness uses `/dev/shm` only for Plasma. Actual Ray spill files use a
per-trial path on RAID and a worker rejects a spill path below `/dev/shm`.

## DGX evidence

Hardware: one DGX with 16 V100 32-GB GPUs and NVSwitch. Runtime evidence uses
the exact Ray 2.55.1 wheel at commit `237c2455eb`.

- Exact 160,000-row BTS smoke: PyArrow, resident GPU, and forced-external GPU
  produced identical schema and every row/value.
- Fixed 80,738,761-row BTS cohort:
  - narrow payload: 15.931s GPU median, 12.80x over default Ray/PyArrow;
  - core payload: 33.667s, 7.23x;
  - full 63.881-GiB payload: 53.904s, 5.00x;
  - one string, one integer, two string, and four natural keys all remained
    near 50.7-53.9s on GPU.
- Full-payload forced externalization:
  - resident: 53.904s;
  - light: 153.625s, 32 initial runs and one merge pass;
  - medium: 141.000s, the same run-geometry plateau;
  - heavy: 202.285s, 48 initial runs and one merge pass.
- A 191.642-GiB, 242,216,283-row full-payload `Origin` sort completed in
  507.708s with 136 initial runs, 51 replacement runs, two GPU merge passes,
  781.553 GiB of Ray writes to RAID, global ordering, bounded GPU memory, and
  zero CPU sort/merge/fallback.

Two GPU observations per trend/spill cell are directional, not a statistical
study. The complete artifacts and phase telemetry are in `RESULTS.md` and the
benchmark harness output.

## AWS L4 evidence

The same production backend ran on 16 retained `g6.4xlarge` L4 nodes against
default Ray/PyArrow on 16 retained `m5dn.4xlarge` nodes. Ray was restarted for
every observation without reprovisioning EC2; default Plasma was used and Ray
filesystem spill went to `/mnt/nvme`.

- The exact 160,000-row, 16-rank transport smoke matched every row/value.
- On the fixed 80,738,761-row cohort, cloud GPU medians were 7.165s narrow,
  23.135s core, and 39.129s full, for 9.64x, 5.00x, and 3.04x over the CPU
  observations.
- Full-payload GPU medians remained near-flat for `Origin`, `Origin, Dest`, and
  the four-key baseline: 40.791, 39.706, and 39.129 seconds. CPU took 82.643,
  122.760, and 118.931 seconds.
- At natural 2x size (127.762 GiB), GPU externalized every row, completed in a
  137.641-second median, and remained 3.63x faster than CPU's 499.946 seconds.
- At natural 2.45x size (156.505 GiB), GPU completed twice in 159.308 and
  162.441 seconds with 48 initial runs, 16 replacement runs, one GPU merge
  pass, and zero CPU fallback. Default PyArrow reproducibly lost the head at
  the map-to-reduce boundary, so no 2.45x speedup is claimed.

The numerical cloud comparisons are directional. Only the core cell is strict
accepted end to end; the other displayed speedups completed exact row/schema/
checksum/order validation but had incomplete Ray ObjectRef location metadata
in at least one observation and are labeled `telemetry-warning` in the report.
The PR-ready source also includes two post-run fixes that do not touch the
measured ascending BTS paths: correct direction-independent libcudf null
placement for descending keys, and aggregation/enforcement of the existing
per-rank pinned-output fallback counter (zero in every archived rank).

All campaign fleets were terminated and exact-tag scopes verified empty. The
full directional results, spill amplification, phase telemetry, warnings, and
repair provenance are in `RESULTS.md` and the generated cloud report.

## Validation

- One exact live PyArrow/resident-GPU/external-GPU equivalence smoke.
- Focused tests for comparator ordering, nullable typed output, deterministic
  equal-key spreading, inverse-inclusion sample weights, spill transition, and
  bounded merge progress, plus cloud locality, wave selection, task-output
  ownership, lifecycle, artifact provenance, and report overlays.
- Ruff, Black, compilation, diff, and allowlist checks.
- One exact DGX smoke and one exact 16-node cloud transport smoke; no broad Ray
  suite or repeated correctness matrix.

## Known limitation

Natural BTS skew completed safely, with 1.06x four-key and 1.89x single-Origin
range balance at 64 GiB and 2.00x in the three-copy Origin proof. Wave admission
is sender-bounded rather than receiver-credit-controlled. An adversarial,
unsampled all-to-one range can aggregate traffic from several senders before
the receiver externalizes it. A strict adversarial bound requires a future
receiver-credit/backpressure protocol; this PR does not claim that guarantee.
