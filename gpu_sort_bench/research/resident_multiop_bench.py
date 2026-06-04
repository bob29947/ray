"""Resident multi-op microbenchmark -- the "why keep it in VRAM" evidence.

This makes the amortization argument for the general GPU sort *empirically*,
on the SAME 64 GiB dataset, with 16 one-GPU rapidsmpf actors (4 GiB/GPU). The
full end-to-end sort pays, per call:

    FULL = H2D + GPU_only + D2H        (and H2D + D2H dominate FULL)

If Ray kept blocks device-resident across operators, H2D would be paid once at
ingest and D2H once at output, so each chained GPU op would run at ~GPU_only.
This bench demonstrates exactly that by running K sorts two ways:

  resident:    H2D ONCE (host Arrow -> resident cuDF), then K device-resident
               sorts back-to-back WITHOUT round-tripping to host (the key is
               re-sorted ascending/descending alternately so every op is a real
               full re-sort), then D2H ONCE. Per-op work -> GPU_only.
               Amortized per-op wall = (H2D + K*GPU_only + D2H) / K  ->  GPU_only.

  host_staged: K sorts where EACH op pays H2D (Arrow->cuDF) + device sort + D2H
               (cuDF->Arrow), i.e. the round-trip is repaid every op.
               Per-op wall -> FULL, independent of K.

The device sort is the SAME pipeline the general engine uses: local cuDF
``sort_by_key`` -> range-partition (``split_and_pack``) -> rapidsmpf Shuffler
all-to-all over NVLink -> ``unpack_and_concat`` -> final per-range sort. Boundaries
are fixed, evenly spaced int32 values (the key is uniform random int32, so this
range-partitions into balanced, globally ordered partitions). The bench also
checks the resident result is globally sorted across ranks.

Run:
    .venv/bin/python resident_multiop_bench.py                 # 64 GiB, K=8
    .venv/bin/python resident_multiop_bench.py --k 8 --gpus 16
    .venv/bin/python resident_multiop_bench.py --rows $((1024*1024*256))  # 16 GiB
"""

from __future__ import annotations

import argparse
import json
import time
from typing import List


def gib(nbytes: int) -> str:
    return f"{nbytes / 2**30:.2f} GiB"


def _build_resident_actor():
    import os

    import ray
    from rapidsmpf.utils.ray_utils import BaseShufflingActor

    @ray.remote(num_gpus=1, num_cpus=4)
    class _ResidentSorter(BaseShufflingActor):
        """Owns ONE GPU; holds its 1/N slice of the dataset both as a host Arrow
        table (for host-staged H2D every op) and as a resident cuDF table (loaded
        once for the resident loop)."""

        def __init__(self, nranks: int):
            # setup_ray_ucxx_cluster passes only nranks to every actor, so the
            # per-rank index is supplied later (build_host); seeds the slice.
            super().__init__(nranks)
            # Force UCX onto NVLink/CUDA-IPC (same config as the general engine).
            try:
                from ray.data._internal.planner.gpu_sort_general import _ucx_env
                for k, v in _ucx_env().items():
                    os.environ.setdefault(k, v)
            except Exception:
                os.environ.setdefault("UCX_TLS", "cuda_copy,cuda_ipc,sm,tcp")
                os.environ.setdefault("UCX_SOCKADDR_TLS_PRIORITY", "tcp")
                os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
            self.index = None
            self.cols = None
            self.host_tbl = None       # pyarrow table in host RAM (the input)
            self.df = None             # resident cuDF table (H2D paid once)
            self.last = None           # last resident sort result (kept on device)
            self.br = None
            self._mr = None
            self._bnd_cache = {}

        # -- per-process device setup (RMM pool + buffer resource) -------- #
        def setup_worker(self, root_address_bytes: bytes) -> None:
            super().setup_worker(root_address_bytes)
            import rmm
            from rapidsmpf.memory.buffer_resource import (
                BufferResource,
                LimitAvailableMemory,
            )
            from rapidsmpf.memory.buffer import MemoryType
            from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor

            total = rmm.mr.available_device_memory()[1]
            self._mr = RmmResourceAdaptor(
                rmm.mr.PoolMemoryResource(
                    rmm.mr.CudaMemoryResource(),
                    initial_pool_size=(int(total * 0.5) // 256) * 256,
                    maximum_pool_size=(int(total * 0.8) // 256) * 256,
                )
            )
            rmm.mr.set_current_device_resource(self._mr)
            # Keeping the input resident across ops costs an extra slice of VRAM
            # vs the production engine (which frees its input mid-sort), so allow
            # device->host spill past a high-water mark as cheap insurance.
            limit = int(total * 0.75)
            br_avail = {MemoryType.DEVICE: LimitAvailableMemory(self._mr, limit=limit)}
            self.br = BufferResource(self._mr, memory_available=br_avail)

        def is_ready(self) -> bool:
            return self.is_initialized() and self.br is not None

        # -- build the host-resident input slice (untimed) ---------------- #
        def build_host(self, rows_per_gpu: int, cols: int, index: int) -> int:
            import numpy as np
            import pyarrow as pa

            self.index = index
            self.cols = cols
            # Distinct seed per rank so the 16 slices together look like one
            # uniform-random int32 key column (same family as the main dataset).
            rng = np.random.default_rng(1000 + self.index)
            data = {"c0": rng.integers(0, 2**31 - 1, rows_per_gpu, dtype=np.int32)}
            for j in range(1, cols):
                data[f"c{j}"] = np.zeros(rows_per_gpu, dtype=np.int32)
            self.host_tbl = pa.table(data)
            return self.host_tbl.nbytes

        # -- fixed, evenly spaced int32 range-partition boundaries -------- #
        def _boundaries(self, ascending: bool):
            import cudf
            import numpy as np

            key = ("asc" if ascending else "desc")
            if key in self._bnd_cache:
                return self._bnd_cache[key]
            nranks = self.nranks()
            step = (2**31) / nranks
            vals = np.array([int((j + 1) * step) for j in range(nranks - 1)],
                            dtype=np.int32)
            if not ascending:
                vals = vals[::-1].copy()
            bnd = cudf.DataFrame({"c0": vals})
            self._bnd_cache[key] = bnd
            return bnd

        # -- the device-resident sort pipeline (same phases as the engine) - #
        def _device_sort(self, df, ascending: bool, op_id: int):
            import cudf
            import cupy as cp
            import pylibcudf as plc
            from rmm.pylibrmm.stream import DEFAULT_STREAM
            from rapidsmpf.integrations.cudf.partition import (
                split_and_pack,
                unpack_and_concat,
                unspill_partitions,
            )
            from rapidsmpf.utils.cudf import (
                cudf_to_pylibcudf_table,
                pylibcudf_to_cudf_dataframe,
            )
            from ray.data._internal.planner.gpu_sort_general import _plc_order_null

            nranks = self.nranks()
            names = list(df.columns)
            key_idx = [names.index("c0")]
            order, null_prec = _plc_order_null([ascending], "last")

            def _sort(table):
                keys = plc.Table([table.columns()[i] for i in key_idx])
                return plc.sorting.sort_by_key(table, keys, order, null_prec)

            # local sort
            full = cudf_to_pylibcudf_table(df)
            srt = _sort(full)
            del full

            # range-partition by fixed global boundaries
            bnd = self._boundaries(ascending)
            ds_keys = plc.Table([srt.columns()[i] for i in key_idx])
            bnd_keys = cudf_to_pylibcudf_table(bnd)
            splits_col = plc.search.lower_bound(ds_keys, bnd_keys, order, null_prec)
            splits = cudf.Series.from_pylibcudf(splits_col).to_numpy().tolist()
            n_rows = srt.num_rows()
            splits = [int(min(max(s, 0), n_rows)) for s in splits]
            while len(splits) < nranks - 1:
                splits.append(n_rows)
            splits = splits[: nranks - 1]
            packed = split_and_pack(srt, splits, DEFAULT_STREAM, self.br)
            del srt

            # rapidsmpf all-to-all shuffle over NVLink
            shuffler = self.create_shuffler(
                op_id, total_num_partitions=nranks, buffer_resource=self.br
            )
            shuffler.insert_chunks(packed)
            shuffler.insert_finished(list(range(nranks)))
            outputs = {}
            while not shuffler.finished():
                pid = shuffler.wait_any()
                chunks = shuffler.extract(pid)
                tbl = unpack_and_concat(
                    unspill_partitions(chunks, br=self.br, allow_overbooking=True),
                    DEFAULT_STREAM,
                    self.br,
                )
                outputs[pid] = tbl
            shuffler.shutdown()

            # final sort of each received key range -> this rank's sorted range
            parts = []
            for pid in sorted(outputs):
                parts.append(pylibcudf_to_cudf_dataframe(_sort(outputs[pid]), names))
            res = parts[0] if len(parts) == 1 else cudf.concat(parts, ignore_index=True)
            return res

        # -- H2D once: host Arrow -> resident cuDF ------------------------ #
        def load_resident(self) -> float:
            import cudf
            import cupy as cp

            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            self.df = cudf.DataFrame.from_arrow(self.host_tbl)
            cp.cuda.runtime.deviceSynchronize()
            return time.perf_counter() - t0

        # -- one RESIDENT op: sort the resident self.df, keep result on GPU - #
        def resident_op(self, op_id: int, ascending: bool) -> float:
            import cupy as cp

            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            self.last = self._device_sort(self.df, ascending, op_id)
            cp.cuda.runtime.deviceSynchronize()
            return time.perf_counter() - t0

        # -- D2H once: last resident result -> host Arrow ----------------- #
        def store_resident(self) -> float:
            import cupy as cp

            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            _ = self.last.to_arrow()
            cp.cuda.runtime.deviceSynchronize()
            return time.perf_counter() - t0

        # -- one HOST-STAGED op: H2D + device sort + D2H every time ------- #
        def host_staged_op(self, op_id: int, ascending: bool) -> float:
            import cudf
            import cupy as cp

            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            df = cudf.DataFrame.from_arrow(self.host_tbl)   # H2D
            res = self._device_sort(df, ascending, op_id)   # VRAM->VRAM
            _ = res.to_arrow()                              # D2H
            cp.cuda.runtime.deviceSynchronize()
            return time.perf_counter() - t0

        # -- bonus: prove the resident result is globally sorted ---------- #
        def range_summary(self) -> dict:
            # The last resident op sorted ASCENDING (we end on ascending), so the
            # per-rank min/max let the driver assert min(rank r+1) >= max(rank r).
            df = self.last
            c0 = df["c0"]
            n = int(len(df))
            mono = bool((c0.iloc[1:].reset_index(drop=True)
                         >= c0.iloc[:-1].reset_index(drop=True)).all()) if n > 1 else True
            return {"index": self.index, "n": n,
                    "min": int(c0.min()) if n else None,
                    "max": int(c0.max()) if n else None,
                    "monotonic": mono}

        def release(self) -> None:
            self.df = None
            self.last = None

    return _ResidentSorter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024 * 1024 * 1024,
                   help="total rows across all GPUs (default 1Gi -> 64 GiB)")
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--k", type=int, default=8, help="number of chained sorts")
    p.add_argument("--json", action="store_true", help="emit RESULT_JSON line")
    args = p.parse_args()

    import ray
    from rapidsmpf.integrations.ray import setup_ray_ucxx_cluster

    nd = args.gpus
    assert args.rows % nd == 0, "rows must split evenly across GPUs"
    rows_per_gpu = args.rows // nd
    assert rows_per_gpu % nd == 0, "rows_per_gpu must split evenly into nd chunks"
    raw_bytes = args.rows * args.cols * 4

    print(f"=== resident multi-op bench: {gib(raw_bytes)} "
          f"({args.rows:,} rows x {args.cols} int32), {nd} GPUs, K={args.k} ===",
          flush=True)

    ray.init(object_store_memory=32 * 2**30)

    cls = _build_resident_actor()
    actors = setup_ray_ucxx_cluster(cls, nd)
    nbytes = ray.get([a.build_host.remote(rows_per_gpu, args.cols, i)
                      for i, a in enumerate(actors)])
    print(f"host input resident: {nd} x {gib(rows_per_gpu * args.cols * 4)} "
          f"= {gib(sum(nbytes))}", flush=True)

    op = [0]

    def step():
        op[0] += 1
        return op[0] % 256

    # ----- warmup (not counted): one resident load + 2 ops each mode ----- #
    print("warmup...", flush=True)
    ray.get([a.load_resident.remote() for a in actors])
    for asc in (True, False):
        oid = step()
        ray.get([a.resident_op.remote(oid, asc) for a in actors])
    oid = step()
    ray.get([a.host_staged_op.remote(oid, True) for a in actors])
    ray.get([a.release.remote() for a in actors])

    # ===== RESIDENT: H2D once, K device-resident sorts, D2H once ========= #
    print(f"\n[resident] H2D once -> {args.k} resident sorts -> D2H once",
          flush=True)
    h2d = max(ray.get([a.load_resident.remote() for a in actors]))
    resident_ops = []
    for i in range(args.k):
        asc = (i % 2 == 0)  # alternate direction so each op is a real re-sort
        oid = step()
        dt = max(ray.get([a.resident_op.remote(oid, asc) for a in actors]))
        resident_ops.append(dt)
        print(f"  resident op {i + 1}/{args.k} ({'asc' if asc else 'desc'}): "
              f"{dt * 1e3:8.1f} ms", flush=True)
    # Uncounted normalization: leave the resident result ascending so the
    # cross-rank global-sortedness check below is unambiguous (not in the K-op
    # timing; it's the same resident sort, just pinned to one direction).
    oid = step()
    ray.get([a.resident_op.remote(oid, True) for a in actors])
    d2h = max(ray.get([a.store_resident.remote() for a in actors]))

    # Global-sortedness check on the resident (ascending) result. Each rank
    # holds ONE global key range (the partition matching its rapidsmpf rank, not
    # its build order), so order the per-rank ranges by their min value and
    # verify they are each internally sorted AND non-overlapping (max[i] <=
    # min[i+1]) -- that is exactly "the concatenation in key order is sorted".
    summ = ray.get([a.range_summary.remote() for a in actors])
    total_rows = sum(s["n"] for s in summ)
    all_mono = all(s["monotonic"] for s in summ)
    nonempty = sorted((s for s in summ if s["n"] > 0), key=lambda s: s["min"])
    cross_ok = True
    prev_max = None
    for s in nonempty:
        if prev_max is not None and s["min"] < prev_max:
            cross_ok = False
        prev_max = s["max"]
    correct = all_mono and cross_ok and (total_rows == args.rows)
    print(f"[resident] result rows={total_rows:,} ({'ok' if total_rows == args.rows else 'BAD'})  "
          f"globally_sorted={'PASS' if (all_mono and cross_ok) else 'FAIL'}", flush=True)

    # ===== HOST-STAGED: H2D + sort + D2H every op ======================== #
    print(f"\n[host_staged] {args.k} sorts, each pays H2D + sort + D2H",
          flush=True)
    ray.get([a.release.remote() for a in actors])
    staged_ops = []
    for i in range(args.k):
        asc = (i % 2 == 0)
        oid = step()
        dt = max(ray.get([a.host_staged_op.remote(oid, asc) for a in actors]))
        staged_ops.append(dt)
        print(f"  host_staged op {i + 1}/{args.k} ({'asc' if asc else 'desc'}): "
              f"{dt * 1e3:8.1f} ms", flush=True)

    # ----- summary + the amortization curve ------------------------------ #
    gpu_only = min(resident_ops)                  # best device-resident sort
    gpu_only_med = sorted(resident_ops)[len(resident_ops) // 2]
    staged = min(staged_ops)                      # best host-staged (= FULL)
    print("\n" + "=" * 78)
    print(f"  dataset {gib(raw_bytes)}, {nd} GPUs, K={args.k}")
    print(f"  H2D (once) : {h2d * 1e3:8.1f} ms")
    print(f"  D2H (once) : {d2h * 1e3:8.1f} ms")
    print(f"  GPU-only per resident sort : best {gpu_only * 1e3:7.1f} ms   "
          f"median {gpu_only_med * 1e3:7.1f} ms")
    print(f"  FULL per host-staged sort  : best {staged * 1e3:7.1f} ms   "
          f"(= H2D + GPU-only + D2H, repaid every op)")
    print("-" * 78)
    print("  amortized resident wall/op = (H2D + N*GPU_only + D2H) / N   "
          "vs host-staged FULL/op")
    print(f"  {'N':>3} | {'resident /op':>14} | {'host-staged /op':>16} | {'speedup':>8}")
    for N in range(1, args.k + 1):
        amort = (h2d + N * gpu_only + d2h) / N
        spd = staged / amort
        print(f"  {N:>3} | {amort * 1e3:11.1f} ms | {staged * 1e3:13.1f} ms | "
              f"{spd:6.2f}x")
    print(f"  N->inf converges to GPU-only = {gpu_only * 1e3:.1f} ms "
          f"({staged / gpu_only:.1f}x vs host-staged FULL)")
    print("=" * 78 + "\n")

    if args.json:
        print("RESULT_JSON:" + json.dumps({
            "rows": args.rows, "gpus": nd, "k": args.k,
            "h2d_s": h2d, "d2h_s": d2h,
            "gpu_only_best_s": gpu_only, "gpu_only_median_s": gpu_only_med,
            "full_staged_best_s": staged,
            "resident_ops_s": resident_ops, "staged_ops_s": staged_ops,
            "correct": bool(correct), "rows_out": total_rows,
        }), flush=True)

    try:
        from ray.data._internal.planner.gpu_sort_general import kill_actor_pool  # noqa
    except Exception:
        pass
    ray.get([a.release.remote() for a in actors])
    for a in actors:
        try:
            ray.kill(a)
        except Exception:
            pass
    ray.shutdown()


if __name__ == "__main__":
    main()
