"""End-to-end multi-GPU sort wired into Ray Data's ``ds.sort()`` path.

This is an experimental, opt-in sort backend (enabled via the
``RAY_DATA_GPU_SORT=1`` environment variable) that replaces the CPU
object-store shuffle sort with a single-pass distributed sample sort across all
local GPUs.

A single Ray actor owns every local GPU and performs:

    H2D    : pull input blocks (Arrow, in the object store / RAM) -> per-GPU
             int32 device arrays (RAM -> VRAM).
    SORT   : sample -> range-partition -> P2P all-to-all over NVLink/NVSwitch
             -> local radix sort (VRAM -> VRAM, fully on device).
    D2H    : sorted device blocks -> Arrow blocks back in the object store (RAM).

Output block ``i`` holds the i-th global key range in sorted order, so the
concatenation of output blocks is globally sorted.

Scope: this implementation targets the fixed-width integer benchmark dataset
(all columns int32, single int32 sort key, ascending, no nulls). It is not a
general replacement for the CPU sort.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import numpy as np

import ray
from ray.data._internal.execution.interfaces import RefBundle, TaskContext
from ray.data.block import BlockAccessor, BlockMetadata

# Populated on the driver after each GPU sort so a benchmark/driver can read the
# detailed phase timings and the on-GPU correctness summary.
LAST_RUN_STATS: dict = {}

_SORTER_NAME = "__ray_gpu_sorter__"


# --------------------------------------------------------------------------- #
# GPU sample-sort primitives (one process owning all GPUs; P2P over NVLink)
# --------------------------------------------------------------------------- #
def _enable_peer_access(devices):
    import cupy as cp

    for dev in devices:
        with cp.cuda.Device(dev):
            for peer in devices:
                if peer == dev:
                    continue
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(peer)
                except cp.cuda.runtime.CUDARuntimeError as e:
                    if "AlreadyEnabled" not in str(e):
                        raise


@ray.remote(num_gpus=16)
class _GpuSorter:
    """Owns all local GPUs and sorts a set of Arrow blocks end to end."""

    def __init__(self, num_gpus: int, num_cols: int, key_index: int = 0):
        import cupy as cp

        self.cp = cp
        self.devices = list(range(num_gpus))
        self.nd = num_gpus
        self.num_cols = num_cols
        self.key_index = key_index
        self.col_names: Optional[List[str]] = None
        self.pool = ThreadPoolExecutor(max_workers=num_gpus)
        _enable_peer_access(self.devices)
        self.comp_streams = []
        self.copy_streams = []
        for dev in self.devices:
            with cp.cuda.Device(dev):
                self.comp_streams.append(cp.cuda.Stream(non_blocking=True))
                self.copy_streams.append(cp.cuda.Stream(non_blocking=True))
        self.last_stats: dict = {}
        # Per-GPU page-locked (pinned) host buffers for the D2H copy, reused
        # across sorts. Pinned D2H is ~2.3x pageable on this box (~48.9 vs ~21.8
        # GiB/s aggregate) and runs async on a copy stream so all 16 devices
        # overlap. (H2D instead DMAs straight from the pageable Arrow buffers,
        # which already reach the same ~46 GiB/s aggregate ceiling as pinned, so
        # an extra host->pinned staging copy would only slow it down.)
        # Stored per slot as (capacity_rows, ndarray).
        self._pinned: List[Optional[Tuple[int, "np.ndarray"]]] = [None] * num_gpus

    def _pinned_2d(self, slot: int, dev: int, cols: int, rows: int):
        """Return a (cols, capacity) pinned host array, (re)allocating if needed.

        Sized with headroom so the same buffer is reused across trials despite
        small per-GPU row-count imbalance from the partition step.
        """
        cp = self.cp
        cached = self._pinned[slot]
        if cached is None or cached[0] < rows:
            cap = int(rows * 1.2) + 1
            with cp.cuda.Device(dev):
                mem = cp.cuda.alloc_pinned_memory(cols * cap * 4)
            buf = np.frombuffer(mem, dtype=np.int32, count=cols * cap).reshape(cols, cap)
            self._pinned[slot] = (cap, buf)
            cached = self._pinned[slot]
        return cached[1]

    # -- helpers ---------------------------------------------------------- #
    def _par(self, fn):
        futs = [self.pool.submit(fn, s, d) for s, d in enumerate(self.devices)]
        return [f.result() for f in futs]

    def _sync_all(self):
        cp = self.cp

        def _s(slot, dev):
            with cp.cuda.Device(dev):
                cp.cuda.runtime.deviceSynchronize()

        self._par(_s)

    # -- phase: H2D (RAM -> VRAM) ---------------------------------------- #
    def _load(self, tables) -> list:
        """Build one row-major int32 device array per GPU from Arrow blocks.

        Key optimization: never build a row-major host array. Each Arrow column
        buffer is already a contiguous int32 array, so we DMA it straight into a
        contiguous slice of a *column-major* device array ``Xc`` (no per-column
        strided host writes -- the old "host[:, j] = col" transpose was ~0.5
        GiB/s and dominated H2D). The cheap row<->col transpose is then done on
        the GPU (hundreds of GB/s) so the proven row-major sample sort below is
        untouched. The 16 device threads issue their copies concurrently; the
        GIL is released during each cudaMemcpy so all 16 PCIe links run at once.
        """
        cp = self.cp
        nd = self.nd
        if self.col_names is None:
            self.col_names = list(tables[0].schema.names)
        cols = self.num_cols

        # Assign input blocks to GPUs round-robin (balances uneven counts).
        assign = [[] for _ in range(nd)]
        for i, t in enumerate(tables):
            assign[i % nd].append(t)

        X = [None] * nd

        def build(slot, dev):
            with cp.cuda.Device(dev):
                my = assign[slot]
                rows = sum(t.num_rows for t in my)
                Xc = cp.empty((cols, rows), dtype=cp.int32)  # column-major
                off = 0
                for t in my:
                    n = t.num_rows
                    for j in range(cols):
                        col = t.column(j).to_numpy(zero_copy_only=False)
                        # contiguous H2D straight into column j's slice
                        Xc[j, off : off + n].set(col)
                    off += n
                # GPU-side transpose to the row-major layout the sort expects
                X[slot] = cp.ascontiguousarray(Xc.T)
                del Xc

        self._par(build)
        self._sync_all()
        return X

    # -- phase: SORT (VRAM -> VRAM) -------------------------------------- #
    def _sample_sort(self, X: list, sample_per_gpu: int = 1 << 14) -> list:
        cp = self.cp
        nd = self.nd
        cols = self.num_cols
        ki = self.key_index
        itemsize = 4 * cols

        # phase 0: global quantile splitters from a key sample
        def sample(slot, dev):
            with cp.cuda.Device(dev):
                keys = X[slot][:, ki]
                stride = max(1, keys.shape[0] // sample_per_gpu)
                return cp.asnumpy(keys[::stride])

        samples = self._par(sample)
        alls = np.concatenate(samples)
        alls.sort()
        qs = [k / nd for k in range(1, nd)]
        splitters = np.quantile(alls, qs).astype(np.int32)
        for i in range(1, len(splitters)):
            if splitters[i] <= splitters[i - 1]:
                splitters[i] = splitters[i - 1] + 1

        # phase 1: partition rows into destination buckets (grouped contiguous)
        Xs = [None] * nd
        counts = [None] * nd

        def partition(slot, dev):
            with cp.cuda.Device(dev), self.comp_streams[slot]:
                sp = cp.asarray(splitters)
                keys = cp.ascontiguousarray(X[slot][:, ki])
                bucket = cp.searchsorted(sp, keys, side="right").astype(cp.int32)
                perm = cp.argsort(bucket)
                Xs[slot] = X[slot][perm]
                counts[slot] = cp.asnumpy(cp.bincount(bucket, minlength=nd))

        self._par(partition)

        # host-side exchange plan
        M = np.stack(counts).astype(np.int64)  # M[i, j] = rows src i -> dst j
        send_off = np.zeros((nd, nd + 1), dtype=np.int64)
        send_off[:, 1:] = np.cumsum(M, axis=1)
        recv_off = np.zeros((nd, nd + 1), dtype=np.int64)
        recv_off[:, 1:] = np.cumsum(M.T, axis=1)
        recv_tot = M.sum(axis=0)

        R = [None] * nd

        def alloc_recv(slot, dev):
            with cp.cuda.Device(dev):
                R[slot] = cp.empty((int(recv_tot[slot]), cols), dtype=cp.int32)

        self._par(alloc_recv)

        # phase 2: all-to-all P2P exchange over NVLink/NVSwitch
        def exchange(slot, dev):
            i = slot
            with cp.cuda.Device(dev):
                st = self.copy_streams[i]
                base = Xs[i].data.ptr
                for j in range(nd):
                    n = int(M[i, j])
                    if n == 0:
                        continue
                    nbytes = n * itemsize
                    src = base + int(send_off[i, j]) * itemsize
                    dst = R[j].data.ptr + int(recv_off[j, i]) * itemsize
                    cp.cuda.runtime.memcpyPeerAsync(dst, self.devices[j], src, dev, nbytes, st.ptr)
                st.synchronize()

        self._par(exchange)

        # phase 3: local radix sort of received bucket
        S = [None] * nd

        def local_sort(slot, dev):
            with cp.cuda.Device(dev), self.comp_streams[slot]:
                keys = cp.ascontiguousarray(R[slot][:, ki])
                kp = cp.argsort(keys)
                S[slot] = R[slot][kp]
                self.comp_streams[slot].synchronize()

        self._par(local_sort)
        return S

    # -- on-GPU correctness summary (cheap; no host round trip) ----------- #
    def _verify(self, S: list) -> dict:
        cp = self.cp
        ki = self.key_index

        def per(slot, dev):
            with cp.cuda.Device(dev):
                k = S[slot][:, ki]
                n = int(k.shape[0])
                if n == 0:
                    return (0, None, None, True, 0)
                asc = bool(cp.all(k[1:] >= k[:-1]).item())
                ksum = int(k.astype(cp.int64).sum().item())
                return (n, int(k[0].item()), int(k[-1].item()), asc, ksum)

        res = self._par(per)
        total = sum(r[0] for r in res)
        all_asc = all(r[3] for r in res)
        ksum = sum(r[4] for r in res)
        kmin = min(r[1] for r in res if r[0] > 0)
        kmax = max(r[2] for r in res if r[0] > 0)
        boundaries_ok = True
        prev = None
        for r in res:
            if r[0] == 0:
                continue
            if prev is not None and r[1] < prev:
                boundaries_ok = False
            prev = r[2]
        return {
            "rows": total,
            "globally_sorted": all_asc and boundaries_ok,
            "key_sum": ksum,
            "key_min": kmin,
            "key_max": kmax,
            "block_rows": [r[0] for r in res],
        }

    # -- phase: D2H (VRAM -> RAM) ---------------------------------------- #
    def _store(self, S: list):
        """Sorted device blocks -> Arrow tables in host RAM.

        Mirror of ``_load``: transpose to column-major on the GPU, then DMA each
        contiguous column straight into a reused *pinned* host buffer on a copy
        stream (pinned D2H is ~2.3x pageable here and runs async so the 16
        devices overlap). Each host column is wrapped zero-copy into an Arrow
        array -- no per-column ``np.ascontiguousarray`` host transpose.
        """
        cp = self.cp
        import pyarrow as pa

        names = self.col_names
        cols = self.num_cols

        def to_table(slot, dev):
            with cp.cuda.Device(dev):
                blk = S[slot]  # (rows_out, cols) row-major
                rows_out = int(blk.shape[0])
                Sc = cp.ascontiguousarray(blk.T)  # (cols, rows_out) column-major
                pin = self._pinned_2d(slot, dev, cols, rows_out)
                st = self.copy_streams[slot]
                for j in range(cols):
                    Sc[j].get(out=pin[j, :rows_out], stream=st)  # async pinned D2H
                st.synchronize()
            # Zero-copy views of the pinned columns; ray.put() copies into the
            # object store before this buffer is reused on the next sort.
            return pa.table({names[j]: pa.array(pin[j, :rows_out]) for j in range(cols)})

        return self._par(to_table)

    # -- public entrypoint ------------------------------------------------ #
    def sort(self, block_refs):
        """Sort the given object-store blocks; returns (out_blocks, schema, stats).

        block_refs is a list of ObjectRef[Block]; the actor fetches them directly
        from the (same-node) object store, so no data is double-copied through the
        calling worker. out_blocks is a list of (ObjectRef[Block], BlockMetadata)
        in global sorted order.
        """
        t0 = time.perf_counter()
        tables = ray.get(block_refs)  # zero-copy from shared-memory object store
        X = self._load(tables)
        t1 = time.perf_counter()

        self._sync_all()
        ts0 = time.perf_counter()
        S = self._sample_sort(X)
        self._sync_all()
        ts1 = time.perf_counter()

        verify = self._verify(S)

        out_tables = self._store(S)
        t2 = time.perf_counter()

        out_blocks = []
        schema = None
        for tbl in out_tables:
            ref = ray.put(tbl)
            meta = BlockAccessor.for_block(tbl).get_metadata()
            if schema is None:
                schema = BlockAccessor.for_block(tbl).schema()
            out_blocks.append((ref, meta))

        stats = {
            "h2d_s": t1 - t0,
            "gpu_sort_s": ts1 - ts0,
            "d2h_s": t2 - ts1,
            "full_s": t2 - t0,  # RAM -> VRAM -> sort -> VRAM -> RAM
            "verify": verify,
        }
        self.last_stats = stats
        return out_blocks, schema, stats

    def get_last_stats(self) -> dict:
        return self.last_stats


def _get_sorter(num_gpus: int, num_cols: int):
    # Detached so the actor survives the short-lived worker that runs the
    # all-to-all transform fn, and stays queryable by the driver (for stats)
    # and reusable across trials.
    return _GpuSorter.options(
        name=_SORTER_NAME,
        get_if_exists=True,
        lifetime="detached",
        num_gpus=num_gpus,
    ).remote(num_gpus=num_gpus, num_cols=num_cols)


# --------------------------------------------------------------------------- #
# Ray Data transform fn (the ds.sort() hook)
# --------------------------------------------------------------------------- #
def generate_gpu_sort_fn(sort_key, data_context, num_gpus: int = 16):
    """Return an AllToAllTransformFn that sorts all input blocks on the GPUs."""

    def fn(refs: List[RefBundle], ctx: TaskContext) -> Tuple[List[RefBundle], dict]:
        block_refs = []
        schema_in = None
        for rb in refs:
            schema_in = schema_in or rb.schema
            block_refs.extend(rb.block_refs)
        if len(block_refs) == 0:
            return refs, {"GPUSort": []}

        num_cols = len(schema_in.names)
        sorter = _get_sorter(num_gpus, num_cols)
        out_blocks, schema_out, stats = ray.get(sorter.sort.remote(block_refs))

        LAST_RUN_STATS.clear()
        LAST_RUN_STATS.update(stats)

        out = []
        stats_list = []
        for ref, meta in out_blocks:
            stats_list.append(meta.to_stats())
            out.append(RefBundle([(ref, meta)], owns_blocks=True, schema=schema_out))
        return out, {"GPUSort": stats_list}

    return fn
