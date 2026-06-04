"""
End-to-end multi-GPU sort designed to crush the Ray Data CPU sort baseline.

Timing contract (same idea as sort.py / gpu_sort.py):
    t0 = input blocks already resident in GPU memory
    t1 = globally sorted blocks resident in GPU memory (all GPUs synchronized)

Algorithm: single-pass distributed sample sort across N GPUs on one NVSwitch node.

    phase 0  sample keys on every GPU  -> N-1 global quantile splitters (N buckets)
    phase 1  partition each GPU's rows into N destination buckets (grouped contiguous)
    phase 2  all-to-all exchange of bucket slices via cudaMemcpyPeerAsync over NVLink
    phase 3  local radix sort of each GPU's received bucket

After phase 3 GPU j holds the j-th global key range in sorted order, so the
concatenation block0..blockN-1 is globally sorted.

Each phase is driven by one worker thread per GPU so device work overlaps; the
CUDA calls release the GIL, so the 16 devices run their kernels concurrently.
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cupy as cp


def gib(n):
    return f"{n / 2**30:.2f} GiB"


def list_devices(n):
    return list(range(n))


def enable_peer_access(devices):
    """Enable bidirectional P2P access between every pair of GPUs."""
    enabled = 0
    for dev in devices:
        with cp.cuda.Device(dev):
            for peer in devices:
                if peer == dev:
                    continue
                if not cp.cuda.runtime.deviceCanAccessPeer(dev, peer):
                    raise RuntimeError(f"GPU {dev} cannot peer-access GPU {peer}")
                try:
                    cp.cuda.runtime.deviceEnablePeerAccess(peer)
                    enabled += 1
                except cp.cuda.runtime.CUDARuntimeError as e:
                    # 704 == cudaErrorPeerAccessAlreadyEnabled, which is fine.
                    if "AlreadyEnabled" not in str(e):
                        raise
    return enabled


class Workers:
    """One pinned worker thread per GPU so phases overlap across devices."""

    def __init__(self, devices):
        self.devices = devices
        self.pool = ThreadPoolExecutor(max_workers=len(devices))

    def run(self, fn):
        """Run fn(slot, dev) on every GPU in parallel, return results in slot order."""
        futs = [self.pool.submit(fn, slot, dev) for slot, dev in enumerate(self.devices)]
        return [f.result() for f in futs]

    def close(self):
        self.pool.shutdown(wait=True)


def build_inputs(devices, rows_per_gpu, cols, workers, seed=0, check=False):
    """Materialize one input block per GPU (column 0 random int32, rest zero).

    In --check mode column 1 is set equal to the key so we can verify rows
    travel together through the shuffle.
    """
    X = [None] * len(devices)

    def make(slot, dev):
        with cp.cuda.Device(dev):
            rs = cp.random.RandomState(seed + dev)
            x = cp.zeros((rows_per_gpu, cols), dtype=cp.int32)
            x[:, 0] = rs.randint(0, 2**31 - 1, size=rows_per_gpu, dtype=cp.int32)
            if check and cols > 1:
                x[:, 1] = x[:, 0]
            cp.cuda.runtime.deviceSynchronize()
            X[slot] = x

    workers.run(make)
    return X


def sync_all(devices, workers):
    def _sync(slot, dev):
        with cp.cuda.Device(dev):
            cp.cuda.runtime.deviceSynchronize()

    workers.run(_sync)


def compute_splitters(X, devices, workers, n_buckets, sample_per_gpu):
    """Sample keys from every GPU and derive N-1 global quantile splitters."""

    def sample(slot, dev):
        with cp.cuda.Device(dev):
            keys = X[slot][:, 0]
            n = keys.shape[0]
            stride = max(1, n // sample_per_gpu)
            s = keys[::stride]
            return cp.asnumpy(s)

    samples = workers.run(sample)
    alls = np.concatenate(samples)
    alls.sort()
    qs = [k / n_buckets for k in range(1, n_buckets)]
    splitters = np.quantile(alls, qs).astype(np.int32)
    # Force strictly increasing splitters so searchsorted yields all N buckets.
    for i in range(1, len(splitters)):
        if splitters[i] <= splitters[i - 1]:
            splitters[i] = splitters[i - 1] + 1
    return splitters


def run_sort(X, devices, workers, n_buckets, cols, sample_per_gpu, comp_streams, copy_streams):
    """The timed region: sample -> partition -> all-to-all -> local sort.

    Each phase ends with a per-device sync inside its worker callable, so the
    host-side perf_counter deltas are an accurate phase breakdown.
    """
    nd = len(devices)
    itemsize = 4 * cols
    phase = {}

    # ---- phase 0: splitters ------------------------------------------------
    s = time.perf_counter()
    splitters_host = compute_splitters(X, devices, workers, n_buckets, sample_per_gpu)
    phase["sample"] = time.perf_counter() - s

    # ---- phase 1: partition rows into destination buckets ------------------
    Xs = [None] * nd
    counts = [None] * nd

    def partition(slot, dev):
        with cp.cuda.Device(dev), comp_streams[slot]:
            sp = cp.asarray(splitters_host)
            keys = cp.ascontiguousarray(X[slot][:, 0])
            bucket = cp.searchsorted(sp, keys, side="right").astype(cp.int32)
            perm = cp.argsort(bucket)
            Xs[slot] = X[slot][perm]
            cnt = cp.bincount(bucket, minlength=n_buckets)
            host_cnt = cp.asnumpy(cnt)  # syncs this device's stream only
            counts[slot] = host_cnt

    s = time.perf_counter()
    workers.run(partition)
    phase["partition"] = time.perf_counter() - s

    # ---- host-side exchange plan ------------------------------------------
    # M[i, j] = #rows on src i destined for dst j
    M = np.stack(counts).astype(np.int64)
    send_off = np.zeros((nd, nd + 1), dtype=np.int64)
    send_off[:, 1:] = np.cumsum(M, axis=1)
    recv_off = np.zeros((nd, nd + 1), dtype=np.int64)
    recv_off[:, 1:] = np.cumsum(M.T, axis=1)  # recv_off[j, i]
    recv_tot = M.sum(axis=0)

    # ---- phase 2: all-to-all P2P exchange over NVLink ----------------------
    R = [None] * nd

    def alloc_recv(slot, dev):
        with cp.cuda.Device(dev):
            R[slot] = cp.empty((int(recv_tot[slot]), cols), dtype=cp.int32)

    def exchange(slot, dev):
        i = slot
        with cp.cuda.Device(dev):
            st = copy_streams[i]
            base_src = Xs[i].data.ptr
            for j in range(nd):
                n = int(M[i, j])
                if n == 0:
                    continue
                nbytes = n * itemsize
                src_ptr = base_src + int(send_off[i, j]) * itemsize
                dst_ptr = R[j].data.ptr + int(recv_off[j, i]) * itemsize
                cp.cuda.runtime.memcpyPeerAsync(dst_ptr, devices[j], src_ptr, dev, nbytes, st.ptr)
            st.synchronize()

    s = time.perf_counter()
    workers.run(alloc_recv)
    workers.run(exchange)
    phase["exchange"] = time.perf_counter() - s

    # ---- phase 3: local sort of received bucket ----------------------------
    S = [None] * nd

    def local_sort(slot, dev):
        with cp.cuda.Device(dev), comp_streams[slot]:
            keys = cp.ascontiguousarray(R[slot][:, 0])
            kp = cp.argsort(keys)
            S[slot] = R[slot][kp]
            comp_streams[slot].synchronize()

    s = time.perf_counter()
    workers.run(local_sort)
    phase["sort"] = time.perf_counter() - s

    return S, M, phase


def make_streams(devices):
    comp, copy = [], []
    for dev in devices:
        with cp.cuda.Device(dev):
            comp.append(cp.cuda.Stream(non_blocking=True))
            copy.append(cp.cuda.Stream(non_blocking=True))
    return comp, copy


def verify(S, devices, workers, rows_total, cols, check_payload):
    """Untimed correctness checks on the sorted blocks."""
    nd = len(devices)

    def per_block(slot, dev):
        with cp.cuda.Device(dev):
            block = S[slot]
            keys = block[:, 0]
            n = int(keys.shape[0])
            if n == 0:
                return (n, None, None, True, True)
            ascending = bool(cp.all(keys[1:] >= keys[:-1]).item())
            payload_ok = True
            if check_payload and cols > 1:
                payload_ok = bool(cp.all(block[:, 1] == block[:, 0]).item())
            return (n, int(keys[0].item()), int(keys[-1].item()), ascending, payload_ok)

    results = workers.run(per_block)

    total = sum(r[0] for r in results)
    all_sorted = all(r[3] for r in results)
    payload_ok = all(r[4] for r in results)
    boundaries_ok = True
    prev_max = None
    for r in results:
        if r[0] == 0:
            continue
        if prev_max is not None and r[1] < prev_max:
            boundaries_ok = False
        prev_max = r[2]

    ok = total == rows_total and all_sorted and boundaries_ok and payload_ok
    print("\n=== correctness ===")
    print(f"rows out: {total:,} (expected {rows_total:,}) -> {'OK' if total == rows_total else 'MISMATCH'}")
    print(f"each block ascending: {'OK' if all_sorted else 'FAIL'}")
    print(f"cross-block boundaries: {'OK' if boundaries_ok else 'FAIL'}")
    if check_payload:
        print(f"payload travels with key: {'OK' if payload_ok else 'FAIL'}")
    print(f"block row counts: {[r[0] for r in results]}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=1024 * 1024 * 1024)
    p.add_argument("--cols", type=int, default=16)
    p.add_argument("--gpus", type=int, default=16)
    p.add_argument("--buckets", type=int, default=None, help="default: one bucket per GPU")
    p.add_argument("--sample-per-gpu", type=int, default=1 << 14)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--baseline-seconds", type=float, default=45.691,
                   help="Ray Data sort+materialize baseline to compare against")
    p.add_argument("--check", action="store_true", help="small-scale correctness run")
    args = p.parse_args()

    if args.check:
        args.rows = min(args.rows, 1 << 22)
        args.trials = 1

    devices = list_devices(args.gpus)
    n_buckets = args.buckets or len(devices)
    assert n_buckets == len(devices), "this design maps one bucket per GPU"
    assert args.rows % args.gpus == 0, "rows must divide evenly across GPUs"
    rows_per_gpu = args.rows // args.gpus
    raw_bytes = args.rows * args.cols * 4

    print("=== gpu_isosort: end-to-end multi-GPU sample sort ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"cupy: {cp.__version__}")
    print(f"gpus: {args.gpus}")
    print(f"rows: {args.rows:,}")
    print(f"cols: {args.cols}")
    print(f"rows per gpu: {rows_per_gpu:,}")
    print(f"raw dataset size: {gib(raw_bytes)}")
    print(f"raw block size: {gib(rows_per_gpu * args.cols * 4)}")
    print(f"buckets: {n_buckets}")
    print(f"sort key: c0 (int32)")

    for dev in devices:
        with cp.cuda.Device(dev):
            free, total = cp.cuda.runtime.memGetInfo()
            name = cp.cuda.runtime.getDeviceProperties(dev)["name"].decode()
            print(f"gpu {dev}: {name} free={gib(free)} total={gib(total)}")

    workers = Workers(devices)
    n_peer = enable_peer_access(devices)
    print(f"\npeer-access links enabled: {n_peer}")

    comp_streams, copy_streams = make_streams(devices)

    print("\nmaterializing input blocks on GPUs...")
    X = build_inputs(devices, rows_per_gpu, args.cols, workers, check=args.check)
    sync_all(devices, workers)
    print("input resident on all GPUs.")

    # Warm up: primes the cupy memory pool and cub/thrust workspaces so the
    # timed trials reflect steady-state (matching Ray's already-warm workers),
    # not first-touch cudaMalloc syncs. The pool is intentionally NOT freed so
    # the output buffers are reused across trials.
    print("\nwarmup trial...")
    S, M, _ = run_sort(X, devices, workers, n_buckets, args.cols, args.sample_per_gpu, comp_streams, copy_streams)
    sync_all(devices, workers)

    ok = verify(S, devices, workers, args.rows, args.cols, check_payload=args.check)
    if not ok:
        print("\nCORRECTNESS FAILED")
        workers.close()
        sys.exit(1)
    del S

    print("\n=== timed trials (input resident -> sorted blocks resident) ===")
    times = []
    best_phase = None
    for t in range(args.trials):
        sync_all(devices, workers)
        t0 = time.perf_counter()
        S, M, phase = run_sort(X, devices, workers, n_buckets, args.cols, args.sample_per_gpu, comp_streams, copy_streams)
        sync_all(devices, workers)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        if best_phase is None or dt <= min(times):
            best_phase = phase
        print(f"trial {t}: {dt:.4f} s   rows/s={args.rows / dt:,.0f}   {(raw_bytes / 2**30) / dt:.2f} GiB/s")
        del S
        sync_all(devices, workers)

    best = min(times)
    mean = sum(times) / len(times)

    print("\n=== phase breakdown (best trial, with per-phase barriers) ===")
    for name in ("sample", "partition", "exchange", "sort"):
        ms = best_phase[name] * 1e3
        print(f"{name:>10}: {ms:8.2f} ms  ({100 * best_phase[name] / best:5.1f}%)")

    print("\n=== RESULT ===")
    print(f"dataset: {gib(raw_bytes)}  ({args.rows:,} rows x {args.cols} int32 cols)")
    print(f"gpu sort seconds (best): {best:.4f}")
    print(f"gpu sort seconds (mean): {mean:.4f}")
    print(f"rows/sec (best): {args.rows / best:,.0f}")
    print(f"raw GiB/sec (best): {(raw_bytes / 2**30) / best:.3f}")
    if args.baseline_seconds > 0:
        print(f"\nray baseline: {args.baseline_seconds:.3f} s")
        print(f"speedup vs ray (best): {args.baseline_seconds / best:.1f}x")
        print(f"speedup vs ray (mean): {args.baseline_seconds / mean:.1f}x")

    workers.close()


if __name__ == "__main__":
    main()
