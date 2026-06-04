"""Probe the OUTPUT-side cost of the GPU sort to size the "DMA sorted blocks
straight into the Ray object store" idea.

Today each output block makes TWO host trips:
    (1) D2H : VRAM -> pinned host buffer        (counted in full_s, ~1.36 s/64GiB)
    (2) ray.put : pinned host -> plasma /dev/shm (a 2nd 64 GiB host memcpy, the
        materialize-wall gap we want to remove)

"Direct to object store" = make the D2H land in the plasma buffer, so (2) is gone.
Whether that pays off hinges on three measured numbers:

    A. ray.put throughput (pinned-backed Arrow -> plasma)  <- the copy we'd remove
    B. cudaHostRegister throughput on a fresh host buffer  <- cost to pin a plasma
       buffer per sort so the direct DMA can be async (Option C)
    C. D2H pinned vs pageable, 1 GPU                        <- floor for the direct DMA

All CUDA work is done BEFORE ray.init() so Ray can't hide the GPU from the driver.
"""
import os
import time

os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pyarrow as pa
import cupy as cp

GIB = 2 ** 30


def best(fn, n=3):
    fn()  # warmup
    return min(fn() for _ in range(n))


def main():
    rows = 64 * 1024 * 1024   # 64M rows
    cols = 16                 # x 16 int32 = 4 GiB (one GPU's share at 64 GiB/16)
    nbytes = rows * cols * 4
    gib = nbytes / GIB
    print(f"=== objstore_probe: one block = {rows:,} rows x {cols} int32 = {gib:.1f} GiB ===")

    # ---- CUDA-side measurements FIRST (before ray.init touches the env) -----
    dev = cp.zeros((cols, rows), dtype=cp.int32)
    mem = cp.cuda.alloc_pinned_memory(nbytes)
    hbuf = np.frombuffer(mem, dtype=np.int32, count=rows * cols).reshape(cols, rows)
    pag = np.zeros((cols, rows), dtype=np.int32)
    cp.cuda.runtime.deviceSynchronize()

    # C. D2H pinned vs pageable, single GPU
    def d2h_pinned():
        t0 = time.perf_counter()
        dev.get(out=hbuf)
        cp.cuda.runtime.deviceSynchronize()
        return time.perf_counter() - t0

    def d2h_pageable():
        t0 = time.perf_counter()
        dev.get(out=pag)
        cp.cuda.runtime.deviceSynchronize()
        return time.perf_counter() - t0

    dp = best(d2h_pinned)
    dq = best(d2h_pageable)

    # B. cudaHostRegister throughput on a fresh pageable buffer
    reg = None
    regbuf = np.zeros(rows * cols, dtype=np.int32)
    ptr = regbuf.ctypes.data
    try:
        def reg_once():
            t0 = time.perf_counter()
            cp.cuda.runtime.hostRegister(ptr, nbytes, 0)
            dt = time.perf_counter() - t0
            cp.cuda.runtime.hostUnregister(ptr)
            return dt
        reg = best(reg_once, n=2)
    except Exception as e:
        print(f"   (cudaHostRegister failed: {e})")

    # Build the Arrow table (zero-copy over the pinned host buffer) for ray.put.
    tbl = pa.table({f"c{j}": pa.array(hbuf[j]) for j in range(cols)})

    # ---- A. ray.put: pinned-backed Arrow -> plasma --------------------------
    import ray
    ray.init(object_store_memory=40 * GIB, logging_level="ERROR")

    def put_once():
        t0 = time.perf_counter()
        r = ray.put(tbl)
        dt = time.perf_counter() - t0
        del r
        return dt

    put = best(put_once)

    # ---- report ------------------------------------------------------------
    print(f"\nA. ray.put 1 block (pinned->plasma): {put * 1e3:8.1f} ms  {gib / put:6.2f} GiB/s")
    print(f"   => 16 blocks SEQUENTIAL (as the code does today): ~{16 * put:5.2f} s for 64 GiB")
    if reg is not None:
        print(f"\nB. cudaHostRegister 1 block: {reg * 1e3:8.1f} ms  {gib / reg:6.2f} GiB/s")
        print(f"   => pinning 64 GiB of fresh plasma per sort (Option C tax): ~{16 * reg:5.2f} s")
    print(f"\nC. D2H 1 GPU pinned  : {dp * 1e3:8.1f} ms  {gib / dp:6.2f} GiB/s")
    print(f"   D2H 1 GPU pageable: {dq * 1e3:8.1f} ms  {gib / dq:6.2f} GiB/s")

    ray.shutdown()


if __name__ == "__main__":
    main()
