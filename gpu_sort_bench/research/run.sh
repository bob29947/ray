#!/usr/bin/env bash
#
# Run the CPU vs Polars vs GPU sort benchmark (cpu_vs_gpu.py).
#
# Each backend runs in its own fresh process / fresh ray.init(), and each result
# is independently verified to be actually globally sorted.
#
# Usage:
#   bash run.sh              # full 64 GiB, 3 timed trials per backend (~10-13 min)
#   bash run.sh --quick      # tiny 1 GiB sanity check (~1 min)
#   bash run.sh --trials 5   # more trials
#
set -uo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python

# Best-effort cleanup: stop any leftover Ray cluster / detached GPU sorter actor
# from a previous (maybe Ctrl-C'd) run so the GPUs and /dev/shm start clean.
echo "Cleaning up any leftover Ray state..."
.venv/bin/ray stop --force >/dev/null 2>&1 || true
sleep 2

echo "GPUs:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader | head -16 || true
echo

exec "$PY" cpu_vs_gpu.py "$@"
