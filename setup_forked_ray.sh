#!/usr/bin/env bash
#
# Make THIS worktree's Ray ("3.0.0.dev0" from source) the importable `ray`,
# instead of any pip-installed Ray. We do NOT rebuild the C++ core: this Ray
# source tree is pure-Python except for compiled artifacts (``_raylet.so``, the
# protobuf ``*_pb2.py``, the ``raylet`` / ``gcs_server`` binaries, vendored
# ``thirdparty_files``). Those are git-ignored build outputs, so we overlay them
# (via symlinks) from an already-built Ray checkout at the SAME commit, then put
# this worktree's ``python/`` first on PYTHONPATH.
#
# Usage:
#   source setup_forked_ray.sh        # ensure overlay + activate in this shell
#   ./setup_forked_ray.sh             # ensure overlay + print activation hint
#
# Override the build source / interpreter if your paths differ:
#   RAY_BUILD_SRC=/path/to/built/ray  RAY_FORK_VENV=/path/to/venv  source setup_forked_ray.sh
set -uo pipefail

# Worktree root = directory containing this script (works in any worktree).
_WT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# A fully-built Ray checkout at the same base commit, providing the compiled
# artifacts to overlay (defaults to the editable GPU build).
RAY_BUILD_SRC="${RAY_BUILD_SRC:-/bobbwang/projects/ray}"
# Interpreter that already has the GPU stack (cudf/cupy/rmm + rapidsmpf/ucxx).
RAY_FORK_VENV="${RAY_FORK_VENV:-/bobbwang/projects/ray/.venv-gpu-sort}"

_SRC="$RAY_BUILD_SRC/python/ray"
_DST="$_WT/python/ray"

if [[ ! -e "$_SRC/_raylet.so" ]]; then
    echo "ERROR: no compiled Ray core at $_SRC/_raylet.so" >&2
    echo "       set RAY_BUILD_SRC to a built Ray checkout at the same commit." >&2
    return 1 2>/dev/null || exit 1
fi

# Overlay the git-ignored compiled / generated build outputs (idempotent).
_overlay() { ln -sfn "$_SRC/$1" "$_DST/$1"; }
mkdir -p "$_DST/core/src/ray/gcs"
_overlay _raylet.so
_overlay core/generated
_overlay core/libjemalloc.so
_overlay thirdparty_files
_overlay serve/generated
_overlay core/src/ray/raylet/raylet
_overlay core/src/ray/gcs/gcs_server

# thirdparty_files is matched by a trailing-slash .gitignore rule that does not
# cover the symlink, so exclude it locally to keep `git status` clean.
_exc="$(git -C "$_WT" rev-parse --git-path info/exclude 2>/dev/null)"
if [[ -n "${_exc:-}" ]] && ! grep -qxF '/python/ray/thirdparty_files' "$_exc" 2>/dev/null; then
    echo '/python/ray/thirdparty_files' >> "$_exc"
fi

export PYTHONPATH="$_WT/python:${PYTHONPATH:-}"
export RAY_FORK_PYTHON="$RAY_FORK_VENV/bin/python"
# Default GPU count for the experimental sort engine (override as needed).
export RAY_DATA_GPU_SORT_NUM_GPUS="${RAY_DATA_GPU_SORT_NUM_GPUS:-16}"

echo "Forked Ray activated:"
echo "  worktree      : $_WT"
echo "  compiled core : $_SRC (overlaid)"
echo "  PYTHONPATH    : $_WT/python (first)"
echo "  interpreter   : \$RAY_FORK_PYTHON = $RAY_FORK_PYTHON"
echo
echo "Sanity check:"
echo "  \$RAY_FORK_PYTHON -c 'import ray, cudf, rapidsmpf; print(ray.__version__, ray.__file__)'"
echo "Expect: 3.0.0.dev0  $_WT/python/ray/__init__.py"
