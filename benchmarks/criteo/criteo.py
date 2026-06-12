"""Loader + column-role logic for the CriteoPrivateAd CPU preprocessing baseline.

Analogous to ``yambda.py`` but for the CriteoPrivateAd parquet dataset
(Hive-partitioned by ``day_int``). The root defaults to the local DGX copy
(``/bobbwang/datasets/CriteoPrivateAd/data``) and can be repointed to any local
path or ``s3://`` URI via :func:`set_data_root` (used by the benchmark's
``--data-root``), so the SAME loader serves both the local box and an AWS Ray
cluster reading from S3. One row = one ad impression (banner display).

This module only decides *roles* (which column is a target / categorical /
numeric / list / dropped) from the parquet schema and the per-column null
fractions read straight from row-group statistics (no full scan). The actual
transforms live in ``bench_criteo_cpu_baseline.py``.

Feature buckets (from the dataset README / arXiv:2502.12103):

* ``features_kv_bits_constrained_*``      -- single-domain USER signals (12-bit).
* ``features_kv_not_constrained_*``       -- CAMPAIGN / interest-group signals.
* ``features_browser_bits_constrained_*`` -- cross-domain BROWSER signals (12-bit).
* ``features_ctx_not_constrained_*``      -- CONTEXT signals.
* ``features_not_available_*``            -- NOT available at inference -> dropped.

Targets: ``is_clicked``, ``is_click_landed``, ``is_visit`` (binary) and
``nb_sales`` (count; null == no attributed sale).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

import ray

# --------------------------------------------------------------------------- #
# Dataset layout
# --------------------------------------------------------------------------- #
DATA_ROOT = "/bobbwang/datasets/CriteoPrivateAd/data"

# Labels / targets (kept, never fed through feature transforms).
TARGET_BINARY = ["is_clicked", "is_click_landed", "is_visit"]
SALES_RAW = "nb_sales"  # int64 count; null == no attributed sale.

# Identifiers. ``id`` is always dropped; ``user_id`` is the sort key and is then
# dropped (mode A); ``campaign_id`` / ``publisher_id`` are kept as categoricals;
# ``display_order`` is both a sort key and a numeric feature.
ID_DROP = ["id"]
SORT_USER = "user_id"
SORT_DISPLAY = "display_order"
# Hive partition column. Physically absent from the parquet files (it comes from
# the ``day_int=<n>`` folder name); Ray exposes it as a *string* on read, so it
# must be cast to int before it can be used as a numeric sort key. Kept in the
# output as a metadata / sort-key column (never a transformed feature).
SORT_DAY = "day_int"
CATEGORICAL_IDS = ["campaign_id", "publisher_id"]

# Post-display conversion-delay arrays: label leakage -> always dropped.
DELAY_ARRAYS = [
    "sale_delay_after_display_array",
    "click_delay_after_display_array",
    "landed_click_delay_after_display_array",
]

# Pre-display feature buckets we keep (inference-available).
KEPT_BUCKETS: Tuple[str, ...] = (
    "features_kv_bits_constrained_",
    "features_kv_not_constrained_",
    "features_browser_bits_constrained_",
    "features_ctx_not_constrained_",
)
# Cross-device / training-only bucket: not available at inference -> dropped.
NOT_AVAILABLE_BUCKET = "features_not_available_"

# A column is treated as "all null" (and dropped) at/above this null fraction.
ALL_NULL_THRESHOLD = 0.999


# --------------------------------------------------------------------------- #
# Data root + filesystem helpers (local path OR ``s3://`` URI)
# --------------------------------------------------------------------------- #
def is_s3(path: str) -> bool:
    """True if ``path`` is an ``s3://`` URI (vs a local filesystem path)."""
    return str(path).startswith("s3://")


def set_data_root(path: str) -> None:
    """Point the loader at a new dataset root: a local path or an ``s3://`` URI.

    The benchmark calls this once (from ``--data-root``) before any discovery,
    so the SAME code reads either the local DGX copy or an S3-staged copy of
    CriteoPrivateAd. ``s3://`` roots are normalized without a trailing slash.
    """
    global DATA_ROOT
    DATA_ROOT = str(path).rstrip("/") if is_s3(path) else str(path)


def _s3_fs_and_path(uri: str):
    """``(pyarrow.fs.FileSystem, fs_native_path)`` for an ``s3://`` URI."""
    import pyarrow.fs as pafs

    return pafs.FileSystem.from_uri(uri)


def _s3_children(uri: str):
    """Immediate children of an ``s3://`` "directory" as
    ``(child_s3_uri, is_dir, is_file)`` tuples. Uses ``pyarrow.fs`` -- never
    local ``glob`` / ``os.path`` -- so it is safe for object storage."""
    import pyarrow.fs as pafs

    fs, path = pafs.FileSystem.from_uri(uri)
    selector = pafs.FileSelector(path, recursive=False, allow_not_found=True)
    out = []
    for info in fs.get_file_info(selector):
        out.append(
            (
                "s3://" + info.path,
                info.type == pafs.FileType.Directory,
                info.type == pafs.FileType.File,
            )
        )
    return out


def _parquet_metadata(path: str):
    """Parquet footer metadata for a local path OR an ``s3://`` URI."""
    if is_s3(path):
        fs, p = _s3_fs_and_path(path)
        return pq.ParquetFile(p, filesystem=fs).metadata
    return pq.ParquetFile(path).metadata


def _read_schema(path: str):
    """Parquet schema for a local path OR an ``s3://`` URI."""
    if is_s3(path):
        fs, p = _s3_fs_and_path(path)
        return pq.read_schema(p, filesystem=fs)
    return pq.read_schema(path)


# --------------------------------------------------------------------------- #
# Paths / IO
# --------------------------------------------------------------------------- #
def day_path(day: int) -> str:
    if is_s3(DATA_ROOT):
        return f"{DATA_ROOT}/day_int={day}"
    return os.path.join(DATA_ROOT, f"day_int={day}")


def parquet_paths(day: int) -> List[str]:
    if is_s3(DATA_ROOT):
        return sorted(
            uri
            for uri, _is_dir, is_file in _s3_children(day_path(day))
            if is_file and uri.endswith(".parquet")
        )
    return sorted(glob.glob(os.path.join(day_path(day), "*.parquet")))


def parquet_paths_days(days: List[int]) -> List[str]:
    out: List[str] = []
    for d in days:
        out.extend(parquet_paths(d))
    return out


def discover_days() -> List[int]:
    """All ``day_int=<n>`` partitions present under ``DATA_ROOT`` (local or s3)."""
    days: List[int] = []
    if is_s3(DATA_ROOT):
        for uri, is_dir, _is_file in _s3_children(DATA_ROOT):
            base = uri.rstrip("/").rsplit("/", 1)[-1]
            if is_dir and base.startswith("day_int="):
                try:
                    days.append(int(base.split("=", 1)[1]))
                except (IndexError, ValueError):
                    continue
        return sorted(days)
    for p in glob.glob(os.path.join(DATA_ROOT, "day_int=*")):
        if os.path.isdir(p):
            try:
                days.append(int(os.path.basename(p).split("=", 1)[1]))
            except (IndexError, ValueError):
                continue
    return sorted(days)


def parse_days(spec: str, available: Optional[List[int]] = None) -> List[int]:
    """Parse a ``--days`` spec into an explicit, ascending list of day ints.

    Accepts ``"1"``, ``"1-5"``, ``"1-30"``, ``"all"``. Every requested day must
    physically exist under ``DATA_ROOT`` (else ``ValueError``).
    """
    avail = available if available is not None else discover_days()
    avail_set = set(avail)
    s = str(spec).strip().lower()
    if s == "all":
        return list(avail)
    if "-" in s:
        lo_s, hi_s = s.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise ValueError(f"--days range end < start: {spec!r}")
        requested = list(range(lo, hi + 1))
    else:
        requested = [int(s)]
    missing = [d for d in requested if d not in avail_set]
    if missing:
        raise ValueError(
            f"requested day(s) not present in {DATA_ROOT}: {missing} "
            f"(available: {avail})"
        )
    return requested


def _first_nonempty(paths: List[str]) -> str:
    """First shard with >0 rows (day 1 ships an empty 0-row placeholder)."""
    for p in paths:
        if _parquet_metadata(p).num_rows > 0:
            return p
    return paths[0]


def _first_nonempty_across(days: List[int]) -> str:
    """First >0-row shard across all selected days (schema is uniform)."""
    for d in days:
        for p in parquet_paths(d):
            if _parquet_metadata(p).num_rows > 0:
                return p
    return parquet_paths(days[0])[0]


def read_ray(day: int, *, override_num_blocks: Optional[int] = None) -> "ray.data.Dataset":
    """Read one ``day_int=<day>`` partition. The empty 0-row shard is harmless."""
    return ray.data.read_parquet(
        day_path(day), override_num_blocks=override_num_blocks
    )


def _has_day_int(ds: "ray.data.Dataset") -> bool:
    try:
        return SORT_DAY in ds.schema().names
    except Exception:
        return False


def read_ray_days(days: List[int]) -> "ray.data.Dataset":
    """Read all selected ``day_int=<d>`` folders into a single Ray Dataset.

    ``day_int`` is the Hive partition column: it is *not* stored inside the
    parquet files. Ray normally re-derives it from the folder name (exposing it
    as a string). We rely on that fast path, but if Ray does *not* surface it we
    fall back to reading each folder on its own and tagging its rows with the
    correct day. Either way the returned dataset has a real ``day_int`` column
    (cast to int64 later, in the prep stage, so it sorts numerically).
    """
    days = list(days)
    paths = [day_path(d) for d in days]
    ds = ray.data.read_parquet(paths)
    if _has_day_int(ds):
        return ds

    # Fallback: partition column not exposed -> add it manually per folder.
    parts: List["ray.data.Dataset"] = []
    for d in days:
        dsd = ray.data.read_parquet(day_path(d))
        dsd = dsd.map_batches(
            lambda t, _d=d: t.append_column(
                SORT_DAY, pa.array([_d] * t.num_rows, type=pa.int64())
            ),
            batch_format="pyarrow",
            batch_size=None,
        )
        parts.append(dsd)
    ds = parts[0]
    for p in parts[1:]:
        ds = ds.union(p)
    return ds


def sort_key(*, multi_day: bool = False) -> List[str]:
    """Realistic recsys ordering: each user's impressions in display order.

    day_int=1: ``[user_id, display_order]``; multi-day adds ``day_int`` in the
    middle so a user's days stay contiguous and ordered.
    """
    if multi_day:
        return [SORT_USER, "day_int", SORT_DISPLAY]
    return [SORT_USER, SORT_DISPLAY]


# --------------------------------------------------------------------------- #
# Metadata (no scan): per-column null fractions from row-group statistics
# --------------------------------------------------------------------------- #
def null_fractions(day: int) -> Tuple[Dict[str, float], int]:
    return null_fractions_days([day])


def null_fractions_days(days: List[int]) -> Tuple[Dict[str, float], int]:
    """Per-column null fraction + row count aggregated over the selected days.

    Reads only parquet footer/row-group statistics (no data scan). Aggregating
    across all selected days matters: a column can be all-null on day 1 yet
    populated later, so the all-null drop and the null-indicator threshold must
    be decided on the *full* selected dataset, not on a single day.
    """
    schema = _read_schema(_first_nonempty_across(days))
    total = 0
    nulls = dict.fromkeys(schema.names, 0)
    for d in days:
        for path in parquet_paths(d):
            md = _parquet_metadata(path)
            total += md.num_rows
            for rg in range(md.num_row_groups):
                rgm = md.row_group(rg)
                for c in range(rgm.num_columns):
                    col = rgm.column(c)
                    st = col.statistics
                    if st is not None and col.path_in_schema in nulls:
                        nulls[col.path_in_schema] += st.null_count
    denom = total or 1
    return {name: nulls[name] / denom for name in schema.names}, total


# --------------------------------------------------------------------------- #
# Column-role resolution
# --------------------------------------------------------------------------- #
def _in_kept_bucket(name: str) -> bool:
    return any(name.startswith(b) for b in KEPT_BUCKETS)


@dataclass
class ColumnRoles:
    """Resolved roles for the selected day(s), grounded in schema + null stats."""

    days: List[int]
    total_rows: int
    null_fracs: Dict[str, float]

    targets_binary: List[str] = field(default_factory=list)
    sales_raw: str = SALES_RAW
    categorical: List[str] = field(default_factory=list)  # -> OrdinalEncoder
    numeric_raw: List[str] = field(default_factory=list)  # double feature cols
    list_features: List[str] = field(default_factory=list)  # -> <col>_len
    dropped: Dict[str, List[str]] = field(default_factory=dict)
    sort_key: List[str] = field(default_factory=list)

    null_indicator_threshold: float = 0.01
    # "lean" (default): the inference-realistic recipe -- drop the 80
    # ``features_not_available_*`` (not available at serving). "wide": keep that
    # bucket as additional features (classified by dtype, all-null still
    # dropped), giving the fused stage a much wider frame. Same steps run on CPU
    # and GPU either way; this only changes how many columns are processed.
    feature_set: str = "lean"

    @property
    def day(self) -> int:
        """First selected day (back-compat / labelling helper)."""
        return self.days[0]

    @property
    def multi_day(self) -> bool:
        return len(self.days) > 1

    @property
    def metadata_keys(self) -> List[str]:
        """Sort-key / metadata columns kept RAW in the output (user_id, day_int,
        display_order). Never encoded or scaled; they exist so the saved output
        stays globally sortable and verifiable. ``user_id`` would only become a
        feature in a separate high-cardinality encoder stress mode."""
        return [SORT_USER, SORT_DAY, SORT_DISPLAY]

    # Derived names produced by the prep stage.
    @property
    def list_len_cols(self) -> List[str]:
        return [f"{c}_len" for c in self.list_features]

    @property
    def numeric_features(self) -> List[str]:
        """Numeric feature columns the StandardScaler scales.

        ``display_order`` is intentionally NOT here: it is a raw sort-key /
        metadata column (see ``metadata_keys``) kept unscaled so the saved sort
        key stays directly verifiable on the written parquet."""
        return self.numeric_raw + self.list_len_cols

    @property
    def impute_numeric(self) -> List[str]:
        """Numeric cols that actually have nulls (mean-impute these)."""
        return [c for c in self.numeric_raw if self.null_fracs.get(c, 0.0) > 0.0]

    @property
    def impute_categorical(self) -> List[str]:
        """Categorical cols that actually have nulls (most_frequent-impute these)."""
        return [c for c in self.categorical if self.null_fracs.get(c, 0.0) > 0.0]

    @property
    def indicator_cols(self) -> List[str]:
        """Numeric cols above the null threshold -> add <col>_isnull (unscaled)."""
        return [
            c
            for c in self.numeric_raw
            if self.null_fracs.get(c, 0.0) > self.null_indicator_threshold
        ]

    @property
    def targets(self) -> List[str]:
        """Final target columns after prep (sales_count + is_sale replace nb_sales)."""
        return self.targets_binary + ["sales_count", "is_sale"]


def column_roles(
    day: int, *, null_indicator_threshold: float = 0.01, multi_day: bool = False,
    feature_set: str = "lean",
) -> ColumnRoles:
    """Back-compat single-day wrapper around :func:`column_roles_multi`."""
    return column_roles_multi(
        [day], null_indicator_threshold=null_indicator_threshold,
        feature_set=feature_set,
    )


def column_roles_multi(
    days: List[int], *, null_indicator_threshold: float = 0.01,
    feature_set: str = "lean",
) -> ColumnRoles:
    """Resolve column roles from the selected days' schema + null fractions.

    Null fractions are aggregated across *all* selected days so the all-null drop
    and the null-indicator threshold reflect the full selected dataset. The
    sort key gains ``day_int`` in the middle whenever more than one day is read.

    ``feature_set``: ``"lean"`` (default) drops the ``features_not_available_*``
    bucket (not available at inference -- the realistic recipe). ``"wide"`` keeps
    that bucket as features, classified by dtype exactly like the kept buckets
    (list -> ``<col>_len``, floating -> numeric, integer -> categorical),
    all-null columns still dropped. ``"wide"`` roughly doubles the processed
    column count, giving the fused GPU stage a much wider frame to amortize its
    fixed per-batch/launch costs; the steps themselves are unchanged, so the CPU
    and GPU pipelines stay a fair, identical-work comparison.
    """
    if feature_set not in ("lean", "wide"):
        raise ValueError(f"feature_set must be 'lean' or 'wide', got {feature_set!r}")
    days = list(days)
    fracs, total = null_fractions_days(days)
    schema = _read_schema(_first_nonempty_across(days))
    multi = len(days) > 1

    roles = ColumnRoles(
        days=days,
        total_rows=total,
        null_fracs=fracs,
        null_indicator_threshold=null_indicator_threshold,
        feature_set=feature_set,
        sort_key=sort_key(multi_day=multi),
    )
    dropped = {
        "id": [],
        "metadata_sort_keys": [SORT_USER, SORT_DAY, SORT_DISPLAY],
        "delay_arrays_leakage": [],
        "not_available_at_inference": [],
        "all_null": [],
    }
    keep_not_available = feature_set == "wide"

    for f_ in schema:
        name, typ = f_.name, f_.type

        if name in TARGET_BINARY:
            roles.targets_binary.append(name)
            continue
        if name == SALES_RAW:
            continue  # handled into sales_count / is_sale by prep
        if name in ID_DROP:
            dropped["id"].append(name)
            continue
        if name == SORT_USER or name == SORT_DISPLAY or name == SORT_DAY:
            continue  # kept raw as metadata / sort keys, never transformed
        if name in DELAY_ARRAYS:
            dropped["delay_arrays_leakage"].append(name)
            continue
        is_not_available = name.startswith(NOT_AVAILABLE_BUCKET)
        # lean: drop the not-available-at-inference bucket. wide: keep it as
        # features (fall through to the dtype-based classification below).
        if is_not_available and not keep_not_available:
            dropped["not_available_at_inference"].append(name)
            continue
        if fracs.get(name, 0.0) >= ALL_NULL_THRESHOLD:
            dropped["all_null"].append(name)
            continue

        if name in CATEGORICAL_IDS:
            roles.categorical.append(name)
            continue
        if _in_kept_bucket(name) or is_not_available:
            if pa.types.is_list(typ) or pa.types.is_large_list(typ):
                roles.list_features.append(name)
            elif pa.types.is_floating(typ):
                roles.numeric_raw.append(name)
            elif pa.types.is_integer(typ):
                roles.categorical.append(name)
            continue
        # Anything unclassified is conservatively dropped (none expected).
        dropped.setdefault("other", []).append(name)

    roles.dropped = dropped
    return roles
