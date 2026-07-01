import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.fs as pa_fs
import pyarrow.parquet as pq
import pytest

import ray.data._internal.datasource.parquet_datasource as parquet_datasource_module
from ray.data._internal.datasource.parquet_datasource import (
    ParquetDatasource,
    _ParquetFragment,
)
from ray.data._internal.datasource.parquet_range import (
    ParquetRangePlanningError,
    plan_parquet_row_groups,
)
from ray.data._internal.datasource.parquet_range_v1 import (
    ParquetSourceIdentityError,
    PosixSourceIdentity,
    capture_posix_source_identity,
    extract_v1_parquet_row_group_metadata,
    try_extract_v1_parquet_row_group_metadata,
    verify_posix_source_identity,
)


def _write_parquet(
    path: Path,
    users,
    *,
    row_group_size=2,
    write_statistics=True,
    extra_columns=None,
):
    columns = {
        "User": users,
        "Card": list(range(len(users))),
        "payload": [value * 10 for value in range(len(users))],
    }
    if extra_columns:
        columns.update(extra_columns)
    pq.write_table(
        pa.table(columns),
        path,
        row_group_size=row_group_size,
        write_statistics=write_statistics,
    )


def _datasource_state(
    path: Path,
    *,
    fragment=None,
    file_size=None,
    filesystem=None,
    partition_columns=(),
    include_paths=False,
):
    if fragment is None:
        fragment = next(
            pds.dataset(str(path), format="parquet").get_fragments()
        )
    if file_size is None:
        file_size = path.stat().st_size
    if filesystem is None:
        filesystem = pa_fs.LocalFileSystem()
    return SimpleNamespace(
        _pq_fragments=[_ParquetFragment(fragment, file_size)],
        _pq_paths=[str(path)],
        _filesystem=filesystem,
        _partition_columns=list(partition_columns),
        _include_paths=include_paths,
        _include_row_hash=False,
    )


def test_extracts_exact_v1_fragments_and_projected_sizes(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1, 2, 3])
    datasource = _datasource_state(path)

    result = try_extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User", "Card"],
    )

    assert result.extracted
    assert result.rejection_reason is None
    assert result.projection == ("Card", "User")
    assert result.elapsed_s >= 0
    assert [(entry.key_min, entry.key_max) for entry in result.row_groups] == [
        (0, 1),
        (2, 3),
    ]
    assert all(entry.null_count == 0 for entry in result.row_groups)
    assert all(
        [size.column for size in entry.column_sizes] == ["Card", "User"]
        for entry in result.row_groups
    )
    assert all(
        sum(size.encoded_bytes for size in entry.column_sizes)
        < entry.encoded_bytes
        for entry in result.row_groups
    )
    assert all(
        verify_posix_source_identity(entry.path, entry.source_identity)
        for entry in result.row_groups
    )

    plan = plan_parquet_row_groups(
        result.row_groups,
        num_partitions=2,
        projection=result.projection,
    )
    assert plan.metrics.projection_size_complete
    assert plan.metrics.source_encoded_bytes == sum(
        sum(size.encoded_bytes for size in entry.column_sizes)
        for entry in result.row_groups
    )


def test_accepts_real_parquet_datasource_state(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [10, 11, 12, 13])
    exact_state = _datasource_state(path)
    datasource = ParquetDatasource.__new__(ParquetDatasource)
    datasource.__dict__.update(exact_state.__dict__)

    row_groups = extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User", "Card"],
    )

    assert [(entry.key_min, entry.key_max) for entry in row_groups] == [
        (10, 11),
        (12, 13),
    ]
    assert {size.column for size in row_groups[0].column_sizes} == {
        "User",
        "Card",
    }


def test_adapter_never_resolves_or_relists_paths(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    datasource = _datasource_state(path)

    def unexpected_discovery(*args, **kwargs):
        raise AssertionError("adapter attempted path discovery")

    monkeypatch.setattr(
        parquet_datasource_module,
        "_resolve_paths_and_filesystem",
        unexpected_discovery,
    )
    monkeypatch.setattr(
        parquet_datasource_module,
        "_list_files",
        unexpected_discovery,
    )

    row_groups = extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User"],
    )
    assert len(row_groups) == 1


def test_honors_existing_row_group_subset_and_aggregates_nested_leaves(
    tmp_path,
):
    path = tmp_path / "nested.parquet"
    nested = pa.array(
        [
            {"left": 1, "right": 2},
            {"left": 3, "right": 4},
            {"left": 5, "right": 6},
            {"left": 7, "right": 8},
        ]
    )
    _write_parquet(
        path,
        [0, 1, 2, 3],
        extra_columns={"nested": nested},
    )
    fragment = next(
        pds.dataset(str(path), format="parquet").get_fragments()
    ).split_by_row_group()[1]
    datasource = _datasource_state(path, fragment=fragment)

    row_groups = extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User", "nested"],
    )

    assert len(row_groups) == 1
    assert row_groups[0].row_group_id == 1
    assert (row_groups[0].key_min, row_groups[0].key_max) == (2, 3)
    by_column = {size.column: size for size in row_groups[0].column_sizes}
    metadata = fragment.metadata.row_group(1)
    expected_nested_encoded = sum(
        metadata.column(index).total_compressed_size for index in (3, 4)
    )
    assert by_column["nested"].encoded_bytes == expected_nested_encoded


def test_projection_ignores_v1_partition_and_synthetic_columns(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    datasource = _datasource_state(
        path,
        partition_columns=["partition"],
        include_paths=True,
    )

    result = try_extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User", "Card", "partition", "path"],
    )

    assert result.extracted
    assert result.projection == ("Card", "User")


@pytest.mark.parametrize(
    ("users", "write_statistics", "reason_code"),
    [
        ([1, 2], False, "missing_statistics"),
        ([1, None], True, "null_keys"),
        ([1.0, 2.0], True, "unsupported_key_type"),
    ],
)
def test_unsafe_key_statistics_fail_closed(
    tmp_path, users, write_statistics, reason_code
):
    path = tmp_path / f"{reason_code}.parquet"
    _write_parquet(path, users, write_statistics=write_statistics)

    result = try_extract_v1_parquet_row_group_metadata(
        _datasource_state(path),
        key="User",
        projection=["User"],
    )

    assert not result.extracted
    assert result.row_groups == ()
    assert result.rejection_reason == reason_code
    assert result.exception_type == "ParquetRangePlanningError"


def test_missing_projection_and_nonlocal_filesystem_fail_closed(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])

    missing = try_extract_v1_parquet_row_group_metadata(
        _datasource_state(path),
        key="User",
        projection=["User", "missing"],
    )
    assert missing.rejection_reason == "missing_projection_column"

    nonlocal_result = try_extract_v1_parquet_row_group_metadata(
        _datasource_state(path, filesystem=object()),
        key="User",
        projection=["User"],
    )
    assert nonlocal_result.rejection_reason == "unsupported_filesystem"


def test_v1_discovery_size_mismatch_fails_closed(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    datasource = _datasource_state(
        path, file_size=path.stat().st_size + 1
    )

    result = try_extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User"],
    )

    assert result.rejection_reason == "source_changed"


def test_detects_source_change_while_reading_footer(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    fragment = next(
        pds.dataset(str(path), format="parquet").get_fragments()
    )

    class MutatingFragment:
        path = fragment.path
        row_groups = fragment.row_groups

        @property
        def metadata(self):
            metadata = fragment.metadata
            source_stat = os.stat(path)
            os.utime(
                path,
                ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
            )
            return metadata

    datasource = _datasource_state(path, fragment=MutatingFragment())

    result = try_extract_v1_parquet_row_group_metadata(
        datasource,
        key="User",
        projection=["User"],
    )

    assert result.rejection_reason == "source_changed_during_planning"


def test_worker_identity_verifier_detects_modify_replace_and_missing(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    encoded = capture_posix_source_identity(path)
    decoded = PosixSourceIdentity.decode(encoded)
    assert decoded.size == path.stat().st_size
    assert verify_posix_source_identity(path, encoded)

    source_stat = path.stat()
    os.utime(
        path,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000_000),
    )
    with pytest.raises(ParquetSourceIdentityError) as exc_info:
        verify_posix_source_identity(path, encoded)
    assert exc_info.value.reason_code == "source_changed"

    current = capture_posix_source_identity(path)
    path.unlink()
    with pytest.raises(ParquetSourceIdentityError) as exc_info:
        verify_posix_source_identity(path, current)
    assert exc_info.value.reason_code == "source_unavailable"


def test_identity_verifier_rejects_malformed_identity_and_non_file(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])

    with pytest.raises(ParquetSourceIdentityError) as exc_info:
        verify_posix_source_identity(path, "bad")
    assert exc_info.value.reason_code == "invalid_source_identity"

    with pytest.raises(ParquetSourceIdentityError) as exc_info:
        capture_posix_source_identity(tmp_path)
    assert exc_info.value.reason_code == "unsupported_source_type"


def test_metadata_result_is_pickle_serializable(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    result = try_extract_v1_parquet_row_group_metadata(
        _datasource_state(path),
        key="User",
        projection=["User", "Card"],
    )

    restored = pickle.loads(pickle.dumps(result))
    assert restored == result
    assert restored.extracted


def test_invalid_fragment_state_is_deterministic(tmp_path):
    path = tmp_path / "data.parquet"
    _write_parquet(path, [0, 1])
    datasource = _datasource_state(path)
    datasource._pq_paths = [str(tmp_path / "other.parquet")]

    with pytest.raises(ParquetRangePlanningError) as exc_info:
        extract_v1_parquet_row_group_metadata(
            datasource,
            key="User",
            projection=["User"],
        )
    assert exc_info.value.reason_code == "fragment_state_mismatch"
