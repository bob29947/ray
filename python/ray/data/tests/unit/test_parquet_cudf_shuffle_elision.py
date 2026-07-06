import os
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource.parquet_datasource import (
    ParquetDatasource,
    _ParquetCudfDirectReadSpec,
    _parquet_cudf_direct_paths_are_ambient,
)
from ray.data._internal.object_extensions.arrow import ArrowPythonObjectType
from ray.data._internal.logical.operators import MapBatches, MapGroups, Read
from ray.data._internal.planner import parquet_cudf_shuffle_elision as optimizer
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.grouped_data import ParquetCudfShuffleElisionConfig

MIB = 1024**2
GIB = 1024**3


class Tokenizer:
    def __call__(self, batch):
        return batch


def group_fn(batch):
    return batch


def partition_fn(batch, context):
    return batch


def _row(
    path,
    row_group,
    rows,
    lower,
    upper,
    compressed=None,
    uncompressed=None,
):
    identity = optimizer._SourceIdentity(100, "marker")
    return optimizer._RowGroup(
        path,
        identity,
        row_group,
        rows,
        lower,
        upper,
        rows if compressed is None else compressed,
        rows if uncompressed is None else uncompressed,
    )


def _plan(rows, *, partitions=1, shuffle_bytes=MIB, peak_bytes=1, gpu=100 * GIB):
    return optimizer._plan_ranges(
        tuple(rows),
        num_partitions=partitions,
        num_workers=1,
        shuffle_bytes_per_row=shuffle_bytes,
        peak_gpu_bytes_per_row=peak_bytes,
        gpu_memory_bytes=gpu,
    )


@pytest.mark.parametrize(
    "paths,filesystem,expected",
    [
        ("/tmp/data.parquet", None, True),
        ("file:///tmp/data.parquet", None, True),
        (["s3://bucket/a", "s3://bucket/b"], None, True),
        ("s3://bucket/key%3Fpart", None, True),
        ("s3://user:secret@bucket/key", None, False),
        ("s3://bucket/key?region=us-west-2", None, False),
        ("s3://bucket/key#fragment", None, False),
        ("s3a://bucket/key", None, False),
        ("s3://bucket/key", object(), False),
    ],
)
def test_direct_paths_require_ambient_configuration(paths, filesystem, expected):
    assert _parquet_cudf_direct_paths_are_ambient(paths, filesystem) is expected


def test_datasource_owns_direct_read_eligibility():
    datasource = ParquetDatasource.__new__(ParquetDatasource)
    datasource._parquet_cudf_direct_options_compatible = True
    datasource._predicate_expr = None
    datasource._partition_columns = []
    datasource._projection_map = {"User": "User", "Card": "Card"}
    datasource._filesystem = pafs.LocalFileSystem()
    datasource._pq_paths = ["/tmp/a.parquet"]
    datasource._pq_fragments = [
        SimpleNamespace(original=SimpleNamespace(path="/tmp/a.parquet"), file_size=123)
    ]
    datasource._file_schema = pa.schema([("User", pa.int64()), ("Card", pa.int8())])

    result = datasource._get_parquet_cudf_direct_read_spec()
    assert result.reason is None
    assert result.spec.source_kind == "local"
    assert result.spec.projection == ("User", "Card")
    assert result.spec.listed_file_sizes == (123,)

    datasource._predicate_expr = object()
    assert datasource._get_parquet_cudf_direct_read_spec().reason == "predicate"


def test_range_planner_uses_first_safe_p_2p_4p_count():
    rows = [_row("z", 0, 128, 1, 1), _row("a", 0, 128, 0, 0)]
    plan = _plan(rows, peak_bytes=12 * MIB, gpu=4 * GIB)
    assert [(item.lower, item.upper) for item in plan.work] == [(0, 0), (1, 1)]
    assert [item.files[0].path for item in plan.work] == ["a", "z"]
    assert plan.gpu_budget == 2 * GIB


def test_range_planner_applies_inclusive_cost_thresholds():
    plan = _plan([_row("a", 0, 256, 0, 0)])
    assert plan.net_savings == 256 * MIB
    with pytest.raises(optimizer._PlanningRejected, match="range_thresholds"):
        _plan([_row("a", 0, 256, 0, 0)], shuffle_bytes=MIB - 1)

    balanced = [_row("a", 0, 192, 0, 0), _row("b", 0, 64, 1, 1)]
    assert _plan(balanced, partitions=2).skew == 1.5
    with pytest.raises(optimizer._PlanningRejected, match="range_thresholds"):
        _plan(
            [_row("a", 0, 193, 0, 0), _row("b", 0, 63, 1, 1)],
            partitions=2,
        )


def test_range_planner_accounts_for_duplicate_reads_and_memory():
    overlapping = [
        _row("a", 0, 128, 0, 0, compressed=300 * MIB),
        _row("b", 0, 128, 0, 1, compressed=300 * MIB),
        _row("c", 0, 128, 1, 1, compressed=300 * MIB),
    ]
    with pytest.raises(optimizer._PlanningRejected, match="range_thresholds"):
        _plan(overlapping, partitions=2, shuffle_bytes=2 * MIB)

    assert _plan([_row("a", 0, 256, 0, 0)], peak_bytes=8 * MIB, gpu=4 * GIB)
    with pytest.raises(optimizer._PlanningRejected, match="range_thresholds"):
        _plan(
            [_row("a", 0, 256, 0, 0)],
            peak_bytes=8 * MIB + 1,
            gpu=4 * GIB,
        )


def test_footer_planning_uses_only_metadata_and_detects_changes(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    pq.write_table(
        pa.table({"User": [10, 11, 0, 1], "Card": [1, 2, 3, 4]}),
        path,
        row_group_size=2,
    )
    info = pafs.LocalFileSystem().get_file_info(str(path))
    spec = _ParquetCudfDirectReadSpec(
        filesystem=pafs.LocalFileSystem(),
        source_kind="local",
        paths=(str(path),),
        listed_file_sizes=(info.size,),
        projection=("User", "Card"),
        file_schema=pa.schema([("User", pa.int64()), ("Card", pa.int64())]),
        region=None,
    )
    summary = optimizer._read_footer_summary(spec, "User")
    assert [(row.lower, row.upper) for row in summary.row_groups] == [(0, 1), (10, 11)]
    assert summary.footer_bytes > 0

    identity = optimizer._source_identity(spec, str(path))
    calls = 0

    def changing_identity(read_spec, source_path, client=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return identity
        return optimizer._SourceIdentity(identity.size, "changed")

    monkeypatch.setattr(optimizer, "_source_identity", changing_identity)
    with pytest.raises(optimizer._PlanningRejected, match="source_changed"):
        optimizer._read_footer_summary(spec, "User")


def _candidate_fixture(*, explicit=True):
    datasource = ParquetDatasource.__new__(ParquetDatasource)
    datasource._parquet_cudf_direct_options_compatible = True
    datasource._predicate_expr = None
    datasource._partition_columns = []
    datasource._projection_map = {"User": "User", "Card": "Card"}
    datasource._filesystem = pafs.LocalFileSystem()
    datasource._pq_paths = ["/tmp/a.parquet"]
    datasource._pq_fragments = [
        SimpleNamespace(original=SimpleNamespace(path="/tmp/a.parquet"), file_size=1)
    ]
    datasource._file_schema = pa.schema([("User", pa.int64()), ("Card", pa.int8())])
    read = Read(
        datasource=datasource,
        datasource_or_legacy_reader=datasource,
        parallelism=1,
    )
    resources = {"num_cpus": 1.0, "num_gpus": 1.0}
    tokenized = MapBatches(
        Tokenizer,
        input_dependencies=[read],
        batch_size=1024,
        batch_format="cudf",
        zero_copy_batch=True,
        can_modify_num_rows=False,
        compute=ActorPoolStrategy(size=2, max_tasks_in_flight_per_actor=1),
        ray_remote_args=resources,
    )
    config = ParquetCudfShuffleElisionConfig(
        partition_fn=partition_fn,
        shuffle_bytes_per_input_row=MIB,
        peak_gpu_bytes_per_input_row=1,
        gpu_memory_bytes=32 * GIB,
    )
    grouped = MapGroups(
        key=["User", "Card"],
        num_partitions=2,
        num_partitions_explicit=explicit,
        shuffle_strategy=ShuffleStrategy.HASH_SHUFFLE,
        fn=group_fn,
        batch_format="cudf",
        zero_copy_batch=True,
        compute=TaskPoolStrategy(size=2),
        ray_remote_args=resources,
        parquet_cudf_shuffle_elision=config,
        input_dependencies=[tokenized],
    )
    return grouped


def test_selector_accepts_only_the_explicit_exact_pipeline():
    candidate = optimizer._select_candidate(
        _candidate_fixture(), DataContext.get_current().copy()
    )
    assert candidate.group_keys == ("User", "Card")
    assert candidate.num_partitions == candidate.num_workers == 2

    with pytest.raises(optimizer._PlanningRejected, match="partitions"):
        optimizer._select_candidate(
            _candidate_fixture(explicit=False), DataContext.get_current().copy()
        )

    no_backpressure = _candidate_fixture()
    object.__setattr__(
        no_backpressure.input_dependencies[0],
        "compute",
        ActorPoolStrategy(size=2),
    )
    with pytest.raises(optimizer._PlanningRejected, match="tokenizer_pool"):
        optimizer._select_candidate(no_backpressure, DataContext.get_current().copy())


def test_selector_rejects_nondefault_error_policy():
    context = DataContext.get_current().copy()
    context.retried_map_errors = True
    with pytest.raises(optimizer._PlanningRejected, match="context_retried_map_errors"):
        optimizer._select_candidate(_candidate_fixture(), context)


def test_selector_rejects_extension_type_projection():
    grouped = _candidate_fixture()
    datasource = grouped.input_dependencies[0].input_dependencies[0].datasource
    datasource._file_schema = pa.schema(
        [("User", pa.int64()), ("Card", ArrowPythonObjectType())]
    )

    with pytest.raises(optimizer._PlanningRejected, match="projection_type"):
        optimizer._select_candidate(grouped, DataContext.get_current().copy())


def test_configured_fallback_reason_is_visible(caplog):
    caplog.set_level("INFO", logger=optimizer.__name__)
    optimizer.logger.addHandler(caplog.handler)
    try:
        assert (
            optimizer.try_plan_parquet_cudf_shuffle_elision(
                _candidate_fixture(explicit=False), DataContext.get_current().copy()
            )
            is None
        )
    finally:
        optimizer.logger.removeHandler(caplog.handler)
    assert "fallback: reason=partitions probe_bytes=0" in caplog.text


def test_rebatching_is_linear_and_exact():
    class FakeCudf:
        concat_calls = 0

        @classmethod
        def concat(cls, frames, ignore_index):
            cls.concat_calls += 1
            return pd.concat(frames, ignore_index=ignore_index)

    frames = [pd.DataFrame({"x": range(start, start + 3)}) for start in (0, 3, 6)]
    batches = list(optimizer._rebatched_frames(frames, 4, FakeCudf))
    assert [len(batch) for batch in batches] == [4, 4, 1]
    assert pd.concat(batches)["x"].tolist() == list(range(9))
    assert FakeCudf.concat_calls <= len(batches)


@pytest.mark.parametrize("violation", ["rows", "key"])
def test_runtime_validates_tokenizer_contract(violation):
    worker = optimizer._ParquetCudfShuffleElisionWorker.__new__(
        optimizer._ParquetCudfShuffleElisionWorker
    )

    class FakeCudf:
        DataFrame = pd.DataFrame
        concat = staticmethod(pd.concat)

    frame = pd.DataFrame({"User": [1, 2], "Card": [0, 0]})
    worker.source_kind = "local"
    worker.cudf = FakeCudf
    worker.range_key = "User"
    worker.tokenizer_batch_size = 2
    worker.tokenizer_args = ()
    worker.tokenizer_kwargs = {}
    worker._read_frames = lambda work: iter((frame,))
    if violation == "rows":
        worker.tokenizer = lambda batch: batch.iloc[:1]
    else:
        worker.tokenizer = lambda batch: batch.assign(User=[2, 1])

    batch = {"work": [pickle.dumps(optimizer._RangeWork(0, 2, ()))]}
    with pytest.raises(ValueError, match="row/key preservation"):
        list(worker(batch))


def test_s3_chunk_read_retries_matching_io_errors(monkeypatch):
    worker = optimizer._ParquetCudfShuffleElisionWorker.__new__(
        optimizer._ParquetCudfShuffleElisionWorker
    )
    attempts = 0
    identity_checks = 0

    class FakeCudf:
        @staticmethod
        def get_option(name):
            assert name == "kvikio_remote_io"
            return True

        @staticmethod
        def read_parquet(path, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("transient KvikIO failure")
            return pd.DataFrame({"User": [1], "Card": [2]})

    def verify_identity(file):
        nonlocal identity_checks
        identity_checks += 1

    monkeypatch.setattr("ray.data._internal.util.time.sleep", lambda _: None)
    worker.source_kind = "s3"
    worker.cudf = FakeCudf
    worker.projection = ("User", "Card")
    worker.range_key = "User"
    worker.retried_io_errors = ("transient KvikIO",)
    worker._verify_identity = verify_identity
    file = optimizer._FileWork(
        "bucket/data.parquet", optimizer._SourceIdentity(1, "etag"), (0,)
    )

    frames = list(worker._read_frames(optimizer._RangeWork(0, 2, (file,))))

    assert attempts == 2
    assert identity_checks == 3
    assert frames[0].to_dict("list") == {"User": [1], "Card": [2]}


def test_s3_chunk_read_exhausts_matching_io_errors(monkeypatch):
    worker = optimizer._ParquetCudfShuffleElisionWorker.__new__(
        optimizer._ParquetCudfShuffleElisionWorker
    )
    attempts = 0

    class FakeCudf:
        @staticmethod
        def get_option(name):
            return True

        @staticmethod
        def read_parquet(path, **kwargs):
            nonlocal attempts
            attempts += 1
            raise RuntimeError("transient KvikIO failure")

    monkeypatch.setattr("ray.data._internal.util.time.sleep", lambda _: None)
    worker.source_kind = "s3"
    worker.cudf = FakeCudf
    worker.projection = ("User", "Card")
    worker.range_key = "User"
    worker.retried_io_errors = ("transient KvikIO",)
    worker._verify_identity = lambda file: None
    file = optimizer._FileWork(
        "bucket/data.parquet", optimizer._SourceIdentity(1, "etag"), (0,)
    )

    with pytest.raises(RuntimeError, match="transient KvikIO"):
        list(worker._read_frames(optimizer._RangeWork(0, 2, (file,))))

    assert attempts == 10


def test_partition_result_accepts_non_generator_iterator():
    outputs = iter(({"x": np.array([1])}, {"x": np.array([2])}))

    assert [batch["x"].tolist() for batch in optimizer._one_or_generator(outputs)] == [
        [1],
        [2],
    ]


def test_partition_function_runs_once_with_sorted_composite_groups(monkeypatch):
    worker = optimizer._ParquetCudfShuffleElisionWorker.__new__(
        optimizer._ParquetCudfShuffleElisionWorker
    )

    class FakeCudf:
        DataFrame = pd.DataFrame

        @staticmethod
        def concat(frames, ignore_index=True):
            return pd.concat(frames, ignore_index=ignore_index)

    frame = pd.DataFrame(
        {
            "User": [2, 1, 1],
            "Card": [0, 2, 1],
            "value": [20, 12, 11],
        }
    )
    calls = []

    def partition(batch, context):
        calls.append((batch.copy(), context))
        return {"value": batch["value"].to_numpy()}

    def pandas_boundaries(batch, keys, cupy):
        changed = (
            batch.loc[:, list(keys)].ne(batch.loc[:, list(keys)].shift(1)).any(axis=1)
        )
        return tuple(int(index) for index in np.flatnonzero(changed)) + (len(batch),)

    monkeypatch.setattr(optimizer, "_group_boundaries", pandas_boundaries)
    worker.source_kind = "local"
    worker.cudf = FakeCudf
    worker.cupy = None
    worker.range_key = "User"
    worker.group_keys = ("User", "Card")
    worker.tokenizer_batch_size = 10
    worker.tokenizer = lambda batch: batch
    worker.tokenizer_args = ()
    worker.tokenizer_kwargs = {}
    worker.partition_fn = partition
    worker.group_args = ()
    worker.group_kwargs = {}
    worker.target_bytes = MIB
    worker._read_frames = lambda work: iter((frame,))

    descriptor = optimizer._RangeWork(1, 2, ())
    result = list(worker({"work": [pickle.dumps(descriptor)]}))
    assert len(calls) == 1
    assert calls[0][0][["User", "Card"]].values.tolist() == [
        [1, 1],
        [1, 2],
        [2, 0],
    ]
    assert calls[0][1].input_group_boundaries == (0, 1, 2, 3)
    assert result[0]["value"].tolist() == [11, 12, 20]


def test_output_splitting():
    parts = list(optimizer._split_output({"x": pa.array(range(100))}, 160))
    assert sum(len(part["x"]) for part in parts) == 100


@pytest.mark.parametrize(
    "path,expected",
    [
        ("bucket/a b", "s3://bucket/a%20b"),
        ("bucket/literal%20", "s3://bucket/literal%2520"),
        ("bucket/hash#part", "s3://bucket/hash%23part"),
        ("bucket/query?part", "s3://bucket/query%3Fpart"),
        ("bucket/unicode-é", "s3://bucket/unicode-%C3%A9"),
    ],
)
def test_s3_uri_encodes_normalized_keys_exactly_once(path, expected):
    assert optimizer._s3_uri(path) == expected


def test_credential_refresh_removes_expired_session_token(monkeypatch):
    frozen = SimpleNamespace(access_key="access", secret_key="secret", token=None)
    credentials = SimpleNamespace(get_frozen_credentials=lambda: frozen)
    session = SimpleNamespace(get_credentials=lambda: credentials)
    monkeypatch.setenv("AWS_SESSION_TOKEN", "expired")
    returned = optimizer._refresh_aws_credentials("us-west-2", session, credentials)
    assert returned == (session, credentials)
    assert "AWS_SESSION_TOKEN" not in os.environ
    assert os.environ["AWS_ACCESS_KEY_ID"] == "access"
    assert os.environ["AWS_REGION"] == "us-west-2"


def test_s3_worker_recreates_client_after_refresh(monkeypatch):
    created = []

    class Client:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Session:
        def create_client(self, service, region_name):
            created.append((service, region_name))
            return Client()

    old_client = Client()
    worker = optimizer._ParquetCudfShuffleElisionWorker.__new__(
        optimizer._ParquetCudfShuffleElisionWorker
    )
    worker.source_kind = "s3"
    worker.region = "us-west-2"
    worker._aws_session = Session()
    worker._aws_credentials = object()
    worker._s3_client = old_client
    worker.cudf = SimpleNamespace()
    worker.tokenizer_batch_size = 1
    worker._read_frames = lambda work: iter(())
    monkeypatch.setattr(optimizer, "_refresh_aws_credentials", lambda *args: args)

    descriptor = optimizer._RangeWork(0, 0, ())
    batch = {"work": [pickle.dumps(descriptor)]}
    assert list(worker(batch)) == []
    assert old_client.closed
    assert created == [("s3", "us-west-2")]
