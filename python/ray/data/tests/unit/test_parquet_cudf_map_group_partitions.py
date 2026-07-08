import os
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq
import pytest

from ray.data._internal.compute import ActorPoolStrategy
from ray.data._internal.datasource.parquet_datasource import (
    ParquetDatasource,
    _ParquetCudfDirectReadSpec,
    _parquet_cudf_direct_paths_are_ambient,
)
from ray.data._internal.logical.operators import (
    MapBatches,
    MapGroupPartitions,
    Read,
)
from ray.data._internal.planner import parquet_cudf_map_group_partitions as optimizer
from ray.data.context import DataContext

MIB = 1024**2
GIB = 1024**3


class Tokenizer:
    def __call__(self, batch):
        return batch


def partition_fn(batch, context):
    return batch


@pytest.fixture(autouse=True)
def cluster_with_four_gpus(monkeypatch):
    monkeypatch.setattr(
        "ray.data._internal.execution.operators.hash_shuffle"
        "._get_total_cluster_resources",
        lambda: SimpleNamespace(gpu=4),
    )


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


def _plan(rows, *, ranges=1, workers=1):
    return optimizer._plan_ranges(
        tuple(rows),
        target_ranges=ranges,
        num_workers=workers,
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

    datasource._projection_map = None
    result = datasource._get_parquet_cudf_direct_read_spec()
    assert result.reason is None
    assert result.spec.projection == ("User", "Card")

    datasource._projection_map = {"renamed": "User"}
    assert datasource._get_parquet_cudf_direct_read_spec().reason == "projection"

    datasource._projection_map = {"User": "User", "Card": "Card"}
    datasource._predicate_expr = object()
    assert datasource._get_parquet_cudf_direct_read_spec().reason == "predicate"


def test_range_planner_uses_exact_target_and_stable_file_order():
    rows = [_row("z", 0, 128, 1, 1), _row("a", 0, 128, 0, 0)]
    plan = _plan(rows, ranges=2)
    assert [(item.lower, item.upper) for item in plan.work] == [(0, 0), (1, 1)]
    assert [item.files[0].path for item in plan.work] == ["a", "z"]
    assert plan.amplification == (1.0, 1.0, 1.0)


def test_range_planner_warns_instead_of_rejecting_costly_data(caplog):
    caplog.set_level("WARNING", logger=optimizer.__name__)
    optimizer.logger.addHandler(caplog.handler)
    try:
        overlapping = [
            _row("a", 0, 100, 0, 0),
            _row("b", 0, 100, 0, 1),
            _row("c", 0, 100, 1, 1),
        ]
        plan = _plan(overlapping, ranges=2)
        assert max(plan.amplification) == pytest.approx(4 / 3)
        assert "read amplification" in caplog.text

        caplog.clear()
        skewed = [_row("a", 0, 193, 0, 0), _row("b", 0, 63, 1, 1)]
        plan = _plan(skewed, ranges=2)
        assert plan.skew > 1.5
        assert "range skew" in caplog.text
    finally:
        optimizer.logger.removeHandler(caplog.handler)


def test_range_planner_caps_domain_and_warns_about_underutilization(caplog):
    caplog.set_level("WARNING", logger=optimizer.__name__)
    optimizer.logger.addHandler(caplog.handler)
    try:
        plan = _plan([_row("a", 0, 256, 7, 7)], ranges=8, workers=4)
        assert [(item.lower, item.upper) for item in plan.work] == [(7, 7)]
        assert "capped ranges" in caplog.text
        assert "underutilize GPUs" in caplog.text
    finally:
        optimizer.logger.removeHandler(caplog.handler)


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
    with pytest.raises(optimizer._PlanningError, match="source changed"):
        optimizer._read_footer_summary(spec, "User")


def _candidate_fixture(
    *,
    num_partitions=2,
    compute=None,
    resources=None,
):
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
    resources = resources or {"num_cpus": 1.0, "num_gpus": 1.0}
    tokenized = MapBatches(
        Tokenizer,
        input_dependencies=[read],
        batch_size=1024,
        batch_format="cudf",
        can_modify_num_rows=True,
        compute=compute or ActorPoolStrategy(size=2),
        ray_remote_args=resources,
    )
    grouped = MapGroupPartitions(
        key=["User", "Card"],
        num_partitions=num_partitions,
        fn=partition_fn,
        input_dependencies=[tokenized],
    )
    return grouped


def test_selector_accepts_exact_pipeline_and_explicit_range_count():
    candidate = optimizer._select_candidate(
        _candidate_fixture(), DataContext.get_current().copy()
    )
    assert candidate.group_keys == ("User", "Card")
    assert candidate.target_ranges == candidate.num_workers == 2
    assert candidate.compute.max_tasks_in_flight_per_actor == 1


def test_selector_derives_two_ranges_per_bounded_pool_worker():
    candidate = optimizer._select_candidate(
        _candidate_fixture(
            num_partitions=None,
            compute=ActorPoolStrategy(min_size=1, max_size=3),
        ),
        DataContext.get_current().copy(),
    )
    assert candidate.num_workers == 3
    assert candidate.target_ranges == 6
    assert candidate.compute.min_size == 1
    assert candidate.compute.max_size == 3


def test_selector_allows_bounded_max_above_total_gpus(monkeypatch):
    monkeypatch.setattr(
        "ray.data._internal.execution.operators.hash_shuffle"
        "._get_total_cluster_resources",
        lambda: SimpleNamespace(gpu=2),
    )
    candidate = optimizer._select_candidate(
        _candidate_fixture(
            num_partitions=None,
            compute=ActorPoolStrategy(min_size=1, max_size=8),
        ),
        DataContext.get_current().copy(),
    )
    assert candidate.num_workers == 8
    assert candidate.target_ranges == 16
    assert candidate.compute.max_size == 8


def test_selector_caps_default_pool_at_detected_gpus():
    candidate = optimizer._select_candidate(
        _candidate_fixture(num_partitions=None, compute=ActorPoolStrategy()),
        DataContext.get_current().copy(),
    )
    assert candidate.num_workers == 4
    assert candidate.target_ranges == 8
    assert candidate.compute.max_size == 4


@pytest.mark.parametrize(
    ("total_gpus", "compute", "minimum"),
    [
        (0, ActorPoolStrategy(), 1),
        (1, ActorPoolStrategy(size=2), 2),
    ],
)
def test_selector_requires_actor_pool_minimum_to_fit_cluster(
    monkeypatch, total_gpus, compute, minimum
):
    monkeypatch.setattr(
        "ray.data._internal.execution.operators.hash_shuffle"
        "._get_total_cluster_resources",
        lambda: SimpleNamespace(gpu=total_gpus),
    )
    with pytest.raises(
        optimizer._PlanningError,
        match=rf"at least {minimum} total Ray GPU.*detected {total_gpus}",
    ):
        optimizer._select_candidate(
            _candidate_fixture(compute=compute), DataContext.get_current().copy()
        )


def test_selector_requires_one_gpu():
    with pytest.raises(optimizer._PlanningError, match="exactly num_gpus=1"):
        optimizer._select_candidate(
            _candidate_fixture(resources={"num_gpus": 0.5}),
            DataContext.get_current().copy(),
        )


def test_selector_rejects_checkpointing():
    context = DataContext.get_current().copy()
    context._checkpoint_config = object()
    with pytest.raises(optimizer._PlanningError, match="checkpointing"):
        optimizer._select_candidate(_candidate_fixture(), context)


def test_selector_rejects_errored_blocks_policy():
    context = DataContext.get_current().copy()
    context.max_errored_blocks = 1
    with pytest.raises(optimizer._PlanningError, match="max_errored_blocks=0"):
        optimizer._select_candidate(_candidate_fixture(), context)


def test_selector_does_not_allowlist_payload_column_types():
    grouped = _candidate_fixture()
    datasource = grouped.input_dependencies[0].input_dependencies[0].datasource
    datasource._file_schema = pa.schema(
        [("User", pa.int64()), ("Card", pa.list_(pa.int8()))]
    )

    candidate = optimizer._select_candidate(grouped, DataContext.get_current().copy())
    assert candidate.group_keys == ("User", "Card")


def test_public_planner_raises_instead_of_falling_back():
    grouped = _candidate_fixture()
    object.__setattr__(
        grouped,
        "input_dependencies",
        grouped.input_dependencies[0].input_dependencies,
    )
    with pytest.raises(ValueError, match="must immediately follow map_batches"):
        optimizer.plan_parquet_cudf_map_group_partitions(
            grouped, DataContext.get_current().copy()
        )


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


@pytest.mark.parametrize("violation", ["missing_key", "range"])
def test_runtime_validates_tokenizer_contract(violation):
    worker = optimizer._ParquetCudfMapGroupPartitionsWorker.__new__(
        optimizer._ParquetCudfMapGroupPartitionsWorker
    )

    class FakeCudf:
        DataFrame = pd.DataFrame
        concat = staticmethod(pd.concat)

    frame = pd.DataFrame({"User": [1, 2], "Card": [0, 0]})
    worker.source_kind = "local"
    worker.cudf = FakeCudf
    worker.range_key = "User"
    worker.group_keys = ("User", "Card")
    worker.tokenizer_batch_size = 2
    worker.tokenizer_args = ()
    worker.tokenizer_kwargs = {}
    worker._read_frames = lambda work: iter((frame,))
    if violation == "missing_key":
        worker.tokenizer = lambda batch: batch.drop(columns=["User"])
    else:
        worker.tokenizer = lambda batch: batch.assign(User=[3, 1])

    batch = {"work": [pickle.dumps(optimizer._RangeWork(0, 2, ()))]}
    with pytest.raises(ValueError, match="group key|outside its assigned range"):
        list(worker(batch))


def test_s3_chunk_read_retries_matching_io_errors(monkeypatch):
    worker = optimizer._ParquetCudfMapGroupPartitionsWorker.__new__(
        optimizer._ParquetCudfMapGroupPartitionsWorker
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
    worker = optimizer._ParquetCudfMapGroupPartitionsWorker.__new__(
        optimizer._ParquetCudfMapGroupPartitionsWorker
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


def test_partition_function_runs_once_after_row_count_changing_tokenizer(monkeypatch):
    worker = optimizer._ParquetCudfMapGroupPartitionsWorker.__new__(
        optimizer._ParquetCudfMapGroupPartitionsWorker
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
    worker.tokenizer = lambda batch: batch.iloc[[0, 2]]
    worker.tokenizer_args = ()
    worker.tokenizer_kwargs = {}
    worker.partition_fn = partition
    worker.target_bytes = MIB
    worker._read_frames = lambda work: iter((frame,))

    descriptor = optimizer._RangeWork(1, 2, ())
    result = list(worker({"work": [pickle.dumps(descriptor)]}))
    assert len(calls) == 1
    assert calls[0][0][["User", "Card"]].values.tolist() == [
        [1, 1],
        [2, 0],
    ]
    assert calls[0][1].input_group_boundaries == (0, 1, 2)
    assert result[0]["value"].tolist() == [11, 20]


def test_output_splitting():
    parts = list(optimizer._split_output({"x": pa.array(range(100))}, 160))
    assert sum(len(part["x"]) for part in parts) == 100


def test_rmm_pool_uses_live_free_memory():
    assert optimizer._live_rmm_pool_maximum(10 * GIB) == 7 * GIB
    assert optimizer._live_rmm_pool_maximum(3 * GIB) == 1 * GIB
    assert optimizer._live_rmm_pool_maximum(2 * GIB) == 0


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
    worker = optimizer._ParquetCudfMapGroupPartitionsWorker.__new__(
        optimizer._ParquetCudfMapGroupPartitionsWorker
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
