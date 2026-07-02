import json
from dataclasses import asdict
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest

import ray.cloudpickle as ray_pickle
from ray.data.context import DataContext
from ray.data._internal.datasource.parquet_range import (
    ParquetRowGroupMetadata,
    plan_parquet_row_groups,
)
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.planner.map_groups_partition_protocol import (
    MapGroupsPartitionContract,
    MapGroupsPartitionContext,
)
from ray.data._internal.planner.parquet_range_map_groups import (
    ParquetRangeMapGroupsActorPoolMapOperator,
    ParquetRangeMapGroupsStats,
    ParquetRangeMapGroupsTaskPoolMapOperator,
    ParquetRangeMapGroupsWork,
    _default_rmm_pool_initial_bytes,
    _initialization_timings_for_worker_range,
    _iter_exact_cudf_batches,
    _iter_group_outputs,
    _iter_partition_outputs,
    _invoke_row_preserving_tokenizer,
    _iter_group_views,
    _is_sorted_by_group_keys,
    _make_work,
    _resolve_rmm_pool_config,
    _resolve_worker_concurrency,
    _select_execution_backend,
    _validate_group_keys,
    _work_descriptor_metadata,
    build_parquet_range_map_groups_operator,
    validate_cudf_projection_schema,
)


def _row_group(
    path,
    row_group_id,
    key_min,
    key_max,
    *,
    rows=10,
    encoded_bytes=100,
    uncompressed_bytes=200,
):
    return ParquetRowGroupMetadata(
        path=path,
        source_identity=f"identity:{path}",
        row_group_id=row_group_id,
        num_rows=rows,
        key_min=key_min,
        key_max=key_max,
        null_count=0,
        encoded_bytes=encoded_bytes,
        uncompressed_bytes=uncompressed_bytes,
    )


def _source_schema():
    return pa.schema(
        [
            pa.field("User", pa.int64()),
            pa.field("Card", pa.int32()),
            pa.field("token", pa.uint16()),
        ]
    )


def _identity_batch(batch):
    return batch


def _identity_partition(batch, _context):
    return batch


def _builder_udf_ops(num_partitions):
    common = {
        "fn": _identity_batch,
        "fn_args": (),
        "fn_kwargs": {},
        "fn_constructor_args": (),
        "fn_constructor_kwargs": {},
        "compute": None,
        "zero_copy_batch": True,
    }
    tokenizer_op = SimpleNamespace(**common, batch_size=10)
    map_groups_op = SimpleNamespace(**common, num_partitions=num_partitions)
    return tokenizer_op, map_groups_op


def _builder_layout(num_partitions):
    return plan_parquet_row_groups(
        [_row_group("data.parquet", 0, 0, num_partitions * 2 - 1, rows=100)],
        num_partitions=num_partitions,
    )


def _stats(partition_id=0):
    return ParquetRangeMapGroupsStats(
        partition_id=partition_id,
        lower_bound=0,
        upper_bound=9,
        selected_row_groups=2,
        estimated_scanned_rows=20,
        input_rows=18,
        tokenized_rows=18,
        groups_invoked=4,
        output_rows=8,
        output_blocks=2,
        read_time_s=1.0,
        map_time_s=2.0,
        sort_time_s=3.0,
        group_time_s=4.0,
        output_time_s=5.0,
        gpu_peak_memory_bytes=1024,
        host_transfer_bytes=256,
        rmm_pool_initial_bytes=8 * 1024**3,
        rmm_pool_maximum_bytes=24 * 1024**3,
        rmm_pool_reserved_peak_bytes=16 * 1024**3,
    )


def test_partition_execution_uses_a_smaller_startup_pool_by_default():
    assert _default_rmm_pool_initial_bytes(None) == 8 * 1024**3
    assert _default_rmm_pool_initial_bytes(object()) == 1 * 1024**3


@pytest.mark.parametrize(
    (
        "has_partition_contract",
        "num_ranges",
        "worker_concurrency",
        "expected",
    ),
    [
        (True, 4, None, ("task_pool", 4, False)),
        (True, 4, 4, ("task_pool", 4, False)),
        (True, 8, 4, ("actor_pool", 4, True)),
        # Don't create idle workers if the configured pool exceeds the range count.
        (True, 4, 8, ("task_pool", 4, False)),
        # Per-group execution uses the same bounded actor pool and serial reuse.
        (False, 8, 4, ("actor_pool", 4, True)),
    ],
)
def test_select_execution_backend(
    has_partition_contract,
    num_ranges,
    worker_concurrency,
    expected,
):
    assert (
        _select_execution_backend(
            has_partition_contract=has_partition_contract,
            num_ranges=num_ranges,
            worker_concurrency=worker_concurrency,
        )
        == expected
    )


@pytest.mark.parametrize("worker_concurrency", [True, False, 0, -1, 1.5, "4"])
def test_resolve_worker_concurrency_rejects_invalid_values(worker_concurrency):
    with pytest.raises(ValueError, match="worker_concurrency must be a positive"):
        _resolve_worker_concurrency(worker_concurrency, num_ranges=4)


def test_resolve_worker_concurrency_requires_a_nonempty_plan():
    with pytest.raises(ValueError, match="at least one range"):
        _resolve_worker_concurrency(1, num_ranges=0)


def test_initialization_timings_are_only_charged_to_first_worker_range():
    timings = {"total": 1.5, "rmm": 1.0, "tokenizer": 0.5}

    assert _initialization_timings_for_worker_range(timings, 0) is timings
    assert _initialization_timings_for_worker_range(timings, 1) is None
    assert _initialization_timings_for_worker_range(timings, 10) is None
    with pytest.raises(ValueError, match="worker_range_index must be nonnegative"):
        _initialization_timings_for_worker_range(timings, -1)


def test_builder_uses_fixed_actor_pool_for_oversubscribed_partition_ranges():
    layout = _builder_layout(num_partitions=4)
    tokenizer_op, map_groups_op = _builder_udf_ops(num_partitions=4)

    op = build_parquet_range_map_groups_operator(
        layout=layout,
        footer_planning_time_s=0.1,
        projection=("User",),
        partition_key="User",
        group_keys=("User",),
        source_schema=pa.schema([pa.field("User", pa.int64())]),
        tokenizer_op=tokenizer_op,
        map_groups_op=map_groups_op,
        partition_contract=MapGroupsPartitionContract(
            udf=_identity_partition,
            batch_format="cudf",
        ),
        data_context=DataContext.get_current(),
        ray_remote_args={"num_gpus": 1},
        gpu_memory_budget_bytes=8 * 1024**3,
        worker_concurrency=2,
    )

    assert isinstance(op, ParquetRangeMapGroupsActorPoolMapOperator)
    assert op._actor_pool.min_size() == 2
    assert op._actor_pool.max_size() == 2
    assert op._actor_pool.max_tasks_in_flight_per_actor() == 1
    assert op._ray_remote_args["max_restarts"] == 0
    assert op._ray_remote_args["max_task_retries"] == 0
    assert op._parquet_range_plan_metrics["execution_backend"] == "actor_pool"
    assert op._parquet_range_plan_metrics["worker_concurrency"] == 2
    assert op._parquet_range_plan_metrics["actor_reuse_enabled"] is True
    assert op._parquet_range_plan_metrics["max_ranges_per_worker"] == 2


def test_builder_bounds_ordinary_actor_pool_and_reuses_workers():
    layout = _builder_layout(num_partitions=8)
    tokenizer_op, map_groups_op = _builder_udf_ops(num_partitions=8)

    op = build_parquet_range_map_groups_operator(
        layout=layout,
        footer_planning_time_s=0.1,
        projection=("User",),
        partition_key="User",
        group_keys=("User",),
        source_schema=pa.schema([pa.field("User", pa.int64())]),
        tokenizer_op=tokenizer_op,
        map_groups_op=map_groups_op,
        data_context=DataContext.get_current(),
        ray_remote_args={"num_gpus": 1},
        gpu_memory_budget_bytes=8 * 1024**3,
        worker_concurrency=4,
    )

    assert isinstance(op, ParquetRangeMapGroupsActorPoolMapOperator)
    assert op._actor_pool.min_size() == 4
    assert op._actor_pool.max_size() == 4
    assert op._actor_pool.max_tasks_in_flight_per_actor() == 1
    assert op._ray_remote_args["max_restarts"] == 0
    assert op._ray_remote_args["max_task_retries"] == 0
    assert op._parquet_range_plan_metrics["group_execution_mode"] == "per_group"
    assert op._parquet_range_plan_metrics["worker_concurrency"] == 4
    assert op._parquet_range_plan_metrics["actor_reuse_enabled"] is True
    assert op._parquet_range_plan_metrics["max_ranges_per_worker"] == 2


def test_builder_keeps_task_pool_for_one_partition_range_per_worker():
    layout = _builder_layout(num_partitions=4)
    tokenizer_op, map_groups_op = _builder_udf_ops(num_partitions=4)

    op = build_parquet_range_map_groups_operator(
        layout=layout,
        footer_planning_time_s=0.1,
        projection=("User",),
        partition_key="User",
        group_keys=("User",),
        source_schema=pa.schema([pa.field("User", pa.int64())]),
        tokenizer_op=tokenizer_op,
        map_groups_op=map_groups_op,
        partition_contract=MapGroupsPartitionContract(
            udf=_identity_partition,
            batch_format="cudf",
        ),
        data_context=DataContext.get_current(),
        ray_remote_args={"num_gpus": 1},
        gpu_memory_budget_bytes=8 * 1024**3,
        worker_concurrency=4,
    )

    assert isinstance(op, ParquetRangeMapGroupsTaskPoolMapOperator)
    assert op._max_concurrency == 4
    assert op._ray_remote_args["max_retries"] == 0
    assert op._parquet_range_plan_metrics["execution_backend"] == "task_pool"
    assert op._parquet_range_plan_metrics["worker_concurrency"] == 4
    assert op._parquet_range_plan_metrics["actor_reuse_enabled"] is False
    assert op._parquet_range_plan_metrics["max_ranges_per_worker"] == 1


def test_reusable_actor_transform_initializes_once_and_charges_first_range(
    monkeypatch,
):
    import ray.data._internal.planner.parquet_range_map_groups as backend

    layout = _builder_layout(num_partitions=2)
    tokenizer_op, map_groups_op = _builder_udf_ops(num_partitions=2)
    configured_pools = []
    range_calls = []

    def configure_pool(**kwargs):
        state = SimpleNamespace(config=kwargs)
        configured_pools.append(state)
        return state

    def execute_range(work, **kwargs):
        range_calls.append((work.partition_id, kwargs))
        if False:
            yield None
        return _stats(work.partition_id)

    monkeypatch.setattr(backend, "_configure_rmm_pool", configure_pool)
    monkeypatch.setattr(backend, "_execute_range", execute_range)

    op = build_parquet_range_map_groups_operator(
        layout=layout,
        footer_planning_time_s=0.1,
        projection=("User",),
        partition_key="User",
        group_keys=("User",),
        source_schema=pa.schema([pa.field("User", pa.int64())]),
        tokenizer_op=tokenizer_op,
        map_groups_op=map_groups_op,
        partition_contract=MapGroupsPartitionContract(
            udf=_identity_partition,
            batch_format="cudf",
        ),
        data_context=DataContext.get_current(),
        ray_remote_args={"num_gpus": 1},
        gpu_memory_budget_bytes=8 * 1024**3,
        worker_concurrency=1,
    )
    assert isinstance(op, ParquetRangeMapGroupsActorPoolMapOperator)

    transformer = op._map_transformer
    transformer.init()
    works = _make_work(
        layout,
        projection=("User",),
        partition_key="User",
        group_keys=("User",),
        source_schema=_source_schema(),
    )
    reported_stats = []
    for task_index, work in enumerate(works):
        assert (
            list(
                transformer.apply_transform(
                    iter([work]),
                    TaskContext(task_idx=task_index, op_name="range"),
                    reported_stats.append,
                )
            )
            == []
        )

    assert len(configured_pools) == 1
    assert [partition_id for partition_id, _ in range_calls] == [0, 1]
    first_kwargs = range_calls[0][1]
    second_kwargs = range_calls[1][1]
    assert first_kwargs["worker_range_index"] == 0
    assert first_kwargs["initialization_timings"]["total"] >= 0
    assert second_kwargs["worker_range_index"] == 1
    assert second_kwargs["initialization_timings"] is None
    assert [stats.partition_id for stats in reported_stats] == [0, 1]


def test_rmm_pool_config_respects_budget_reserve_and_alignment():
    gib = 1024**3

    config = _resolve_rmm_pool_config(
        free_bytes=30 * gib + 127,
        gpu_memory_budget_bytes=24 * gib + 127,
        initial_bytes=8 * gib + 127,
        reserve_bytes=2 * gib,
    )

    assert config.initial_bytes == 8 * gib
    assert config.maximum_bytes == 24 * gib
    assert config.initial_bytes % 256 == 0
    assert config.maximum_bytes % 256 == 0


def test_rmm_pool_config_clamps_initial_size_to_live_limit():
    gib = 1024**3

    config = _resolve_rmm_pool_config(
        free_bytes=7 * gib,
        gpu_memory_budget_bytes=12 * gib,
        initial_bytes=8 * gib,
        reserve_bytes=2 * gib,
    )

    assert config.initial_bytes == 5 * gib
    assert config.maximum_bytes == 5 * gib


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {
                "free_bytes": 0,
                "gpu_memory_budget_bytes": 1,
                "initial_bytes": 1,
                "reserve_bytes": 0,
            },
            RuntimeError,
            "no free GPU memory",
        ),
        (
            {
                "free_bytes": 1,
                "gpu_memory_budget_bytes": 0,
                "initial_bytes": 1,
                "reserve_bytes": 0,
            },
            ValueError,
            "gpu_memory_budget_bytes",
        ),
        (
            {
                "free_bytes": 1,
                "gpu_memory_budget_bytes": 1,
                "initial_bytes": 0,
                "reserve_bytes": 0,
            },
            ValueError,
            "initial pool size",
        ),
        (
            {
                "free_bytes": 1,
                "gpu_memory_budget_bytes": 1,
                "initial_bytes": 1,
                "reserve_bytes": -1,
            },
            ValueError,
            "reserved GPU memory",
        ),
        (
            {
                "free_bytes": 1024,
                "gpu_memory_budget_bytes": 1024,
                "initial_bytes": 1024,
                "reserve_bytes": 1024,
            },
            RuntimeError,
            "Insufficient free GPU memory",
        ),
    ],
)
def test_rmm_pool_config_rejects_invalid_or_exhausted_bounds(kwargs, error, match):
    with pytest.raises(error, match=match):
        _resolve_rmm_pool_config(**kwargs)


def test_schema_preflight_accepts_supported_primitive_projection():
    schema = pa.schema(
        [
            pa.field("integer", pa.int64()),
            pa.field("floating", pa.float32()),
            pa.field("boolean", pa.bool_()),
            pa.field("string", pa.string()),
            pa.field("large_string", pa.large_string()),
            pa.field("timestamp", pa.timestamp("us")),
            pa.field("unprojected_binary", pa.binary()),
        ]
    )

    assert (
        validate_cudf_projection_schema(
            schema,
            (
                "integer",
                "floating",
                "boolean",
                "string",
                "large_string",
                "timestamp",
            ),
        )
        is None
    )


@pytest.mark.parametrize(
    ("schema", "projection", "reason"),
    [
        (None, ("value",), "missing_source_schema"),
        (pa.schema([("value", pa.int64())]), (), "empty_projection"),
        (
            pa.schema([("value", pa.int64())]),
            ("missing",),
            "invalid_projection_schema",
        ),
        (
            pa.schema([("value", pa.int64()), ("value", pa.int32())]),
            ("value",),
            "invalid_projection_schema",
        ),
        (
            pa.schema([("value", pa.binary())]),
            ("value",),
            "unsupported_projection_type",
        ),
        (
            pa.schema([("value", pa.list_(pa.int64()))]),
            ("value",),
            "unsupported_projection_type",
        ),
        (
            pa.schema([("value", pa.timestamp("ns", tz="UTC"))]),
            ("value",),
            "unsupported_projection_type",
        ),
    ],
)
def test_schema_preflight_fails_closed(schema, projection, reason):
    assert validate_cudf_projection_schema(schema, projection) == reason


class _FakeCudf:
    concat_calls = []

    @classmethod
    def concat(cls, frames, *, ignore_index):
        assert ignore_index
        cls.concat_calls.append(tuple(len(frame) for frame in frames))
        return pd.concat(frames, ignore_index=True)


def test_tokenizer_runtime_contract_accepts_exact_elementwise_preservation(monkeypatch):
    import ray.data._internal.planner.parquet_range_map_groups as backend

    monkeypatch.setattr(
        backend, "_is_cudf_dataframe", lambda value: isinstance(value, pd.DataFrame)
    )
    batch = pd.DataFrame({"User": [1, 2], "Card": [3, 4]})

    result = _invoke_row_preserving_tokenizer(
        batch,
        lambda frame: frame.assign(token=[5, 6]),
        partition_key="User",
        zero_copy_batch=False,
        cudf=_FakeCudf,
    )

    assert result.to_dict("list") == {
        "User": [1, 2],
        "Card": [3, 4],
        "token": [5, 6],
    }


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        (lambda frame: frame.iloc[:1], "udf_modifying_row_count=False"),
        (lambda frame: frame.drop(columns=["User"]), "removed preserved partition"),
        (
            lambda frame: frame.assign(User=[2, 1]),
            "changed preserved partition column",
        ),
        (lambda frame: frame.assign(User=[1, None]), "contains nulls"),
    ],
)
def test_tokenizer_runtime_contract_rejects_violations(monkeypatch, tokenizer, message):
    import ray.data._internal.planner.parquet_range_map_groups as backend

    monkeypatch.setattr(
        backend, "_is_cudf_dataframe", lambda value: isinstance(value, pd.DataFrame)
    )
    batch = pd.DataFrame({"User": [1, 2], "Card": [3, 4]})

    with pytest.raises(ValueError, match=message):
        _invoke_row_preserving_tokenizer(
            batch,
            tokenizer,
            partition_key="User",
            zero_copy_batch=True,
            cudf=_FakeCudf,
        )


def test_tokenizer_runtime_contract_rejects_in_place_partition_mutation(monkeypatch):
    import ray.data._internal.planner.parquet_range_map_groups as backend

    monkeypatch.setattr(
        backend, "_is_cudf_dataframe", lambda value: isinstance(value, pd.DataFrame)
    )
    batch = pd.DataFrame({"User": [1, 2], "Card": [3, 4]})

    def mutate_in_place(frame):
        frame["User"] = [2, 1]
        return frame

    with pytest.raises(ValueError, match="changed preserved partition column"):
        _invoke_row_preserving_tokenizer(
            batch,
            mutate_in_place,
            partition_key="User",
            zero_copy_batch=True,
            cudf=_FakeCudf,
        )


def test_partition_udf_receives_immutable_group_layout_and_generator_outputs():
    context = MapGroupsPartitionContext(
        group_keys=("User", "Card"),
        input_group_boundaries=(0, 2, 5),
    )
    seen = []

    def partition_fn(partition, groups):
        seen.append((partition, groups))
        yield {"value": [1]}
        yield {"value": [2]}

    assert list(_iter_partition_outputs(partition_fn, "partition", context)) == [
        {"value": [1]},
        {"value": [2]},
    ]
    assert seen == [("partition", context)]
    assert context.num_groups == 2


def test_partition_udf_empty_output_and_exceptions_are_not_replayed():
    context = MapGroupsPartitionContext(
        group_keys=("User",),
        input_group_boundaries=(0, 1),
    )
    calls = []

    def empty(partition, groups):
        calls.append((partition, groups))
        if False:
            yield partition

    assert list(_iter_partition_outputs(empty, "partition", context)) == []
    assert calls == [("partition", context)]

    def fails(partition, groups):
        calls.append((partition, groups))
        raise RuntimeError("partition failure")

    with pytest.raises(RuntimeError, match="partition failure"):
        list(_iter_partition_outputs(fails, "partition", context))
    assert calls == [("partition", context), ("partition", context)]


def test_tokenizer_runtime_contract_rejects_empty_generator_and_host_output(
    monkeypatch,
):
    import ray.data._internal.planner.parquet_range_map_groups as backend

    batch = pd.DataFrame({"User": [1], "Card": [3]})

    def empty(_):
        if False:
            yield None

    monkeypatch.setattr(backend, "_is_cudf_dataframe", lambda _: False)
    with pytest.raises(ValueError, match="returned no rows"):
        _invoke_row_preserving_tokenizer(
            batch,
            empty,
            partition_key="User",
            zero_copy_batch=True,
            cudf=_FakeCudf,
        )
    with pytest.raises(TypeError, match="cudf.DataFrame"):
        _invoke_row_preserving_tokenizer(
            batch,
            lambda frame: frame,
            partition_key="User",
            zero_copy_batch=True,
            cudf=_FakeCudf,
        )


def test_exact_batching_combines_splits_and_flushes_pandas_frames():
    _FakeCudf.concat_calls = []
    frames = [
        pd.DataFrame({"value": [0, 1]}),
        pd.DataFrame({"value": pd.Series(dtype="int64")}),
        pd.DataFrame({"value": [2, 3, 4, 5, 6]}),
        pd.DataFrame({"value": [7]}),
    ]

    batches = list(_iter_exact_cudf_batches(frames, 3, _FakeCudf))

    assert [len(batch) for batch in batches] == [3, 3, 2]
    assert [batch["value"].tolist() for batch in batches] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]
    assert all(batch.index.tolist() == list(range(len(batch))) for batch in batches)
    # Only the rows needed to finish a batch are concatenated; the four-row
    # source remainder is retained by cursor rather than recopied.
    assert _FakeCudf.concat_calls == [(2, 1), (1, 1)]


def test_exact_batching_avoids_concat_for_one_frame_and_skips_empty_input():
    _FakeCudf.concat_calls = []
    frame = pd.DataFrame({"value": list(range(6))}, index=range(10, 16))

    batches = list(_iter_exact_cudf_batches([frame], 3, _FakeCudf))
    assert [batch["value"].tolist() for batch in batches] == [
        [0, 1, 2],
        [3, 4, 5],
    ]
    assert all(batch.index.tolist() == [0, 1, 2] for batch in batches)
    assert _FakeCudf.concat_calls == []

    assert (
        list(_iter_exact_cudf_batches([pd.DataFrame({"value": []})], 3, _FakeCudf))
        == []
    )


def test_group_output_single_result_and_empty_dataframe_are_single_batches():
    group = pd.DataFrame({"User": [1, 1], "value": [2, 3]})
    expected = pd.DataFrame({"result": [5]})

    outputs = list(_iter_group_outputs(lambda _: expected, group))
    assert len(outputs) == 1
    assert outputs[0] is expected

    empty = pd.DataFrame({"result": pd.Series(dtype="int64")})
    outputs = list(_iter_group_outputs(lambda _: empty, group))
    assert len(outputs) == 1
    assert outputs[0] is empty


def test_group_key_order_detection_accepts_trivial_frames():
    assert _is_sorted_by_group_keys(
        pd.DataFrame({"User": [1], "Card": [2]}), ("User", "Card")
    )


def test_group_key_order_detection_sorts_when_compiled_check_is_unavailable():
    # Pandas has no ``to_pylibcudf`` bridge and exercises the compatibility
    # path. A False result conservatively requests the ordinary sort.
    assert not _is_sorted_by_group_keys(
        pd.DataFrame({"User": [1, 1], "Card": [1, 2]}), ("User", "Card")
    )


def test_group_views_use_one_segmented_split_when_available():
    class Frame:
        def __init__(self):
            self.calls = []

        def _split(self, boundaries, *, keep_index):
            self.calls.append((boundaries, keep_index))
            return ["first", "second", "third"]

    frame = Frame()
    assert list(_iter_group_views(frame, [0, 2, 5, 9])) == [
        "first",
        "second",
        "third",
    ]
    assert frame.calls == [([2, 5], True)]


def test_group_views_fall_back_to_exact_slices():
    frame = pd.DataFrame({"value": list(range(6))})

    groups = list(_iter_group_views(frame, [0, 2, 5, 6]))

    assert [group["value"].tolist() for group in groups] == [
        [0, 1],
        [2, 3, 4],
        [5],
    ]


def test_group_views_reject_unexpected_segment_count():
    class Frame:
        def _split(self, boundaries, *, keep_index):
            return ["only-one"]

    with pytest.raises(RuntimeError, match="unexpected number"):
        list(_iter_group_views(Frame(), [0, 2, 5]))


def test_group_output_generator_and_empty_generator_semantics():
    group = pd.DataFrame({"User": [1]})

    def generate(_):
        yield pd.DataFrame({"result": [1]})
        yield pd.DataFrame({"result": [2]})

    outputs = list(_iter_group_outputs(generate, group))
    assert [output["result"].item() for output in outputs] == [1, 2]

    def generate_nothing(_):
        if False:
            yield pd.DataFrame()

    assert list(_iter_group_outputs(generate_nothing, group)) == []


def test_group_output_custom_iterator_semantics():
    class OutputIterator:
        def __init__(self, outputs):
            self._outputs = iter(outputs)

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._outputs)

    expected = [
        pd.DataFrame({"result": [1]}),
        pd.DataFrame({"result": [2]}),
    ]
    outputs = list(
        _iter_group_outputs(
            lambda _: OutputIterator(expected),
            pd.DataFrame({"User": [1]}),
        )
    )
    assert outputs == expected


def test_group_output_propagates_udf_and_iterator_exceptions():
    group = pd.DataFrame({"User": [1]})
    direct_error = RuntimeError("direct failure")

    def fail_directly(_):
        raise direct_error

    with pytest.raises(RuntimeError) as exc_info:
        list(_iter_group_outputs(fail_directly, group))
    assert exc_info.value is direct_error

    iterator_error = RuntimeError("iterator failure")

    def fail_during_iteration(_):
        yield pd.DataFrame({"result": [1]})
        raise iterator_error

    outputs = _iter_group_outputs(fail_during_iteration, group)
    assert next(outputs)["result"].item() == 1
    with pytest.raises(RuntimeError) as exc_info:
        next(outputs)
    assert exc_info.value is iterator_error


def test_group_output_validates_every_iterator_item():
    def invalid_output(_):
        yield pd.DataFrame({"result": [1]})
        yield object()

    outputs = _iter_group_outputs(invalid_output, pd.DataFrame({"User": [1]}))
    assert next(outputs)["result"].item() == 1
    with pytest.raises(ValueError, match="map_batches"):
        next(outputs)


def test_group_key_runtime_validation_reports_every_missing_key():
    with pytest.raises(ValueError, match=r"\('Card', 'Merchant'\)"):
        _validate_group_keys(
            pd.DataFrame({"User": [1], "token": [2]}),
            ("User", "Card", "Merchant"),
        )


def test_work_and_stats_are_ray_serializable():
    layout = plan_parquet_row_groups(
        [_row_group("data.parquet", 0, 0, 9)],
        num_partitions=1,
    )
    work = _make_work(
        layout,
        projection=("User", "Card"),
        partition_key="User",
        group_keys=("User", "Card"),
        source_schema=_source_schema(),
    )[0]
    stats = _stats()

    restored_work = ray_pickle.loads(ray_pickle.dumps(work))
    restored_stats = ray_pickle.loads(ray_pickle.dumps(stats))

    assert isinstance(restored_work, ParquetRangeMapGroupsWork)
    assert restored_work == work
    assert restored_work.source_schema.equals(_source_schema())
    assert isinstance(restored_stats, ParquetRangeMapGroupsStats)
    assert restored_stats == stats
    assert json.loads(json.dumps(asdict(stats)))["groups_invoked"] == 4

    descriptor_metadata = _work_descriptor_metadata(work)
    assert descriptor_metadata.num_rows == 1
    assert descriptor_metadata.size_bytes == len(ray_pickle.dumps(work))
    assert descriptor_metadata.input_files == ("data.parquet",)


def test_make_work_preserves_range_and_fragment_shape_including_empty_ranges():
    layout = plan_parquet_row_groups(
        [
            _row_group("first.parquet", 0, 0, 0, rows=2),
            _row_group(
                "last.parquet",
                3,
                99,
                99,
                rows=3,
                encoded_bytes=150,
                uncompressed_bytes=300,
            ),
        ],
        num_partitions=4,
    )
    schema = _source_schema()

    works = _make_work(
        layout,
        projection=("User", "Card"),
        partition_key="User",
        group_keys=("User", "Card"),
        source_schema=schema,
    )

    assert [
        (work.partition_id, work.lower_bound, work.upper_bound) for work in works
    ] == [
        (0, 0, 24),
        (1, 25, 49),
        (2, 50, 74),
        (3, 75, 99),
    ]
    assert works[0].fragments[0].path == "first.parquet"
    assert works[0].fragments[0].source_identity == "identity:first.parquet"
    assert works[1].fragments == ()
    assert works[2].fragments == ()
    assert works[3].fragments[0].row_group_ids == (3,)
    assert [work.estimated_scanned_rows for work in works] == [2, 0, 0, 3]
    assert [work.estimated_encoded_bytes for work in works] == [100, 0, 0, 150]
    assert [work.estimated_uncompressed_bytes for work in works] == [
        200,
        0,
        0,
        300,
    ]
    assert all(work.projection == ("User", "Card") for work in works)
    assert all(work.partition_key == "User" for work in works)
    assert all(work.group_keys == ("User", "Card") for work in works)
    assert all(work.source_schema is schema for work in works)


def test_builder_rejects_layout_that_cannot_shape_one_work_per_partition():
    layout = plan_parquet_row_groups(
        [_row_group("data.parquet", 0, 0, 1)],
        num_partitions=4,
    )
    assert len(layout.partitions) == 2

    with pytest.raises(
        ValueError,
        match="exactly one range per requested partition",
    ):
        build_parquet_range_map_groups_operator(
            layout=layout,
            footer_planning_time_s=0.1,
            projection=("User", "Card"),
            partition_key="User",
            group_keys=("User", "Card"),
            source_schema=_source_schema(),
            tokenizer_op=None,
            map_groups_op=SimpleNamespace(num_partitions=4),
            data_context=None,
            ray_remote_args={},
        )
