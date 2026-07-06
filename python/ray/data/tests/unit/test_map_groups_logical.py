from dataclasses import FrozenInstanceError
from functools import partial

import pytest

from ray.data import (
    ActorPoolStrategy,
    MapGroupsPartitionContext,
    ParquetCudfShuffleElisionConfig,
    TaskPoolStrategy,
)
from ray.data._internal.logical.interfaces import LogicalPlan
from ray.data._internal.logical.operators import (
    InputData,
    MapBatches,
    MapGroups,
    Repartition,
    Sort,
)
from ray.data._internal.logical.rules.combine_shuffles import CombineShuffles
from ray.data._internal.planner import create_planner
from ray.data._internal.planner.exchange.sort_task_spec import SortKey
from ray.data._internal.planner.plan_map_groups_op import (
    _build_map_groups_udf,
    _create_shuffle_op,
)
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.dataset import Dataset
from ray.data.grouped_data import GroupedData


def identity(batch):
    return batch


def generator_identity(batch):
    yield batch


class Identity:
    def __call__(self, batch):
        return batch


class _FakeDataset:
    def __init__(self, logical_plan, context):
        self._logical_plan = logical_plan
        self._context = context

    @property
    def context(self):
        return self._context


@pytest.fixture
def data_context():
    context = DataContext.get_current()
    previous_strategy = context.shuffle_strategy
    previous_parallelism = context.default_hash_shuffle_parallelism
    try:
        yield context
    finally:
        context.shuffle_strategy = previous_strategy
        context.default_hash_shuffle_parallelism = previous_parallelism


@pytest.fixture
def source_dataset(monkeypatch, data_context):
    def from_parent(parent, logical_plan):
        return _FakeDataset(logical_plan, parent.context)

    monkeypatch.setattr(Dataset, "_from_parent", staticmethod(from_parent))
    source_op = InputData(input_data=[])
    return _FakeDataset(LogicalPlan(source_op, data_context), data_context)


def _map_groups(source_dataset, key, *, num_partitions=None, **kwargs):
    return GroupedData(
        source_dataset,
        key,
        num_partitions=num_partitions,
    ).map_groups(identity, **kwargs)


def _config(**overrides):
    values = {
        "partition_fn": identity,
        "shuffle_bytes_per_input_row": 16.0,
        "peak_gpu_bytes_per_input_row": 24.0,
        "gpu_memory_bytes": 16 * 1024**3,
    }
    values.update(overrides)
    return ParquetCudfShuffleElisionConfig(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("shuffle_bytes_per_input_row", 0),
        ("shuffle_bytes_per_input_row", -1),
        ("shuffle_bytes_per_input_row", float("inf")),
        ("shuffle_bytes_per_input_row", float("nan")),
        ("shuffle_bytes_per_input_row", True),
        ("peak_gpu_bytes_per_input_row", 0),
        ("peak_gpu_bytes_per_input_row", float("nan")),
        ("gpu_memory_bytes", 0),
        ("gpu_memory_bytes", 1.5),
        ("gpu_memory_bytes", True),
    ],
)
def test_shuffle_elision_config_rejects_invalid_estimates(field, value):
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_shuffle_elision_config_requires_callable_partition_fn():
    with pytest.raises(TypeError, match="partition_fn must be callable"):
        _config(partition_fn=None)


def test_shuffle_elision_public_types_are_frozen():
    config = _config()
    context = MapGroupsPartitionContext(
        group_keys=("user", "card"),
        input_group_boundaries=(0, 3, 8),
    )

    assert context.num_groups == 2
    with pytest.raises(FrozenInstanceError):
        config.gpu_memory_bytes = 1
    with pytest.raises(FrozenInstanceError):
        context.group_keys = ("other",)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_keys": (), "input_group_boundaries": (0,)},
        {"group_keys": ("key",), "input_group_boundaries": ()},
        {"group_keys": ("key",), "input_group_boundaries": (1, 2)},
        {"group_keys": ("key",), "input_group_boundaries": (0, 2, 2)},
        {"group_keys": ("key",), "input_group_boundaries": (0, True)},
    ],
)
def test_partition_context_rejects_invalid_boundaries(kwargs):
    with pytest.raises(ValueError):
        MapGroupsPartitionContext(**kwargs)


def test_map_groups_builds_first_class_logical_operator(data_context, source_dataset):
    data_context.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    data_context.default_hash_shuffle_parallelism = 7

    def remote_args_fn():
        return {"resources": {"dynamic": 1}}

    dataset = _map_groups(
        source_dataset,
        "id",
        batch_format="pyarrow",
        zero_copy_batch=False,
        fn_args=(1,),
        fn_kwargs={"keyword": 2},
        compute=TaskPoolStrategy(size=3),
        num_cpus=2,
        memory=1024,
        ray_remote_args_fn=remote_args_fn,
        resources={"custom": 1},
    )
    op = dataset._logical_plan.dag

    assert isinstance(op, MapGroups)
    assert op.input_dependencies == [source_dataset._logical_plan.dag]
    assert op.key == "id"
    assert op.num_partitions == 7
    assert not op.num_partitions_explicit
    assert op.shuffle_strategy == ShuffleStrategy.HASH_SHUFFLE
    assert op.fn is identity
    assert op.batch_format == "pyarrow"
    assert not op.zero_copy_batch
    assert op.fn_args == (1,)
    assert op.fn_kwargs == {"keyword": 2}
    assert op.compute == TaskPoolStrategy(size=3)
    assert op.ray_remote_args == {
        "resources": {"custom": 1},
        "num_cpus": 2,
        "memory": 1024,
    }
    assert op.ray_remote_args_fn is remote_args_fn
    logical_ops = list(dataset._logical_plan.dag.post_order_iter())
    assert type(logical_ops[-1]) is MapGroups
    assert not any(
        isinstance(item, (MapBatches, Repartition, Sort)) for item in logical_ops
    )


@pytest.mark.parametrize(
    ("strategy", "key", "expected_type"),
    [
        (ShuffleStrategy.HASH_SHUFFLE, "id", Repartition),
        (ShuffleStrategy.GPU_SHUFFLE, "id", Repartition),
        (ShuffleStrategy.SORT_SHUFFLE_PULL_BASED, "id", Sort),
        (ShuffleStrategy.SORT_SHUFFLE_PUSH_BASED, "id", Sort),
        (ShuffleStrategy.HASH_SHUFFLE, None, Repartition),
    ],
)
def test_stock_shuffle_lowering_shape(
    data_context, source_dataset, strategy, key, expected_type
):
    data_context.shuffle_strategy = strategy
    op = _map_groups(
        source_dataset,
        key,
        num_partitions=2,
    )._logical_plan.dag

    shuffle_op = _create_shuffle_op(op)
    assert isinstance(shuffle_op, expected_type)
    if key is None:
        assert shuffle_op.num_outputs == 1
    elif isinstance(shuffle_op, Repartition):
        assert shuffle_op.keys == ["id"]
        assert shuffle_op.num_outputs == 2
        assert shuffle_op.sort
    else:
        assert shuffle_op.sort_key.get_columns() == ["id"]


@pytest.mark.parametrize(
    ("fn", "expected_name", "expected_compute"),
    [
        (identity, "identity", TaskPoolStrategy),
        (generator_identity, "generator_identity", TaskPoolStrategy),
        (partial(identity), "identity", TaskPoolStrategy),
        (Identity, "Identity", ActorPoolStrategy),
    ],
)
def test_map_groups_preserves_udf_name_and_compute(
    source_dataset, fn, expected_name, expected_compute
):
    op = (
        GroupedData(
            source_dataset,
            "id",
            num_partitions=None,
        )
        .map_groups(fn)
        ._logical_plan.dag
    )

    assert op.name == f"MapGroups({expected_name})"
    assert _build_map_groups_udf(op).__name__ == expected_name
    assert isinstance(op.compute, expected_compute)


def test_normal_planner_lowering_uses_stock_physical_operators(
    data_context, source_dataset
):
    data_context.shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    dataset = _map_groups(
        source_dataset,
        "id",
        num_partitions=2,
        compute=TaskPoolStrategy(size=2),
        num_cpus=1,
    )

    physical_plan, _ = create_planner().plan(dataset._logical_plan)
    reachable = set(physical_plan.dag.post_order_iter())

    assert physical_plan.dag.name == "MapBatches(identity)"
    assert set(physical_plan.op_map) == reachable
    assert isinstance(physical_plan.op_map[physical_plan.dag], MapBatches)
    assert any(
        isinstance(logical_op, Repartition)
        for logical_op in physical_plan.op_map.values()
    )
    assert not any(
        isinstance(logical_op, MapGroups)
        for logical_op in physical_plan.op_map.values()
    )


def test_map_groups_rejects_unknown_shuffle_elision_config(source_dataset):
    with pytest.raises(TypeError, match="ParquetCudfShuffleElisionConfig"):
        _map_groups(
            source_dataset,
            "id",
            parquet_cudf_shuffle_elision=object(),
        )


@pytest.mark.parametrize(
    ("strategy", "upstream_type"),
    [
        (ShuffleStrategy.HASH_SHUFFLE, Repartition),
        (ShuffleStrategy.SORT_SHUFFLE_PULL_BASED, Sort),
    ],
)
def test_combine_shuffles_preserves_all_map_groups_fields(strategy, upstream_type):
    source = InputData(input_data=[])
    if upstream_type is Repartition:
        upstream = Repartition(
            num_outputs=9,
            input_dependencies=[source],
            shuffle=True,
        )
    else:
        upstream = Sort(
            sort_key=SortKey("old_key"),
            input_dependencies=[source],
        )

    config = _config()

    def remote_args_fn():
        return {"resources": {"dynamic": 1}}

    op = MapGroups(
        key=["key", "subkey"],
        num_partitions=3 if upstream_type is Repartition else None,
        num_partitions_explicit=True,
        shuffle_strategy=strategy,
        fn=identity,
        batch_format="pandas",
        zero_copy_batch=False,
        fn_args=(1,),
        fn_kwargs={"value": 2},
        compute=TaskPoolStrategy(size=2),
        ray_remote_args_fn=remote_args_fn,
        ray_remote_args={"num_cpus": 0.5},
        parquet_cudf_shuffle_elision=config,
        input_dependencies=[upstream],
    )

    combined = CombineShuffles._combine(op)

    assert combined is not op
    assert combined.input_dependencies == [source]
    assert combined._get_args() == op._get_args()
    assert combined.parquet_cudf_shuffle_elision is config


@pytest.mark.parametrize(
    ("strategy", "upstream_type", "expected_names"),
    [
        (
            ShuffleStrategy.HASH_SHUFFLE,
            Repartition,
            [
                "Input",
                "HashShuffleMap(keys=('key',), partitions=3)",
                "HashShuffleReduce(keys=('key',), partitions=3)",
                "MapBatches(identity)",
            ],
        ),
        (
            ShuffleStrategy.SORT_SHUFFLE_PULL_BASED,
            Sort,
            ["Input", "Sort", "MapBatches(identity)"],
        ),
    ],
)
def test_combine_shuffles_keeps_one_map_groups_exchange(
    strategy, upstream_type, expected_names
):
    context = DataContext()
    context.shuffle_strategy = strategy
    source = InputData(input_data=[])
    if upstream_type is Repartition:
        upstream = Repartition(
            num_outputs=9,
            input_dependencies=[source],
            shuffle=True,
        )
    else:
        upstream = Sort(
            sort_key=SortKey("old_key"),
            input_dependencies=[source],
        )
    op = MapGroups(
        key="key",
        num_partitions=3 if upstream_type is Repartition else None,
        num_partitions_explicit=True,
        shuffle_strategy=strategy,
        fn=identity,
        compute=TaskPoolStrategy(size=2),
        input_dependencies=[upstream],
    )

    optimized = CombineShuffles().apply(LogicalPlan(op, context))
    assert optimized.dag.input_dependencies == [source]

    physical_plan, _ = create_planner().plan(optimized)
    physical_names = [
        physical_op.name for physical_op in physical_plan.dag.post_order_iter()
    ]
    assert physical_names == expected_names
