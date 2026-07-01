from dataclasses import replace
from functools import partial

import pyarrow.fs as pafs
import pytest

from ray.data._internal.compute import ActorPoolStrategy, TaskPoolStrategy
from ray.data._internal.datasource.parquet_datasource import ParquetDatasource
from ray.data._internal.logical.operators import InputData, MapBatches, MapGroups, Read
from ray.data._internal.planner.map_groups_partition_protocol import (
    MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG,
    MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE,
    MAP_GROUPS_PARTITION_UDF_ATTRIBUTE,
)
from ray.data._internal.planner.parquet_range_map_groups_selector import (
    PARQUET_RANGE_MAP_GROUPS_ENABLED_CONFIG,
    select_parquet_range_map_groups_candidate,
)
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.datasource.datasource import Datasource


class _Tokenizer:
    __ray_data_preserves_partitioning__ = ("User",)

    def __call__(self, batch):
        return batch


def _group_fn(batch):
    return batch


def _datasource(**overrides):
    datasource = ParquetDatasource.__new__(ParquetDatasource)
    Datasource.__init__(datasource)
    datasource._supports_distributed_reads = False
    datasource._filesystem = pafs.LocalFileSystem()
    datasource._pq_paths = ["/tmp/input.parquet"]
    datasource._projection_map = {
        "User": "User",
        "Card": "Card",
        "payload": "payload",
    }
    datasource._partition_columns = []
    datasource._block_udf = None
    datasource._shuffle = None
    datasource._read_schema = None
    datasource._include_paths = False
    datasource._include_row_hash = False
    datasource._scanner_kwargs = {}
    for name, value in overrides.items():
        setattr(datasource, name, value)
    return datasource


def _candidate_plan(
    *,
    datasource=None,
    tokenizer_fn=_Tokenizer,
    group_fn=_group_fn,
    tokenizer_compute=None,
    group_compute=None,
    tokenizer_remote_args=None,
    group_remote_args=None,
    read_remote_args=None,
    **map_groups_kwargs,
):
    datasource = datasource or _datasource()
    resources = {"num_cpus": 1, "num_gpus": 1}
    read_op = Read(
        datasource=datasource,
        datasource_or_legacy_reader=datasource,
        parallelism=2,
        ray_remote_args=read_remote_args
        or {"label_selector": {"ray.io/node-id": "node-1"}},
    )
    tokenizer_op = MapBatches(
        tokenizer_fn,
        input_dependencies=[read_op],
        batch_size=128,
        batch_format="cudf",
        can_modify_num_rows=False,
        compute=tokenizer_compute or ActorPoolStrategy(size=2),
        ray_remote_args=tokenizer_remote_args or resources,
    )
    defaults = {
        "key": ["User", "Card"],
        "fn": group_fn,
        "num_partitions": 2,
        "shuffle_strategy": ShuffleStrategy.HASH_SHUFFLE,
        "num_partitions_explicit": True,
        "input_dependencies": [tokenizer_op],
        "batch_format": "cudf",
        "compute": group_compute or TaskPoolStrategy(size=2),
        "ray_remote_args": group_remote_args or resources,
    }
    defaults.update(map_groups_kwargs)
    map_groups_op = MapGroups(**defaults)
    context = DataContext()
    context.set_config(PARQUET_RANGE_MAP_GROUPS_ENABLED_CONFIG, True)
    return map_groups_op, context


def _select(op, context):
    return select_parquet_range_map_groups_candidate(op, context)


def _replace_tokenizer(op, **changes):
    tokenizer_op = replace(op.input_dependencies[0], **changes)
    return replace(op, input_dependencies=[tokenizer_op])


def test_selects_exact_natural_gpu_pipeline_without_io():
    op, context = _candidate_plan()

    result = _select(op, context)

    assert result.selected
    assert result.fallback_reason is None
    candidate = result.candidate
    assert candidate.read_op is op.input_dependencies[0].input_dependencies[0]
    assert candidate.datasource is candidate.read_op.datasource
    assert candidate.tokenizer_op is op.input_dependencies[0]
    assert candidate.map_groups_op is op
    assert candidate.partition_key == "User"
    assert candidate.group_keys == ("User", "Card")
    assert candidate.num_partitions == 2
    assert candidate.projection == ("User", "Card", "payload")
    assert candidate.compute is candidate.tokenizer_op.compute
    assert candidate.ray_remote_args == {"num_cpus": 1, "num_gpus": 1}
    assert candidate.partition_contract is None


def test_selects_explicit_partition_equivalent_group_udf():
    def partition_group(batch, context):
        return batch

    def group(batch):
        return batch

    setattr(
        group,
        MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE,
        {
            "version": 1,
            "batch_format": "cudf",
            "side_effect_free": True,
            "equivalent_to_per_group": True,
        },
    )
    setattr(group, MAP_GROUPS_PARTITION_UDF_ATTRIBUTE, partition_group)
    op, context = _candidate_plan(group_fn=group)
    context.set_config(MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG, True)

    result = _select(op, context)

    assert result.selected
    assert result.candidate.partition_contract.udf is partition_group


def test_invalid_optional_partition_protocol_keeps_per_group_range_backend():
    def group(batch):
        return batch

    setattr(
        group,
        MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE,
        {"version": 99, "batch_format": "cudf", "side_effect_free": True},
    )
    setattr(group, MAP_GROUPS_PARTITION_UDF_ATTRIBUTE, lambda batch: batch)
    op, context = _candidate_plan(group_fn=group)
    context.set_config(MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG, True)

    result = _select(op, context)

    assert result.selected
    assert result.candidate.partition_contract is None
    assert (
        result.candidate.partition_contract_fallback_reason
        == "group_partition_protocol_invalid"
    )


def test_selection_requires_explicit_opt_in():
    op, context = _candidate_plan()
    context.remove_config(PARQUET_RANGE_MAP_GROUPS_ENABLED_CONFIG)

    result = _select(op, context)

    assert not result.selected
    assert result.candidate is None
    assert result.fallback_reason == "parquet_range_map_groups_disabled"


def test_selection_rejects_checkpointing():
    op, context = _candidate_plan()
    context._checkpoint_config = object()

    assert _select(op, context).fallback_reason == "checkpointing_unsupported"


@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("retried_map_errors", True, "map_error_retries_unsupported"),
        (
            "actor_task_retry_on_errors",
            [ValueError],
            "actor_task_retries_unsupported",
        ),
        ("actor_init_retry_on_errors", True, "actor_init_retries_unsupported"),
        ("max_errored_blocks", 1, "errored_blocks_unsupported"),
        ("max_errored_blocks", -1, "errored_blocks_unsupported"),
    ],
)
def test_selection_rejects_context_level_replay_and_error_suppression(
    attribute, value, reason
):
    op, context = _candidate_plan()
    setattr(context, attribute, value)

    assert _select(op, context).fallback_reason == reason


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (None, "missing_partitioning_contract"),
        ([], "invalid_partitioning_contract"),
        ((), "invalid_partitioning_contract"),
        (("User", "User"), "invalid_partitioning_contract"),
        (("User", 1), "invalid_partitioning_contract"),
        (("",), "invalid_partitioning_contract"),
        (("User", "Card"), "partitioning_contract_not_single_column"),
    ],
)
def test_partition_preservation_protocol_is_strict(value, reason):
    if value is None:

        class Tokenizer:
            def __call__(self, batch):
                return batch

    else:

        class Tokenizer:
            __ray_data_preserves_partitioning__ = value

            def __call__(self, batch):
                return batch

    op, context = _candidate_plan(tokenizer_fn=Tokenizer)

    assert _select(op, context).fallback_reason == reason


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"key": "User"}, "group_keys_not_composite"),
        ({"key": ["User", "User"]}, "group_keys_not_composite"),
        ({"num_partitions_explicit": False}, "num_partitions_not_explicit"),
        ({"num_partitions": 0}, "num_partitions_invalid"),
        (
            {"shuffle_strategy": ShuffleStrategy.SORT_SHUFFLE_PULL_BASED},
            "shuffle_strategy_unsupported",
        ),
        ({"batch_format": "pandas"}, "group_batch_format_not_cudf"),
        (
            {"compute": TaskPoolStrategy()},
            "group_task_pool_not_fixed",
        ),
        (
            {"compute": TaskPoolStrategy(size=1)},
            "group_task_pool_size_mismatch",
        ),
        ({"ray_remote_args": {"num_gpus": 0}}, "group_gpu_resource_not_one"),
    ],
)
def test_map_groups_gates_have_stable_reasons(changes, reason):
    op, context = _candidate_plan(**changes)

    assert _select(op, context).fallback_reason == reason


def test_map_groups_requires_sync_plain_function():
    class CallableGroup:
        def __call__(self, batch):
            return batch

    op, context = _candidate_plan(group_fn=CallableGroup)
    assert (
        _select(op, context).fallback_reason == "group_udf_callable_class_unsupported"
    )

    async def async_group(batch):
        return batch

    op, context = _candidate_plan(group_fn=async_group)
    assert _select(op, context).fallback_reason == "group_udf_async_unsupported"

    op, context = _candidate_plan(group_fn=partial(_group_fn))
    assert _select(op, context).fallback_reason == "group_udf_not_plain_function"


def test_map_groups_allows_sync_generator_function():
    def generator_group(batch):
        yield batch

    op, context = _candidate_plan(group_fn=generator_group)

    assert _select(op, context).selected


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"fn": lambda batch: batch}, "tokenizer_not_callable_class"),
        ({"batch_format": "pandas"}, "tokenizer_batch_format_not_cudf"),
        ({"batch_size": None}, "tokenizer_batch_size_not_fixed"),
        ({"batch_size": 0}, "tokenizer_batch_size_not_fixed"),
        ({"can_modify_num_rows": True}, "tokenizer_may_modify_row_count"),
        ({"per_block_limit": 1}, "tokenizer_per_block_limit_unsupported"),
        (
            {"compute": ActorPoolStrategy(min_size=1, max_size=2)},
            "tokenizer_actor_pool_not_fixed",
        ),
        (
            {"compute": ActorPoolStrategy(size=1)},
            "tokenizer_actor_pool_size_mismatch",
        ),
        (
            {
                "compute": ActorPoolStrategy(
                    size=2,
                    enable_true_multi_threading=True,
                )
            },
            "tokenizer_actor_concurrency_unsupported",
        ),
        (
            {
                "compute": ActorPoolStrategy(
                    size=2,
                    max_tasks_in_flight_per_actor=2,
                )
            },
            "tokenizer_actor_concurrency_unsupported",
        ),
        ({"ray_remote_args": {"num_gpus": 0}}, "tokenizer_gpu_resource_not_one"),
    ],
)
def test_tokenizer_gates_have_stable_reasons(changes, reason):
    op, context = _candidate_plan()
    op = _replace_tokenizer(op, **changes)

    assert _select(op, context).fallback_reason == reason


def test_tokenizer_rejects_async_callable_class_and_dynamic_resources():
    class AsyncTokenizer:
        __ray_data_preserves_partitioning__ = ("User",)

        async def __call__(self, batch):
            return batch

    op, context = _candidate_plan(tokenizer_fn=AsyncTokenizer)
    assert _select(op, context).fallback_reason == "tokenizer_async_unsupported"

    op, context = _candidate_plan()
    op = _replace_tokenizer(op, ray_remote_args_fn=lambda: {"num_gpus": 1})
    assert (
        _select(op, context).fallback_reason
        == "tokenizer_dynamic_remote_args_unsupported"
    )


@pytest.mark.parametrize(
    "option",
    ["max_restarts", "max_task_retries", "max_retries", "retry_exceptions"],
)
def test_replay_options_are_rejected_for_both_udfs(option):
    resources = {"num_cpus": 1, "num_gpus": 1, option: 0}
    op, context = _candidate_plan(tokenizer_remote_args=resources)
    assert _select(op, context).fallback_reason == "tokenizer_retry_options_unsupported"

    op, context = _candidate_plan(group_remote_args=resources)
    assert _select(op, context).fallback_reason == "group_retry_options_unsupported"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"_filesystem": object()}, "parquet_filesystem_not_local"),
        ({"_pq_paths": []}, "parquet_source_paths_invalid"),
        ({"_pq_paths": ["relative.parquet"]}, "parquet_source_paths_invalid"),
        ({"_projection_map": None}, "parquet_projection_missing"),
        (
            {"_projection_map": {"User": "renamed", "Card": "Card"}},
            "parquet_projection_invalid",
        ),
        ({"_partition_columns": ["country"]}, "parquet_partition_columns_unsupported"),
        ({"_predicate_expr": object()}, "parquet_predicate_unsupported"),
        ({"_block_udf": lambda block: block}, "parquet_block_udf_unsupported"),
        ({"_shuffle": "files"}, "parquet_file_shuffle_unsupported"),
        ({"_read_schema": object()}, "parquet_schema_override_unsupported"),
        ({"_include_paths": True}, "parquet_synthetic_columns_unsupported"),
        (
            {"_scanner_kwargs": {"use_threads": False}},
            "parquet_scan_options_unsupported",
        ),
    ],
)
def test_v1_parquet_gates_have_stable_reasons(overrides, reason):
    op, context = _candidate_plan(datasource=_datasource(**overrides))

    assert _select(op, context).fallback_reason == reason


def test_shared_posix_source_and_read_scheduling_strategy_are_supported():
    op, context = _candidate_plan(
        datasource=_datasource(_supports_distributed_reads=True),
        read_remote_args={"scheduling_strategy": "SPREAD"},
    )

    assert _select(op, context).selected


def test_projection_must_contain_partition_and_all_group_keys():
    op, context = _candidate_plan(
        datasource=_datasource(_projection_map={"Card": "Card"})
    )
    assert _select(op, context).fallback_reason == "partition_key_not_projected"

    op, context = _candidate_plan(
        datasource=_datasource(_projection_map={"User": "User", "payload": "payload"})
    )
    assert _select(op, context).fallback_reason == "group_key_not_projected"


def test_partition_key_must_be_the_leading_group_key():
    class Tokenizer:
        __ray_data_preserves_partitioning__ = ("Card",)

        def __call__(self, batch):
            return batch

    op, context = _candidate_plan(tokenizer_fn=Tokenizer)

    assert _select(op, context).fallback_reason == "partition_key_not_leading_group_key"


def test_resources_and_label_selectors_must_be_compatible():
    op, context = _candidate_plan(group_remote_args={"num_cpus": 2, "num_gpus": 1})
    assert (
        _select(op, context).fallback_reason == "tokenizer_group_resources_incompatible"
    )

    resources = {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {"ray.io/node-id": "node-2"},
    }
    op, context = _candidate_plan(
        tokenizer_remote_args=resources,
        group_remote_args=resources,
    )
    assert (
        _select(op, context).fallback_reason
        == "actor_label_selector_conflicts_with_read"
    )

    resources["label_selector"] = {
        "ray.io/node-id": "node-1",
        "accelerator": "v100",
    }
    op, context = _candidate_plan(
        tokenizer_remote_args=resources,
        group_remote_args=resources,
    )
    assert _select(op, context).selected

    tokenizer_resources = {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {"accelerator": "v100"},
    }
    group_resources = {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {"accelerator": "a100"},
    }
    op, context = _candidate_plan(
        tokenizer_remote_args=tokenizer_resources,
        group_remote_args=group_resources,
    )
    assert (
        _select(op, context).fallback_reason
        == "tokenizer_group_label_selectors_conflict"
    )


def test_compatible_actor_label_selectors_are_merged():
    tokenizer_resources = {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {"accelerator": "v100"},
    }
    group_resources = {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {"pool": "batch"},
    }
    op, context = _candidate_plan(
        tokenizer_remote_args=tokenizer_resources,
        group_remote_args=group_resources,
    )

    result = _select(op, context)

    assert result.selected
    assert result.candidate.ray_remote_args == {
        "num_cpus": 1,
        "num_gpus": 1,
        "label_selector": {"accelerator": "v100", "pool": "batch"},
    }


def test_structure_must_be_exact_read_then_tokenizer_then_map_groups():
    op, context = _candidate_plan()
    assert (
        _select(op.input_dependencies[0], context).fallback_reason
        == "root_not_map_groups"
    )

    assert (
        _select(
            replace(
                op, input_dependencies=[op.input_dependencies[0].input_dependencies[0]]
            ),
            context,
        ).fallback_reason
        == "map_groups_input_not_tokenizer"
    )

    op = _replace_tokenizer(op, input_dependencies=[InputData([])])
    assert _select(op, context).fallback_reason == "tokenizer_input_not_read"


def test_read_only_allows_driver_label_selector():
    op, context = _candidate_plan(read_remote_args={"num_cpus": 1})
    assert _select(op, context).fallback_reason == "read_remote_args_unsupported"

    op, context = _candidate_plan(
        read_remote_args={"label_selector": {"ray.io/node-id": 1}}
    )
    assert _select(op, context).fallback_reason == "read_label_selector_invalid"
