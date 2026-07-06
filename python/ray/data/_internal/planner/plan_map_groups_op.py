from functools import partial
from typing import List

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.operators import (
    MapBatches,
    MapGroups,
    Repartition,
    Sort,
)
from ray.data._internal.planner.exchange.sort_task_spec import SortKey
from ray.data._internal.planner.plan_all_to_all_op import plan_all_to_all_op
from ray.data._internal.planner.plan_udf_map_op import plan_udf_map_op
from ray.data.block import CallableClass
from ray.data.context import DataContext, ShuffleStrategy


def _build_map_groups_udf(op: MapGroups):
    """Wrap the group UDF with the existing per-block group iteration behavior."""
    fn = op.fn
    if op.key is None:
        keys = []
    elif isinstance(op.key, str):
        keys = [op.key]
    else:
        keys = op.key
    batch_format = op.batch_format

    # Keep this import lazy to avoid a module cycle through Dataset/GroupedData.
    from ray.data.grouped_data import _apply_udf_to_groups

    if isinstance(fn, CallableClass):

        class wrapped_fn:
            def __init__(self, *args, **kwargs):
                self.fn = fn(*args, **kwargs)

            def __call__(self, batch, *args, **kwargs):
                yield from _apply_udf_to_groups(
                    self.fn, batch, keys, batch_format, *args, **kwargs
                )

    else:

        def wrapped_fn(batch, *args, **kwargs):
            yield from _apply_udf_to_groups(
                fn, batch, keys, batch_format, *args, **kwargs
            )

    if isinstance(fn, partial):
        wrapped_fn.__name__ = fn.func.__name__
    else:
        wrapped_fn.__name__ = fn.__name__
    return wrapped_fn


def plan_map_groups_op(
    op: MapGroups,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    """Plan ``MapGroups``, selecting shuffle elision or the stock lowering."""
    assert len(physical_children) == 1

    if op.parquet_cudf_shuffle_elision is not None:
        from ray.data._internal.planner.parquet_cudf_shuffle_elision import (
            try_plan_parquet_cudf_shuffle_elision,
        )

        optimized = try_plan_parquet_cudf_shuffle_elision(op, data_context)
        if optimized is not None:
            return optimized

    physical_child = physical_children[0]
    shuffle_op = _create_shuffle_op(op)

    physical_shuffle = plan_all_to_all_op(shuffle_op, [physical_child], data_context)
    _set_generated_logical_operators(
        physical_shuffle,
        stop_at=physical_child,
        logical_op=shuffle_op,
    )

    map_batches_op = MapBatches(
        _build_map_groups_udf(op),
        input_dependencies=[shuffle_op],
        batch_size=None,
        can_modify_num_rows=True,
        batch_format=None,
        zero_copy_batch=op.zero_copy_batch,
        min_rows_per_bundled_input=None,
        fn_args=op.fn_args,
        fn_kwargs=op.fn_kwargs,
        fn_constructor_args=op.fn_constructor_args,
        fn_constructor_kwargs=op.fn_constructor_kwargs,
        compute=op.compute,
        ray_remote_args_fn=op.ray_remote_args_fn,
        ray_remote_args=op.ray_remote_args,
    )
    physical_map = plan_udf_map_op(map_batches_op, [physical_shuffle], data_context)
    physical_map.set_logical_operators(map_batches_op)
    return physical_map


def _create_shuffle_op(op: MapGroups):
    input_logical_op = op.input_dependencies[0]
    if op.key is None:
        return Repartition(
            num_outputs=1,
            input_dependencies=[input_logical_op],
        )
    if op.shuffle_strategy in (
        ShuffleStrategy.HASH_SHUFFLE,
        ShuffleStrategy.GPU_SHUFFLE,
    ):
        keys = [op.key] if isinstance(op.key, str) else op.key
        return Repartition(
            num_outputs=op.num_partitions,
            input_dependencies=[input_logical_op],
            keys=keys,
            sort=True,
        )
    return Sort(
        sort_key=SortKey(op.key),
        input_dependencies=[input_logical_op],
    )


def _set_generated_logical_operators(
    root: PhysicalOperator,
    *,
    stop_at: PhysicalOperator,
    logical_op,
) -> None:
    """Label physical operators created while lowering one logical operator."""
    stack = [root]
    seen = set()
    while stack:
        current = stack.pop()
        if current is stop_at or current in seen:
            continue
        seen.add(current)
        current.set_logical_operators(logical_op)
        stack.extend(current.input_dependencies)
