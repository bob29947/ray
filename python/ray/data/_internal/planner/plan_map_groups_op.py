from functools import partial
from typing import List

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.operators import MapBatches, MapGroups, Repartition, Sort
from ray.data._internal.planner.exchange.sort_task_spec import SortKey
from ray.data._internal.planner.plan_all_to_all_op import plan_all_to_all_op
from ray.data._internal.planner.plan_udf_map_op import plan_udf_map_op
from ray.data.block import CallableClass
from ray.data.context import DataContext, ShuffleStrategy


def _build_map_groups_udf(op: MapGroups):
    """Wrap the group UDF with the existing per-block group iteration behavior."""
    # Extract standalone values so the worker closure doesn't capture the logical op
    # (or, transitively, its entire input plan).
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

    # Match the historical wrapped MapBatches name shown in progress and explain
    # output, including functools.partial handling.
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
    """Lower ``MapGroups`` to the existing exchange plus whole-block map DAG."""
    assert len(physical_children) == 1
    input_physical_dag = physical_children[0]

    if op.key is None:
        exchange_op = Repartition(
            num_outputs=1,
            input_dependencies=op.input_dependencies,
        )
    elif op.shuffle_strategy in (
        ShuffleStrategy.HASH_SHUFFLE,
        ShuffleStrategy.GPU_SHUFFLE,
    ):
        exchange_op = Repartition(
            num_outputs=op.num_partitions,
            input_dependencies=op.input_dependencies,
            keys=op.key,
            sort=True,
        )
    else:
        exchange_op = Sort(
            sort_key=SortKey(op.key),
            input_dependencies=op.input_dependencies,
        )

    exchange_physical_dag = plan_all_to_all_op(
        exchange_op, [input_physical_dag], data_context
    )

    map_batches_op = MapBatches(
        _build_map_groups_udf(op),
        input_dependencies=[exchange_op],
        batch_size=None,
        can_modify_num_rows=True,
        # The wrapper receives each underlying block and performs the only required
        # conversion once per group.
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
    return plan_udf_map_op(
        map_batches_op, [exchange_physical_dag], data_context
    )
