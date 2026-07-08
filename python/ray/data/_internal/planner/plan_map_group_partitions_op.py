from typing import List

from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.logical.operators import MapGroupPartitions
from ray.data.context import DataContext


def plan_map_group_partitions_op(
    op: MapGroupPartitions,
    physical_children: List[PhysicalOperator],
    data_context: DataContext,
) -> PhysicalOperator:
    """Plan an explicit Parquet/cuDF group-partition fast path."""
    assert len(physical_children) == 1

    from ray.data._internal.planner.parquet_cudf_map_group_partitions import (
        plan_parquet_cudf_map_group_partitions,
    )

    return plan_parquet_cudf_map_group_partitions(op, data_context)
