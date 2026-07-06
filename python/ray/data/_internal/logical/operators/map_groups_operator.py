from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.map_operator import MapBatches
from ray.data.block import BlockMetadata, DataBatch, UserDefinedFunction
from ray.data.context import ShuffleStrategy

__all__ = ["MapGroups"]


@dataclass(frozen=True, repr=False, eq=False)
class MapGroups(LogicalOperator):
    """Logical operator for :meth:`GroupedData.map_groups`."""

    key: Optional[Union[str, List[str]]]
    num_partitions: Optional[int]
    num_partitions_explicit: bool
    shuffle_strategy: ShuffleStrategy
    fn: UserDefinedFunction[DataBatch, DataBatch]
    batch_format: Optional[str] = "default"
    zero_copy_batch: bool = True
    fn_args: Optional[Iterable[Any]] = None
    fn_kwargs: Optional[Dict[str, Any]] = None
    fn_constructor_args: Optional[Iterable[Any]] = None
    fn_constructor_kwargs: Optional[Dict[str, Any]] = None
    compute: Optional[ComputeStrategy] = None
    ray_remote_args_fn: Optional[Callable[[], Dict[str, Any]]] = None
    ray_remote_args: Dict[str, Any] = field(default_factory=dict)
    parquet_cudf_shuffle_elision: Optional[Any] = None
    input_dependencies: List[LogicalOperator] = field(repr=False, kw_only=True)
    batch_size: None = field(init=False, default=None)
    can_modify_num_rows: bool = field(init=False, default=True)
    min_rows_per_bundled_input: None = field(init=False, default=None)

    def __post_init__(self):
        assert len(self.input_dependencies) == 1, len(self.input_dependencies)
        map_batches_name = MapBatches(
            self.fn,
            input_dependencies=self.input_dependencies,
        ).name
        if map_batches_name.startswith("MapBatches("):
            name = "MapGroups(" + map_batches_name[len("MapBatches(") :]
        else:
            name = f"MapGroups({map_batches_name})"
        object.__setattr__(self, "_name", name)

    @property
    def num_outputs(self) -> Optional[int]:
        if self.key is None or self.shuffle_strategy in (
            ShuffleStrategy.HASH_SHUFFLE,
            ShuffleStrategy.GPU_SHUFFLE,
        ):
            return self.num_partitions
        return None

    def infer_metadata(self) -> BlockMetadata:
        input_metadata = self.input_dependencies[0].infer_metadata()
        return BlockMetadata(
            num_rows=None,
            size_bytes=None,
            input_files=input_metadata.input_files,
            exec_stats=None,
        )
