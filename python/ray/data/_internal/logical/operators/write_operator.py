from typing import Any, Dict, Optional, Union

from ray.data._internal.compute import ComputeStrategy
from ray.data._internal.logical.interfaces import LogicalOperator
from ray.data._internal.logical.operators.map_operator import AbstractMap
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource

__all__ = [
    "Write",
]


class Write(AbstractMap):
    """Logical operator for write."""

    def __init__(
        self,
        input_op: LogicalOperator,
        datasink_or_legacy_datasource: Union[Datasink, Datasource],
        ray_remote_args: Optional[Dict[str, Any]] = None,
        compute: Optional[ComputeStrategy] = None,
        **write_args,
    ):
        if isinstance(datasink_or_legacy_datasource, Datasink):
            min_rows_per_bundled_input = (
                datasink_or_legacy_datasource.min_rows_per_write
            )
            target_bytes_per_bundled_input = (
                datasink_or_legacy_datasource.target_bytes_per_write
            )
        else:
            min_rows_per_bundled_input = None
            target_bytes_per_bundled_input = None

        if (
            min_rows_per_bundled_input is not None
            and target_bytes_per_bundled_input is not None
        ):
            raise ValueError(
                "A Datasink can't set both min_rows_per_write and "
                "target_bytes_per_write."
            )
        if target_bytes_per_bundled_input is not None and (
            not isinstance(target_bytes_per_bundled_input, int)
            or isinstance(target_bytes_per_bundled_input, bool)
            or target_bytes_per_bundled_input <= 0
        ):
            raise ValueError(
                "target_bytes_per_write must be a positive integer or None, got "
                f"{target_bytes_per_bundled_input!r}."
            )

        super().__init__(
            input_op=input_op,
            can_modify_num_rows=True,
            min_rows_per_bundled_input=min_rows_per_bundled_input,
            ray_remote_args=ray_remote_args,
            compute=compute,
        )
        self.datasink_or_legacy_datasource = datasink_or_legacy_datasource
        self.target_bytes_per_bundled_input = target_bytes_per_bundled_input
        self.write_args = write_args
