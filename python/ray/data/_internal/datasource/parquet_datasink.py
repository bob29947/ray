import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional

from ray._common.retry import call_with_retry
from ray.data._internal.arrow_ops.transform_pyarrow import (
    reorder_columns_by_schema,
)
from ray.data._internal.execution.interfaces import TaskContext
from ray.data._internal.planner.plan_write_op import WRITE_UUID_KWARG_NAME
from ray.data._internal.savemode import SaveMode
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.file_based_datasource import _resolve_kwargs
from ray.data.datasource.file_datasink import _FileDatasink
from ray.data.datasource.filename_provider import FilenameProvider

if TYPE_CHECKING:
    import pyarrow

WRITE_FILE_MAX_ATTEMPTS = 10
WRITE_FILE_RETRY_MAX_BACKOFF_SECONDS = 32
FILE_FORMAT = "parquet"

# These args are part of https://arrow.apache.org/docs/python/generated/pyarrow.fs.FileSystem.html#pyarrow.fs.FileSystem.open_output_stream
# and are not supported by ParquetDatasink.
UNSUPPORTED_OPEN_STREAM_ARGS = {"path", "buffer", "metadata"}

# https://arrow.apache.org/docs/python/generated/pyarrow.dataset.write_dataset.html
ARROW_DEFAULT_MAX_ROWS_PER_GROUP = 1024 * 1024

DEFAULT_PARTITIONING_FLAVOR = "hive"

# Experimental, source-neutral Parquet encoding for dense fixed-shape tensor
# columns.  This remains opt-in while the packed on-disk representation gains
# broader compatibility coverage.
PARQUET_WRITE_PACKED_TENSORS_CONFIG = "parquet_write_packed_tensors"

logger = logging.getLogger(__name__)


def _pack_fixed_shape_tensor_columns(
    tables: List["pyarrow.Table"],
) -> tuple[List["pyarrow.Table"], "pyarrow.Schema"]:
    """Pack compatible tensor columns consistently across all input tables.

    A column is converted only if every chunk in every table is compatible.
    This keeps schema selection fail-closed for null, variable-shape, empty-
    dimension, oversized, or unsupported tensors.
    """

    import pyarrow as pa

    from ray.data._internal.tensor_extensions.arrow import (
        pack_arrow_fixed_shape_tensor_array,
    )

    if not tables:
        raise ValueError("At least one table is required for Parquet output.")

    output_tables = list(tables)
    output_fields = list(tables[0].schema)
    for column_index, field in enumerate(output_fields):
        packed_columns = []
        packed_type = None
        compatible = True
        for table in output_tables:
            packed_chunks = []
            for chunk in table.column(column_index).chunks:
                packed = pack_arrow_fixed_shape_tensor_array(chunk)
                if packed is None:
                    compatible = False
                    break
                if packed_type is None:
                    packed_type = packed.type
                elif packed.type != packed_type:
                    compatible = False
                    break
                packed_chunks.append(packed)
            if not compatible or not packed_chunks:
                compatible = False
                break
            packed_columns.append(pa.chunked_array(packed_chunks, type=packed_type))

        if not compatible:
            continue

        packed_field = pa.field(
            field.name,
            packed_type,
            nullable=field.nullable,
            metadata=field.metadata,
        )
        output_fields[column_index] = packed_field
        output_tables = [
            table.set_column(column_index, packed_field, packed_column)
            for table, packed_column in zip(output_tables, packed_columns)
        ]

    output_schema = pa.schema(output_fields, metadata=tables[0].schema.metadata)
    return output_tables, output_schema


def _maybe_enable_byte_stream_split_for_packed_tensors(
    output_schema: "pyarrow.Schema", write_kwargs: Dict[str, Any]
) -> tuple[str, ...]:
    """Use a compression-friendly encoding for eligible packed tensors.

    BYTE_STREAM_SPLIT improves compression throughput for dense, fixed-width
    values.  It is only selected for packed tensor fields whose dictionary
    encoding is explicitly disabled and whose Parquet compression is enabled.
    User-provided encoding choices remain authoritative.

    Returns:
        The column names for which BYTE_STREAM_SPLIT was selected.
    """

    from ray.data._internal.tensor_extensions.arrow import ArrowPackedTensorType

    if "use_byte_stream_split" in write_kwargs or "column_encoding" in write_kwargs:
        return ()

    compression = write_kwargs.get("compression", "snappy")
    use_dictionary = write_kwargs.get("use_dictionary", True)

    def compression_enabled(column_name: str) -> bool:
        configured = compression
        if isinstance(configured, dict):
            configured = configured.get(column_name)
        if configured is None:
            return False
        if isinstance(configured, str):
            return configured.lower() not in {"none", "uncompressed"}
        # Leave unfamiliar per-column/configuration forms to PyArrow rather
        # than guessing how they should be interpreted.
        return False

    def dictionary_enabled(column_name: str) -> bool:
        if isinstance(use_dictionary, bool):
            return use_dictionary
        if isinstance(use_dictionary, (list, tuple, set)):
            return column_name in use_dictionary
        return True

    packed_columns = tuple(
        field.name
        for field in output_schema
        if isinstance(field.type, ArrowPackedTensorType)
        and compression_enabled(field.name)
        and not dictionary_enabled(field.name)
    )
    if packed_columns:
        # A column list avoids changing the encoding of unrelated scalar
        # fields in mixed schemas.
        write_kwargs["use_byte_stream_split"] = list(packed_columns)
        logger.debug(
            "Enabling BYTE_STREAM_SPLIT for packed tensor columns: %s",
            packed_columns,
        )
    return packed_columns


def choose_row_group_limits(
    row_group_size: Optional[int],
    min_rows_per_file: Optional[int],
    max_rows_per_file: Optional[int],
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Configure row-group limits for Pyarrow's ``write_dataset`` API.

    Configures the ``min_rows_per_group``, ``max_rows_per_group``, and
    ``max_rows_per_file`` parameters based on Ray Data's configuration.

    Args:
        row_group_size: The requested row-group size.
        min_rows_per_file: The minimum number of rows per file.
        max_rows_per_file: The maximum number of rows per file.

    Returns:
        A tuple of (min_rows_per_group, max_rows_per_group, max_rows_per_file).
    """

    if (
        row_group_size is None
        and min_rows_per_file is None
        and max_rows_per_file is None
    ):
        return None, None, None

    elif row_group_size is None:
        # No explicit row group size provided. We are defaulting to
        # either the caller's min_rows_per_file or max_rows_per_file limits
        # or Arrow's defaults
        min_rows_per_group, max_rows_per_group, max_rows_per_file = (
            min_rows_per_file,
            max_rows_per_file,
            max_rows_per_file,
        )

        # If min_rows_per_group is provided and max_rows_per_group is not,
        # and min_rows_per_group is greater than Arrow's default max_rows_per_group,
        # we set max_rows_per_group to min_rows_per_group to avoid creating too many row groups.
        if (
            min_rows_per_group is not None
            and max_rows_per_group is None
            and min_rows_per_group > ARROW_DEFAULT_MAX_ROWS_PER_GROUP
        ):
            max_rows_per_group, max_rows_per_file = (
                min_rows_per_group,
                min_rows_per_group,
            )

        return min_rows_per_group, max_rows_per_group, max_rows_per_file

    elif row_group_size is not None and (
        min_rows_per_file is None or max_rows_per_file is None
    ):
        return row_group_size, row_group_size, max_rows_per_file

    else:
        # Clamp the requested `row_group_size` so that it is
        # * no smaller than `min_rows_per_file` (`lower`)
        # * no larger than `max_rows_per_file` (or Arrow's default cap) (`upper`)
        # This keeps each row-group within the per-file limits while staying
        # as close as possible to the requested size.
        clamped_group_size = max(
            min_rows_per_file, min(row_group_size, max_rows_per_file)
        )
        return clamped_group_size, clamped_group_size, max_rows_per_file


class ParquetDatasink(_FileDatasink):
    def __init__(
        self,
        path: str,
        *,
        partition_cols: Optional[List[str]] = None,
        arrow_parquet_args_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        arrow_parquet_args: Optional[Dict[str, Any]] = None,
        min_rows_per_file: Optional[int] = None,
        max_rows_per_file: Optional[int] = None,
        filesystem: Optional["pyarrow.fs.FileSystem"] = None,
        try_create_dir: bool = True,
        open_stream_args: Optional[Dict[str, Any]] = None,
        filename_provider: Optional[FilenameProvider] = None,
        dataset_uuid: Optional[str] = None,
        mode: SaveMode = SaveMode.APPEND,
    ):
        if arrow_parquet_args_fn is None:
            arrow_parquet_args_fn = lambda: {}  # noqa: E731

        if arrow_parquet_args is None:
            arrow_parquet_args = {}

        self.arrow_parquet_args_fn = arrow_parquet_args_fn
        self.arrow_parquet_args = arrow_parquet_args
        self.min_rows_per_file = min_rows_per_file
        self.max_rows_per_file = max_rows_per_file
        self.partition_cols = partition_cols

        if self.min_rows_per_file is not None and self.max_rows_per_file is not None:
            if self.min_rows_per_file > self.max_rows_per_file:
                raise ValueError(
                    "min_rows_per_file must be less than or equal to max_rows_per_file"
                )

        if open_stream_args is not None:
            intersecting_keys = UNSUPPORTED_OPEN_STREAM_ARGS.intersection(
                set(open_stream_args.keys())
            )
            if intersecting_keys:
                logger.warning(
                    "open_stream_args contains unsupported arguments: %s. These arguments "
                    "are not supported by ParquetDatasink. They will be ignored.",
                    intersecting_keys,
                )

            if "compression" in open_stream_args:
                self.arrow_parquet_args["compression"] = open_stream_args["compression"]

        if ("partitioning_flavor" in self.arrow_parquet_args) or (
            self.arrow_parquet_args_fn is not None
            and "partitioning_flavor" in self.arrow_parquet_args_fn()
        ):
            if self.partition_cols is None:
                raise ValueError(
                    "partition_cols must be provided when partitioning_flavor is set."
                )

        super().__init__(
            path,
            filesystem=filesystem,
            try_create_dir=try_create_dir,
            open_stream_args=open_stream_args,
            filename_provider=filename_provider,
            dataset_uuid=dataset_uuid,
            file_format=FILE_FORMAT,
            mode=mode,
        )

    def write(
        self,
        blocks: Iterable[Block],
        ctx: TaskContext,
    ) -> None:
        import pyarrow as pa

        blocks = list(blocks)

        if all(BlockAccessor.for_block(block).num_rows() == 0 for block in blocks):
            return

        blocks = [
            block for block in blocks if BlockAccessor.for_block(block).num_rows() > 0
        ]

        filename = self.filename_provider.get_filename_for_task(
            ctx.kwargs[WRITE_UUID_KWARG_NAME], ctx.task_idx
        )
        write_kwargs = _resolve_kwargs(
            self.arrow_parquet_args_fn, **self.arrow_parquet_args
        )
        user_schema = write_kwargs.pop("schema", None)

        # For partitioning_flavor, if it's not provided, the default is "hive"
        # Otherwise, it follows pyarrow's behavior: None for directory,
        # "hive" for hive, and "filename" for FilenamePartitioning.
        partitioning_flavor = write_kwargs.pop(
            "partitioning_flavor", DEFAULT_PARTITIONING_FLAVOR
        )

        def write_blocks_to_path():
            tables = [BlockAccessor.for_block(block).to_arrow() for block in blocks]
            if user_schema is None:
                output_schema = pa.unify_schemas([table.schema for table in tables])
            else:
                output_schema = user_schema

            self._write_parquet_files(
                tables,
                filename,
                output_schema,
                ctx.kwargs[WRITE_UUID_KWARG_NAME],
                write_kwargs,
                partitioning_flavor,
            )

        logger.debug(f"Writing {filename} file to {self.path}.")

        call_with_retry(
            write_blocks_to_path,
            description=f"write '{filename}' to '{self.path}'",
            match=self._data_context.retried_io_errors,
            max_attempts=WRITE_FILE_MAX_ATTEMPTS,
            max_backoff_s=WRITE_FILE_RETRY_MAX_BACKOFF_SECONDS,
        )

    def _get_basename_template(self, filename: str, write_uuid: str) -> str:
        # Check if write_uuid is present in filename, add if missing
        if write_uuid not in filename and self.mode == SaveMode.APPEND:
            raise ValueError(
                f"Write UUID '{write_uuid}' missing from filename template '{filename}'. This could result in files being overwritten."
                f"Modify your FileNameProvider implementation to include the `write_uuid` into the filename template or change your write mode to SaveMode.OVERWRITE. "
            )
        # Check if filename is already templatized
        if "{i}" in filename:
            # Filename is already templatized, but may need file extension
            if FILE_FORMAT not in filename:
                # Add file extension to templatized filename
                basename_template = f"{filename}.{FILE_FORMAT}"
            else:
                # Already has extension, use as-is
                basename_template = filename
        elif FILE_FORMAT not in filename:
            # No extension and not templatized, add extension and template
            basename_template = f"{filename}-{{i}}.{FILE_FORMAT}"
        else:
            # TODO(@goutamvenkat-anyscale): Add a warning if you pass in a custom
            # filename provider and it isn't templatized.
            # Use pathlib.Path to properly handle filenames with dots
            filename_path = Path(filename)
            stem = filename_path.stem  # filename without extension
            assert "." not in stem, "Filename should not contain a dot"
            suffix = filename_path.suffix  # extension including the dot
            basename_template = f"{stem}-{{i}}{suffix}"
        return basename_template

    def _write_parquet_files(
        self,
        tables: List["pyarrow.Table"],
        filename: str,
        output_schema: "pyarrow.Schema",
        write_uuid: str,
        write_kwargs: Dict[str, Any],
        partitioning_flavor: Optional[str],
    ) -> None:
        import pyarrow.dataset as ds

        # Make every incoming batch conform to the final schema *before* writing.
        # `pa.unify_schemas` above fixed column order from the first block.
        for idx, table in enumerate(tables):
            if output_schema and not table.schema.equals(output_schema):
                table = reorder_columns_by_schema(table, output_schema)
                table = table.cast(output_schema)
            tables[idx] = table

        if (
            self._data_context.get_config(PARQUET_WRITE_PACKED_TENSORS_CONFIG, False)
            is True
        ):
            tables, output_schema = _pack_fixed_shape_tensor_columns(tables)
            _maybe_enable_byte_stream_split_for_packed_tensors(
                output_schema, write_kwargs
            )

        row_group_size = write_kwargs.pop("row_group_size", None)

        # We set this to "overwrite_or_ignore", to avoid the race condition seen in parallel writes when this is set to "error". The driver already handles the save mode check in on_write_start.
        existing_data_behavior = "overwrite_or_ignore"

        (
            min_rows_per_group,
            max_rows_per_group,
            max_rows_per_file,
        ) = choose_row_group_limits(
            row_group_size,
            min_rows_per_file=self.min_rows_per_file,
            max_rows_per_file=self.max_rows_per_file,
        )

        basename_template = self._get_basename_template(filename, write_uuid)

        # Note that the driver already handles the save mode logic, checking if the directory exists and raising an error if it does on SaveMode.ERROR
        ds.write_dataset(
            data=tables,
            base_dir=self.path,
            schema=output_schema,
            basename_template=basename_template,
            filesystem=self.filesystem,
            partitioning=self.partition_cols,
            format=FILE_FORMAT,
            existing_data_behavior=existing_data_behavior,
            partitioning_flavor=partitioning_flavor,
            use_threads=True,
            min_rows_per_group=min_rows_per_group,
            max_rows_per_group=max_rows_per_group,
            max_rows_per_file=max_rows_per_file,
            file_options=ds.ParquetFileFormat().make_write_options(**write_kwargs),
            create_dir=self.try_create_dir,
        )

    @property
    def min_rows_per_write(self) -> Optional[int]:
        return self.min_rows_per_file
