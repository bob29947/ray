"""Bounded, transactional Parquet sink for the streaming sort benchmark.

This is intentionally benchmark-local.  It assumes that input bundles arrive in
global sort order and uses the write task index to preserve that order during
the driver-side commit.  Worker tasks write only transaction-owned staging
files; final files, the manifest, and ``_SUCCESS`` are published by
``on_write_complete``.
"""

from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import json
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from ray.data._internal.execution.interfaces import TaskContext
from ray.data.block import Block, BlockAccessor
from ray.data.datasource.datasink import Datasink, WriteResult


MIB = 1024**2
GIB = 1024**3

DEFAULT_MAX_DECODED_FILE_BYTES = 4 * GIB
DEFAULT_ROW_GROUP_DECODED_BYTES = 256 * MIB
DEFAULT_DATA_PAGE_BYTES = 1 * MIB
DEFAULT_TARGET_BYTES_PER_WRITE = 4 * GIB
DEFAULT_MAX_WRITERS = 16


class StreamingParquetSinkError(RuntimeError):
    """Base class for failures from the benchmark sink."""


class StreamingParquetSinkCapacityError(StreamingParquetSinkError):
    """The output filesystem could not accept more data."""


class StreamingParquetSinkRollbackError(StreamingParquetSinkError):
    """A write failed and transaction-owned data could not be fully removed."""


class _PrivateRollbackRootUnsafe(StreamingParquetSinkRollbackError):
    """An unowned entry may be present in the private rollback directory."""


@dataclass(frozen=True)
class _OwnedFile:
    """Identity of a file created by this transaction at a particular path."""

    path: Path
    device: int
    inode: int
    descriptor: int

    def at(self, path: Path) -> "_OwnedFile":
        return _OwnedFile(
            path=path,
            device=self.device,
            inode=self.inode,
            descriptor=self.descriptor,
        )


def _schema_to_base64(schema: Any) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def _schema_from_base64(value: str) -> Any:
    import pyarrow as pa

    return pa.ipc.read_schema(pa.BufferReader(base64.b64decode(value)))


def _encoded_schemas_equal(left: str, right: str) -> bool:
    """Compare encoded Arrow schemas without relying on serialization identity."""

    if left == right:
        return True
    return _schema_from_base64(left).equals(
        _schema_from_base64(right), check_metadata=True
    )


def _is_owned(path: Path, root: Path, *, allow_root: bool = False) -> bool:
    resolved_path = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return (allow_root and resolved_path == resolved_root) or (
        resolved_path != resolved_root and resolved_root in resolved_path.parents
    )


def _require_owned(path: Path, root: Path, *, allow_root: bool = False) -> None:
    if not _is_owned(path, root, allow_root=allow_root):
        raise StreamingParquetSinkError(
            f"refusing to operate outside transaction root: {path} is not under {root}"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename(source: Path, destination: Path) -> None:
    """Atomically publish ``source`` without replacing an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise StreamingParquetSinkError(
            "libc does not expose renameat2; refusing a racy no-clobber rename"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    if (
        renameat2(
            at_fdcwd,
            os.fsencode(source),
            at_fdcwd,
            os.fsencode(destination),
            rename_noreplace,
        )
        != 0
    ):
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), str(destination))


def _owned_file(path: Path) -> _OwnedFile:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        stat = os.fstat(descriptor)
        return _OwnedFile(
            path=path,
            device=stat.st_dev,
            inode=stat.st_ino,
            descriptor=descriptor,
        )
    except BaseException:
        os.close(descriptor)
        raise


def _release_owned_file(owned: _OwnedFile) -> None:
    try:
        os.close(owned.descriptor)
    except OSError:
        pass


def _verify_owned_destination(owned: _OwnedFile) -> None:
    """Verify that a successful publish moved the transaction's pinned inode."""

    try:
        stat = owned.path.lstat()
    except OSError as error:
        raise StreamingParquetSinkError(
            f"published destination disappeared at {owned.path}: {error}"
        ) from error
    if (stat.st_dev, stat.st_ino) != (owned.device, owned.inode):
        raise StreamingParquetSinkError(
            "published destination does not contain the transaction-owned inode: "
            f"{owned.path}"
        )


def _quarantine_if_owned(owned: _OwnedFile, quarantine_root: Path) -> Optional[str]:
    """Atomically move our pinned inode into a private transaction namespace.

    A separate ``lstat(path)`` followed by ``unlink(path)`` has a replacement
    race. Moving the current directory entry first gives us a private mode-0700
    transaction namespace to inspect. If another file won the source-path race,
    restore it without deleting it. Owned files remain quarantined until the
    whole private tree is removed; there is no identity-check/pathname-unlink
    window. If restoration itself races, preserve the unowned file and report
    its quarantine path rather than deleting it.
    """

    quarantine = quarantine_root / f"{owned.path.name}-{uuid.uuid4().hex}"
    try:
        try:
            _atomic_rename(owned.path, quarantine)
        except FileNotFoundError:
            return None
        except OSError as error:
            return f"{owned.path}: {error}"
        try:
            stat = quarantine.lstat()
        except OSError as error:
            return f"{quarantine}: {error}"
        if (stat.st_dev, stat.st_ino) != (owned.device, owned.inode):
            try:
                _atomic_rename(quarantine, owned.path)
                return None
            except OSError as error:
                return (
                    f"unowned interference preserved at {quarantine}; "
                    f"could not restore {owned.path}: {error}"
                )
        return None
    finally:
        _release_owned_file(owned)


def _syncfs(path: Path) -> None:
    """Drain writes for only the filesystem containing ``path``."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        syncfs = getattr(libc, "syncfs", None)
        if syncfs is None:
            raise StreamingParquetSinkError(
                "libc does not expose syncfs; refusing an unscoped global sync"
            )
        syncfs.argtypes = [ctypes.c_int]
        syncfs.restype = ctypes.c_int
        if syncfs(descriptor) != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), path)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes, rollback_root: Path) -> _OwnedFile:
    """Write and publish metadata through the private rollback namespace.

    The partial is never placed beside the public destination.  If publishing
    fails, an owned partial is left inside the mode-0700 rollback root for
    tree-level cleanup.  If the partial was replaced, signal that the private
    root must be preserved rather than unlinking the replacement by pathname.
    """

    _require_owned(path, path.parent)
    _require_owned(rollback_root, path.parent)
    partial = rollback_root / f"{path.name}.partial"
    partial_ownership: Optional[_OwnedFile] = None
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(partial, flags, 0o600)
        except FileExistsError as error:
            raise _PrivateRollbackRootUnsafe(
                f"unowned metadata collision preserved at {partial}: {error}"
            ) from error
        stat = os.fstat(descriptor)
        partial_ownership = _OwnedFile(
            path=partial,
            device=stat.st_dev,
            inode=stat.st_ino,
            descriptor=descriptor,
        )
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        final_ownership = partial_ownership.at(path)
        _atomic_rename(partial, path)
        _verify_owned_destination(final_ownership)
        partial_ownership = None
        return final_ownership
    except BaseException:
        if partial_ownership is not None:
            try:
                stat = partial.lstat()
            except FileNotFoundError:
                pass
            except OSError as identity_error:
                _release_owned_file(partial_ownership)
                raise _PrivateRollbackRootUnsafe(
                    f"could not verify private partial {partial}: {identity_error}"
                ) from identity_error
            else:
                if (stat.st_dev, stat.st_ino) != (
                    partial_ownership.device,
                    partial_ownership.inode,
                ):
                    _release_owned_file(partial_ownership)
                    raise _PrivateRollbackRootUnsafe(
                        f"unowned metadata replacement preserved at {partial}"
                    )
            _release_owned_file(partial_ownership)
        raise


def _open_parquet_writer(path: Path, schema: Any, options: dict[str, Any]) -> Any:
    import pyarrow.parquet as pq

    return pq.ParquetWriter(str(path), schema, **options)


def _normalize_failure(
    error: BaseException, *, phase: str, path: Optional[Path] = None
) -> BaseException:
    if isinstance(error, StreamingParquetSinkError):
        return error
    if isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT}:
        location = f" at {path}" if path is not None else ""
        return StreamingParquetSinkCapacityError(
            f"{phase} exhausted local output capacity{location}: {error}"
        )
    if isinstance(error, Exception):
        location = f" at {path}" if path is not None else ""
        return StreamingParquetSinkError(f"{phase} failed{location}: {error}")
    return error


def _remove_owned_tree(path: Path, root: Path) -> list[str]:
    """Remove an exact transaction-owned tree and report persistent errors."""

    _require_owned(path, root, allow_root=True)
    if not path.exists():
        return []
    try:
        shutil.rmtree(path)
    except OSError as error:
        return [f"{path}: {error}"]
    return []


def _largest_prefix_within(table: Any, byte_limit: int) -> int:
    """Return the largest nonempty prefix whose logical Arrow size fits."""

    if table.num_rows <= 0:
        return 0
    if table.nbytes <= byte_limit:
        return table.num_rows
    low = 1
    high = table.num_rows
    if table.slice(0, 1).nbytes > byte_limit:
        return 0
    while low < high:
        middle = (low + high + 1) // 2
        if table.slice(0, middle).nbytes <= byte_limit:
            low = middle
        else:
            high = middle - 1
    return low


def _iter_slices_with_limit(table: Any, byte_limit: int) -> Iterator[Any]:
    if byte_limit <= 0:
        raise ValueError("byte_limit must be positive")
    offset = 0
    while offset < table.num_rows:
        remainder = table.slice(offset)
        rows = _largest_prefix_within(remainder, byte_limit)
        if rows <= 0:
            one_row_bytes = remainder.slice(0, 1).nbytes
            raise StreamingParquetSinkError(
                "one Arrow row exceeds the configured decoded-byte bound: "
                f"row_bytes={one_row_bytes}, bound={byte_limit}"
            )
        yield remainder.slice(0, rows)
        offset += rows


class StreamingParquetDatasink(Datasink[dict[str, Any]]):
    """Write ordered blocks to bounded local Parquet files transactionally.

    ``write_datasink(..., concurrency=16)`` and the hard 4 GiB byte bundling
    bound limit scheduled writer input to 64 GiB in aggregate. Oversized
    upstream blocks are row-sliced by Ray's streaming pre-write stage; this sink
    independently rejects any task whose decoded input crosses the hard bound.

    The output and transaction directories are trial-owned mode-0700 ownership
    boundaries. Public-path identity races are detected and unowned replacements
    are preserved. As with any same-user filesystem transaction built on POSIX
    pathnames, a separate process running as the benchmark UID must not mutate
    those private directories while the transaction is active.
    """

    def __init__(
        self,
        output_directory: str | Path,
        *,
        max_decoded_file_bytes: int = DEFAULT_MAX_DECODED_FILE_BYTES,
        row_group_decoded_bytes: int = DEFAULT_ROW_GROUP_DECODED_BYTES,
        data_page_bytes: int = DEFAULT_DATA_PAGE_BYTES,
        target_bytes_per_write: int = DEFAULT_TARGET_BYTES_PER_WRITE,
        recommended_max_writers: int = DEFAULT_MAX_WRITERS,
    ) -> None:
        self.output_directory = Path(output_directory).resolve(strict=False)
        self.max_decoded_file_bytes = int(max_decoded_file_bytes)
        self.row_group_decoded_bytes = int(row_group_decoded_bytes)
        self.data_page_bytes = int(data_page_bytes)
        self._target_bytes_per_write = int(target_bytes_per_write)
        self.recommended_max_writers = int(recommended_max_writers)
        for name, value in (
            ("max_decoded_file_bytes", self.max_decoded_file_bytes),
            ("row_group_decoded_bytes", self.row_group_decoded_bytes),
            ("data_page_bytes", self.data_page_bytes),
            ("target_bytes_per_write", self._target_bytes_per_write),
            ("recommended_max_writers", self.recommended_max_writers),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.row_group_decoded_bytes > self.max_decoded_file_bytes:
            raise ValueError(
                "row_group_decoded_bytes cannot exceed max_decoded_file_bytes"
            )

        self._transaction_id = uuid.uuid4().hex
        self._transaction_root = (
            self.output_directory / ".staging" / self._transaction_id
        )
        self._rollback_root = (
            self.output_directory / f".rollback-{self._transaction_id}"
        )
        self._rollback_root_identity: Optional[tuple[int, int]] = None
        self._rollback_root_unsafe_reason: Optional[str] = None
        self._started = False
        self._committed = False
        self._schema_base64: Optional[str] = None
        self._owned_final_paths: list[_OwnedFile] = []
        self._result: Optional[dict[str, Any]] = None
        self._telemetry: dict[str, Any] = {
            "transaction_committed": False,
            "rollback_errors": [],
        }
        self._write_started_at_ns: Optional[int] = None

    @property
    def target_bytes_per_write(self) -> int:
        """Target input bytes for each write task (generic Datasink API)."""

        return self._target_bytes_per_write

    @property
    def telemetry(self) -> dict[str, Any]:
        return dict(self._telemetry)

    @property
    def result(self) -> Optional[dict[str, Any]]:
        """Driver-side compact commit result, populated after a durable commit."""

        return None if self._result is None else dict(self._result)

    def on_write_start(self, schema: Optional[Any] = None) -> None:
        if self._started:
            if schema is not None and (
                self._schema_base64 is None
                or not _encoded_schemas_equal(
                    self._schema_base64, _schema_to_base64(schema)
                )
            ):
                raise StreamingParquetSinkError("write schema changed after sink start")
            return

        output_created = False
        staging_created = False
        transaction_created = False
        staging_root = self.output_directory / ".staging"
        try:
            if self.output_directory.exists():
                entries = list(self.output_directory.iterdir())
                if entries:
                    raise StreamingParquetSinkError(
                        "output directory must be empty and trial-owned: "
                        f"{self.output_directory} contains {len(entries)} entries"
                    )
            else:
                self.output_directory.mkdir(parents=True, mode=0o700)
                output_created = True
            # The benchmark supplies a trial-owned output directory. Restrict it
            # and both transaction namespaces so their contents form the cleanup
            # ownership boundary against other UIDs.
            self.output_directory.chmod(0o700)
            staging_root.mkdir(mode=0o700)
            staging_created = True
            staging_root.chmod(0o700)
            self._transaction_root.mkdir(mode=0o700)
            transaction_created = True
            self._create_rollback_root()
            _fsync_directory(self.output_directory)
            if schema is not None:
                self._schema_base64 = _schema_to_base64(schema)
        except BaseException as error:
            cleanup_errors: list[str] = []
            if self._rollback_root_identity is not None or (
                self._rollback_root.exists()
                and self._rollback_root_unsafe_reason is None
            ):
                cleanup_errors.extend(
                    _remove_owned_tree(self._rollback_root, self._rollback_root)
                )
                if not cleanup_errors:
                    self._rollback_root_identity = None
            elif self._rollback_root_unsafe_reason is not None:
                cleanup_errors.append(self._rollback_root_unsafe_reason)
            if transaction_created:
                cleanup_errors.extend(
                    _remove_owned_tree(self._transaction_root, self._transaction_root)
                )
            try:
                if (
                    staging_created
                    and staging_root.exists()
                    and not any(staging_root.iterdir())
                ):
                    staging_root.rmdir()
                if (
                    output_created
                    and self.output_directory.exists()
                    and not any(self.output_directory.iterdir())
                ):
                    self.output_directory.rmdir()
            except OSError as cleanup_error:
                cleanup_errors.append(str(cleanup_error))
            self._telemetry = {
                "transaction_committed": False,
                "rollback_errors": cleanup_errors,
            }
            if cleanup_errors:
                raise StreamingParquetSinkRollbackError(
                    "sink setup failure left transaction-owned files: "
                    + "; ".join(cleanup_errors)
                ) from error
            normalized = _normalize_failure(
                error, phase="Parquet sink setup", path=self.output_directory
            )
            if normalized is error:
                raise
            raise normalized from error
        self._started = True
        self._write_started_at_ns = time.time_ns()

    def _create_rollback_root(self) -> None:
        """Create and remember an exclusive private cleanup namespace."""

        try:
            self._rollback_root.mkdir(mode=0o700)
        except FileExistsError as error:
            self._rollback_root_unsafe_reason = (
                f"rollback-root collision preserved at {self._rollback_root}"
            )
            raise _PrivateRollbackRootUnsafe(
                self._rollback_root_unsafe_reason
            ) from error
        # Do not let a permissive process umask weaken the ownership boundary.
        self._rollback_root.chmod(0o700)
        stat = self._rollback_root.lstat()
        self._rollback_root_identity = (stat.st_dev, stat.st_ino)

    def _rollback_root_is_owned(self) -> bool:
        if self._rollback_root_identity is None:
            return False
        try:
            stat = self._rollback_root.lstat()
        except OSError:
            return False
        return (stat.st_dev, stat.st_ino) == self._rollback_root_identity

    def _remove_empty_rollback_root_before_sync(self) -> None:
        """Remove the empty private root without recursively deleting entries."""

        if not self._rollback_root_is_owned():
            self._rollback_root_unsafe_reason = (
                f"rollback root identity changed at {self._rollback_root}"
            )
            raise _PrivateRollbackRootUnsafe(self._rollback_root_unsafe_reason)
        try:
            self._rollback_root.rmdir()
        except OSError as error:
            self._rollback_root_unsafe_reason = (
                f"unexpected rollback-root contents preserved at "
                f"{self._rollback_root}: {error}"
            )
            raise _PrivateRollbackRootUnsafe(
                self._rollback_root_unsafe_reason
            ) from error
        self._rollback_root_identity = None

    def _writer_options(self) -> dict[str, Any]:
        return {
            "version": "2.6",
            "compression": "zstd",
            "compression_level": 3,
            "use_dictionary": True,
            "write_statistics": True,
            "data_page_size": self.data_page_bytes,
            "write_batch_size": 64 * 1024,
            "store_schema": True,
        }

    def write(self, blocks: Iterable[Block], ctx: TaskContext) -> dict[str, Any]:
        if not self._started:
            raise StreamingParquetSinkError(
                "on_write_start must run before the first writer task"
            )

        import pyarrow as pa

        task_index = int(ctx.task_idx)
        attempt_id = uuid.uuid4().hex
        attempt_root = self._transaction_root / (
            f"task-{task_index:08d}-attempt-{attempt_id}"
        )
        _require_owned(attempt_root, self._transaction_root)
        try:
            attempt_root.mkdir(parents=True, mode=0o700)
        except BaseException as error:
            cleanup_errors = _remove_owned_tree(attempt_root, self._transaction_root)
            if cleanup_errors:
                raise StreamingParquetSinkRollbackError(
                    "writer setup failure left transaction-owned files: "
                    + "; ".join(cleanup_errors)
                ) from error
            normalized = _normalize_failure(
                error, phase="Parquet writer task setup", path=attempt_root
            )
            if normalized is error:
                raise
            raise normalized from error

        writer_started_at_ns = time.time_ns()
        started = time.perf_counter()
        schema = None
        schema_base64 = None
        writer = None
        partial_path: Optional[Path] = None
        local_file_index = 0
        file_rows = 0
        file_decoded_bytes = 0
        file_row_groups = 0
        file_max_row_group_bytes = 0
        group_parts: list[Any] = []
        group_rows = 0
        group_decoded_bytes = 0
        ready_files: list[dict[str, Any]] = []

        input_blocks = 0
        skipped_empty_blocks = 0
        split_count = 0
        parquet_write_s = 0.0
        parquet_write_cpu_s = 0.0
        parquet_close_s = 0.0
        parquet_close_cpu_s = 0.0
        file_fsync_s = 0.0
        staging_rename_s = 0.0
        peak_group_decoded_bytes = 0
        peak_file_decoded_bytes = 0
        input_decoded_bytes = 0
        first_parquet_write_started_at_ns: Optional[int] = None
        first_parquet_file_completed_at_ns: Optional[int] = None
        last_parquet_file_completed_at_ns: Optional[int] = None

        def open_file() -> None:
            nonlocal writer, partial_path
            assert schema is not None
            assert writer is None
            partial_path = attempt_root / (
                f"chunk-{local_file_index:04d}.parquet.partial"
            )
            writer = _open_parquet_writer(partial_path, schema, self._writer_options())

        def flush_group() -> None:
            nonlocal group_parts, group_rows, group_decoded_bytes
            nonlocal file_row_groups, file_max_row_group_bytes, parquet_write_s
            nonlocal parquet_write_cpu_s, first_parquet_write_started_at_ns
            if not group_parts:
                return
            assert writer is not None
            table = (
                group_parts[0]
                if len(group_parts) == 1
                else pa.concat_tables(group_parts, promote_options="none")
            )
            if first_parquet_write_started_at_ns is None:
                first_parquet_write_started_at_ns = time.time_ns()
            before = time.perf_counter()
            before_cpu = time.process_time()
            writer.write_table(table, row_group_size=table.num_rows)
            parquet_write_cpu_s += time.process_time() - before_cpu
            parquet_write_s += time.perf_counter() - before
            file_row_groups += 1
            file_max_row_group_bytes = max(
                file_max_row_group_bytes, group_decoded_bytes
            )
            group_parts = []
            group_rows = 0
            group_decoded_bytes = 0

        def close_file() -> None:
            nonlocal writer, partial_path, local_file_index
            nonlocal file_rows, file_decoded_bytes, file_row_groups
            nonlocal file_max_row_group_bytes, file_fsync_s, staging_rename_s
            nonlocal parquet_close_s, parquet_close_cpu_s
            nonlocal first_parquet_file_completed_at_ns
            nonlocal last_parquet_file_completed_at_ns
            if writer is None:
                return
            flush_group()
            before = time.perf_counter()
            before_cpu = time.process_time()
            writer.close()
            parquet_close_cpu_s += time.process_time() - before_cpu
            file_parquet_close_s = time.perf_counter() - before
            parquet_close_s += file_parquet_close_s
            assert partial_path is not None

            before = time.perf_counter()
            _fsync_file(partial_path)
            file_fsync_s += time.perf_counter() - before
            ready_path = attempt_root / (f"chunk-{local_file_index:04d}.parquet.staged")
            before = time.perf_counter()
            _atomic_rename(partial_path, ready_path)
            _fsync_directory(attempt_root)
            staging_rename_s += time.perf_counter() - before
            file_completed_at_ns = time.time_ns()
            if first_parquet_file_completed_at_ns is None:
                first_parquet_file_completed_at_ns = file_completed_at_ns
            last_parquet_file_completed_at_ns = file_completed_at_ns
            ready_files.append(
                {
                    "task_file_index": local_file_index,
                    "staged_relative_path": str(
                        ready_path.relative_to(self.output_directory)
                    ),
                    "num_rows": file_rows,
                    "decoded_bytes": file_decoded_bytes,
                    "file_size_bytes": ready_path.stat().st_size,
                    "row_group_count": file_row_groups,
                    "max_row_group_decoded_bytes": file_max_row_group_bytes,
                    "parquet_close_s": file_parquet_close_s,
                    "completed_at_ns": file_completed_at_ns,
                }
            )
            writer = None
            partial_path = None
            local_file_index += 1
            file_rows = 0
            file_decoded_bytes = 0
            file_row_groups = 0
            file_max_row_group_bytes = 0

        try:
            for block in blocks:
                input_blocks += 1
                accessor = BlockAccessor.for_block(block)
                block_decoded_bytes = accessor.size_bytes()
                if (
                    input_decoded_bytes + block_decoded_bytes
                    > self.target_bytes_per_write
                ):
                    raise StreamingParquetSinkError(
                        "writer input exceeded the configured decoded-byte bound: "
                        f"task={task_index}, prior_bytes={input_decoded_bytes}, "
                        f"block_bytes={block_decoded_bytes}, "
                        f"bound={self.target_bytes_per_write}"
                    )
                input_decoded_bytes += block_decoded_bytes
                if accessor.num_rows() == 0:
                    skipped_empty_blocks += 1
                    continue
                table = accessor.to_arrow()
                if table.num_rows == 0:
                    skipped_empty_blocks += 1
                    continue
                if schema is None:
                    schema = table.schema
                    schema_base64 = _schema_to_base64(schema)
                    if self._schema_base64 is not None and not _encoded_schemas_equal(
                        schema_base64, self._schema_base64
                    ):
                        raise StreamingParquetSinkError(
                            "writer input schema does not match on_write_start schema"
                        )
                elif not table.schema.equals(schema, check_metadata=True):
                    raise StreamingParquetSinkError(
                        f"task {task_index} received inconsistent Arrow schemas"
                    )

                offset = 0
                while offset < table.num_rows:
                    if writer is None:
                        open_file()
                    file_remaining = self.max_decoded_file_bytes - file_decoded_bytes
                    group_remaining = self.row_group_decoded_bytes - group_decoded_bytes
                    limit = min(file_remaining, group_remaining)
                    remainder = table.slice(offset)
                    rows = _largest_prefix_within(remainder, limit)
                    if rows <= 0:
                        # Variable-width rows do not necessarily land exactly on a
                        # byte target.  Seal the partially filled row group/file
                        # and retry the row against a fresh full-size budget.
                        if group_decoded_bytes > 0:
                            flush_group()
                            continue
                        if file_decoded_bytes > 0:
                            close_file()
                            continue
                        one_row_bytes = remainder.slice(0, 1).nbytes
                        raise StreamingParquetSinkError(
                            "one Arrow row exceeds the configured decoded-byte "
                            f"bound: row_bytes={one_row_bytes}, bound={limit}"
                        )
                    piece = remainder.slice(0, rows)
                    if rows < remainder.num_rows:
                        split_count += 1
                    piece_bytes = piece.nbytes
                    group_parts.append(piece)
                    group_rows += piece.num_rows
                    group_decoded_bytes += piece_bytes
                    file_rows += piece.num_rows
                    file_decoded_bytes += piece_bytes
                    offset += piece.num_rows
                    peak_group_decoded_bytes = max(
                        peak_group_decoded_bytes, group_decoded_bytes
                    )
                    peak_file_decoded_bytes = max(
                        peak_file_decoded_bytes, file_decoded_bytes
                    )

                    if group_decoded_bytes >= self.row_group_decoded_bytes:
                        flush_group()
                    if file_decoded_bytes >= self.max_decoded_file_bytes:
                        close_file()

            close_file()
            if not ready_files:
                try:
                    attempt_root.rmdir()
                except OSError:
                    pass
            writer_completed_at_ns = time.time_ns()
            return {
                "task_index": task_index,
                "schema_base64": schema_base64,
                "files": ready_files,
                "metrics": {
                    "input_blocks": input_blocks,
                    "input_decoded_bytes": input_decoded_bytes,
                    "skipped_empty_blocks": skipped_empty_blocks,
                    "num_rows": sum(item["num_rows"] for item in ready_files),
                    "decoded_bytes": sum(item["decoded_bytes"] for item in ready_files),
                    "file_size_bytes": sum(
                        item["file_size_bytes"] for item in ready_files
                    ),
                    "file_count": len(ready_files),
                    "split_count": split_count,
                    "peak_row_group_decoded_bytes": peak_group_decoded_bytes,
                    "peak_file_decoded_bytes": peak_file_decoded_bytes,
                    "parquet_write_s": parquet_write_s,
                    "parquet_write_cpu_s": parquet_write_cpu_s,
                    "parquet_close_s": parquet_close_s,
                    "parquet_close_cpu_s": parquet_close_cpu_s,
                    "parquet_encoding_cpu_s": (
                        parquet_write_cpu_s + parquet_close_cpu_s
                    ),
                    "file_fsync_s": file_fsync_s,
                    "staging_rename_s": staging_rename_s,
                    "elapsed_s": time.perf_counter() - started,
                    "writer_started_at_ns": writer_started_at_ns,
                    "first_parquet_write_started_at_ns": (
                        first_parquet_write_started_at_ns
                    ),
                    "first_parquet_file_completed_at_ns": (
                        first_parquet_file_completed_at_ns
                    ),
                    "last_parquet_file_completed_at_ns": (
                        last_parquet_file_completed_at_ns
                    ),
                    "writer_completed_at_ns": writer_completed_at_ns,
                },
            }
        except BaseException as error:
            if writer is not None:
                try:
                    writer.close()
                except BaseException:
                    pass
            cleanup_errors = _remove_owned_tree(attempt_root, self._transaction_root)
            if cleanup_errors:
                raise StreamingParquetSinkRollbackError(
                    "writer failure left transaction-owned files: "
                    + "; ".join(cleanup_errors)
                ) from error
            normalized = _normalize_failure(
                error, phase="Parquet writer task", path=partial_path or attempt_root
            )
            if normalized is error:
                raise
            raise normalized from error

    def _validate_and_order_returns(
        self, write_returns: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], Optional[str]]:
        by_task: dict[int, dict[str, Any]] = {}
        for result in write_returns:
            if not isinstance(result, dict) or "task_index" not in result:
                raise StreamingParquetSinkError("malformed writer task result")
            task_index = int(result["task_index"])
            if task_index in by_task:
                raise StreamingParquetSinkError(
                    f"duplicate writer result for task {task_index}"
                )
            by_task[task_index] = result
        task_indices = sorted(by_task)
        if task_indices and task_indices != list(range(task_indices[-1] + 1)):
            raise StreamingParquetSinkError(
                f"writer results have non-contiguous task indices: {task_indices}"
            )
        schemas = [
            by_task[task_index]["schema_base64"]
            for task_index in task_indices
            if by_task[task_index].get("schema_base64") is not None
        ]
        returned_schema_base64 = schemas[0] if schemas else None
        if returned_schema_base64 is not None and any(
            not _encoded_schemas_equal(returned_schema_base64, schema)
            for schema in schemas[1:]
        ):
            raise StreamingParquetSinkError(
                "writer tasks returned different Arrow schemas"
            )
        schema_base64 = self._schema_base64 or returned_schema_base64
        if (
            self._schema_base64 is not None
            and returned_schema_base64 is not None
            and not _encoded_schemas_equal(self._schema_base64, returned_schema_base64)
        ):
            raise StreamingParquetSinkError(
                "writer task schema does not match the sink input schema"
            )

        ordered_files: list[dict[str, Any]] = []
        for task_index in sorted(by_task):
            result = by_task[task_index]
            files = sorted(
                result.get("files", []), key=lambda item: item["task_file_index"]
            )
            if [item["task_file_index"] for item in files] != list(range(len(files))):
                raise StreamingParquetSinkError(
                    f"task {task_index} returned non-contiguous staged files"
                )
            for item in files:
                if item["num_rows"] <= 0:
                    raise StreamingParquetSinkError(
                        f"task {task_index} returned an empty staged Parquet file"
                    )
                if not 0 <= item["decoded_bytes"] <= self.max_decoded_file_bytes:
                    raise StreamingParquetSinkError(
                        f"task {task_index} exceeded the decoded file bound"
                    )
                if not (
                    0
                    <= item["max_row_group_decoded_bytes"]
                    <= self.row_group_decoded_bytes
                ):
                    raise StreamingParquetSinkError(
                        f"task {task_index} exceeded the decoded row-group bound"
                    )
                ordered_files.append({"task_index": task_index, **item})
        return ordered_files, schema_base64

    def _rollback_driver(self) -> list[str]:
        errors: list[str] = []
        quarantine_errors: list[str] = []

        # A successful publish removes the empty rollback root immediately
        # before syncfs. Recreate it if syncfs then fails, so already-published
        # owned files can still be atomically removed from public paths.
        if self._rollback_root_identity is None:
            try:
                self._create_rollback_root()
            except Exception as error:
                self._rollback_root_unsafe_reason = (
                    "could not recreate the private rollback root at "
                    f"{self._rollback_root}: {type(error).__name__}: {error}"
                )
                quarantine_errors.append(self._rollback_root_unsafe_reason)
        elif not self._rollback_root_is_owned():
            self._rollback_root_unsafe_reason = (
                f"rollback root identity changed at {self._rollback_root}"
            )
            quarantine_errors.append(self._rollback_root_unsafe_reason)

        if self._rollback_root_is_owned():
            for owned in reversed(self._owned_final_paths):
                error = _quarantine_if_owned(owned, self._rollback_root)
                if error is not None:
                    quarantine_errors.append(error)
        else:
            # Without an owned quarantine namespace, keep every public path in
            # place. Closing the pins is safe; deleting by pathname is not.
            for owned in self._owned_final_paths:
                _release_owned_file(owned)
        self._owned_final_paths = []

        if self._rollback_root_unsafe_reason is not None:
            quarantine_errors.append(self._rollback_root_unsafe_reason)
        quarantine_errors = list(dict.fromkeys(quarantine_errors))
        errors.extend(quarantine_errors)

        if not quarantine_errors and self._rollback_root_is_owned():
            private_cleanup_errors = _remove_owned_tree(
                self._rollback_root, self._rollback_root
            )
            errors.extend(private_cleanup_errors)
            if not private_cleanup_errors:
                self._rollback_root_identity = None
        errors.extend(
            _remove_owned_tree(self._transaction_root, self._transaction_root)
        )
        staging_root = self.output_directory / ".staging"
        try:
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
        except OSError as error:
            errors.append(f"{staging_root}: {error}")
        try:
            if self.output_directory.exists():
                _fsync_directory(self.output_directory)
        except OSError as error:
            errors.append(f"{self.output_directory}: {error}")
        return errors

    def on_write_complete(self, write_result: WriteResult[dict[str, Any]]) -> None:
        if not self._started:
            raise StreamingParquetSinkError("cannot commit a sink that was not started")
        if self._committed:
            raise StreamingParquetSinkError("sink transaction is already committed")

        commit_started_at_ns = time.time_ns()
        commit_started = time.perf_counter()
        commit_rename_s = 0.0
        manifest_write_s = 0.0
        success_write_s = 0.0
        syncfs_s = 0.0
        try:
            ordered_staged, schema_base64 = self._validate_and_order_returns(
                write_result.write_returns
            )
            final_files: list[dict[str, Any]] = []
            for global_index, item in enumerate(ordered_staged):
                staged = self.output_directory / item["staged_relative_path"]
                _require_owned(staged, self._transaction_root)
                if not staged.is_file():
                    raise StreamingParquetSinkError(
                        f"staged Parquet file is missing: {staged}"
                    )
                actual_size = staged.stat().st_size
                if actual_size != item["file_size_bytes"]:
                    raise StreamingParquetSinkError(
                        f"staged Parquet size changed for {staged}: "
                        f"expected {item['file_size_bytes']}, found {actual_size}"
                    )
                final = self.output_directory / f"part-{global_index:05d}.parquet"
                final_ownership = _owned_file(staged).at(final)
                before = time.perf_counter()
                try:
                    _atomic_rename(staged, final)
                    _verify_owned_destination(final_ownership)
                except BaseException:
                    _release_owned_file(final_ownership)
                    raise
                self._owned_final_paths.append(final_ownership)
                commit_rename_s += time.perf_counter() - before
                final_files.append(
                    {
                        "path": final.name,
                        "file_index": global_index,
                        "task_index": item["task_index"],
                        "task_file_index": item["task_file_index"],
                        "num_rows": item["num_rows"],
                        "decoded_bytes": item["decoded_bytes"],
                        "file_size_bytes": item["file_size_bytes"],
                        "logical_to_physical_ratio": (
                            item["decoded_bytes"] / item["file_size_bytes"]
                            if item["file_size_bytes"]
                            else None
                        ),
                        "row_group_count": item["row_group_count"],
                        "max_row_group_decoded_bytes": item[
                            "max_row_group_decoded_bytes"
                        ],
                    }
                )
            _fsync_directory(self.output_directory)

            cleanup_errors = _remove_owned_tree(
                self._transaction_root, self._transaction_root
            )
            if cleanup_errors:
                raise StreamingParquetSinkRollbackError(
                    "could not remove staging before commit: "
                    + "; ".join(cleanup_errors)
                )
            staging_root = self.output_directory / ".staging"
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()
            _fsync_directory(self.output_directory)

            total_decoded_bytes = sum(item["decoded_bytes"] for item in final_files)
            total_file_size_bytes = sum(item["file_size_bytes"] for item in final_files)
            manifest = {
                "format_version": 1,
                "format": "parquet",
                "ordered": True,
                "schema_base64": schema_base64,
                "writer": {
                    "parquet_version": "2.6",
                    "compression": "zstd",
                    "compression_level": 3,
                    "dictionary_encoding": True,
                    "write_statistics": True,
                    "data_page_bytes": self.data_page_bytes,
                    "row_group_decoded_bytes": self.row_group_decoded_bytes,
                    "max_decoded_file_bytes": self.max_decoded_file_bytes,
                    "target_bytes_per_write": self.target_bytes_per_write,
                    "recommended_max_writers": self.recommended_max_writers,
                },
                "totals": {
                    "num_rows": sum(item["num_rows"] for item in final_files),
                    "decoded_bytes": total_decoded_bytes,
                    "file_size_bytes": total_file_size_bytes,
                    "logical_to_physical_ratio": (
                        total_decoded_bytes / total_file_size_bytes
                        if total_file_size_bytes
                        else None
                    ),
                    "file_count": len(final_files),
                },
                "files": final_files,
            }
            manifest_payload = (
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            manifest_path = self.output_directory / "manifest.json"
            before = time.perf_counter()
            manifest_ownership = _atomic_write(
                manifest_path, manifest_payload, self._rollback_root
            )
            self._owned_final_paths.append(manifest_ownership)
            _fsync_directory(self.output_directory)
            manifest_write_s = time.perf_counter() - before
            manifest_closed_at_ns = time.time_ns()

            success_payload = (
                json.dumps(
                    {
                        "manifest": manifest_path.name,
                        "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            success_path = self.output_directory / "_SUCCESS"
            before = time.perf_counter()
            success_ownership = _atomic_write(
                success_path, success_payload, self._rollback_root
            )
            self._owned_final_paths.append(success_ownership)
            _fsync_directory(self.output_directory)
            success_write_s = time.perf_counter() - before
            success_closed_at_ns = time.time_ns()

            self._remove_empty_rollback_root_before_sync()
            _fsync_directory(self.output_directory)
            before = time.perf_counter()
            _syncfs(self.output_directory)
            syncfs_s = time.perf_counter() - before
            drain_completed_at_ns = time.time_ns()

            ordered_task_results = sorted(
                write_result.write_returns, key=lambda item: item["task_index"]
            )
            task_metrics = [
                result.get("metrics", {}) for result in ordered_task_results
            ]
            writer_tasks = [
                {"task_index": int(result["task_index"]), **metrics}
                for result, metrics in zip(ordered_task_results, task_metrics)
            ]
            first_writer_work_started_at_ns = min(
                (
                    item["writer_started_at_ns"]
                    for item in task_metrics
                    if item.get("writer_started_at_ns") is not None
                ),
                default=None,
            )
            first_parquet_file_completed_at_ns = min(
                (
                    item["first_parquet_file_completed_at_ns"]
                    for item in task_metrics
                    if item.get("first_parquet_file_completed_at_ns") is not None
                ),
                default=None,
            )
            last_parquet_file_completed_at_ns = max(
                (
                    item["last_parquet_file_completed_at_ns"]
                    for item in task_metrics
                    if item.get("last_parquet_file_completed_at_ns") is not None
                ),
                default=None,
            )
            self._telemetry = {
                "transaction_committed": True,
                "writer_task_count": len(write_result.write_returns),
                "output_file_count": len(final_files),
                "num_rows": manifest["totals"]["num_rows"],
                "decoded_bytes": manifest["totals"]["decoded_bytes"],
                "file_size_bytes": manifest["totals"]["file_size_bytes"],
                "logical_to_physical_ratio": manifest["totals"][
                    "logical_to_physical_ratio"
                ],
                "writer_elapsed_s_sum": sum(
                    item.get("elapsed_s", 0.0) for item in task_metrics
                ),
                "writer_elapsed_s_max": max(
                    (item.get("elapsed_s", 0.0) for item in task_metrics),
                    default=0.0,
                ),
                "writer_input_decoded_bytes_sum": sum(
                    item.get("input_decoded_bytes", 0) for item in task_metrics
                ),
                "writer_input_decoded_bytes_max": max(
                    (item.get("input_decoded_bytes", 0) for item in task_metrics),
                    default=0,
                ),
                "configured_peak_writer_input_bytes": (
                    self.target_bytes_per_write * self.recommended_max_writers
                ),
                "writer_input_target_overshoot_tasks": sum(
                    item.get("input_decoded_bytes", 0) > self.target_bytes_per_write
                    for item in task_metrics
                ),
                "parquet_write_s_sum": sum(
                    item.get("parquet_write_s", 0.0) for item in task_metrics
                ),
                "parquet_close_s_sum": sum(
                    item.get("parquet_close_s", 0.0) for item in task_metrics
                ),
                "parquet_write_cpu_s_sum": sum(
                    item.get("parquet_write_cpu_s", 0.0) for item in task_metrics
                ),
                "parquet_close_cpu_s_sum": sum(
                    item.get("parquet_close_cpu_s", 0.0) for item in task_metrics
                ),
                "parquet_encoding_cpu_s_sum": sum(
                    item.get("parquet_encoding_cpu_s", 0.0) for item in task_metrics
                ),
                "file_fsync_s_sum": sum(
                    item.get("file_fsync_s", 0.0) for item in task_metrics
                ),
                "staging_rename_s_sum": sum(
                    item.get("staging_rename_s", 0.0) for item in task_metrics
                ),
                "commit_rename_s": commit_rename_s,
                "manifest_write_s": manifest_write_s,
                "success_write_s": success_write_s,
                "syncfs_s": syncfs_s,
                "commit_elapsed_s": time.perf_counter() - commit_started,
                "skipped_empty_blocks": sum(
                    item.get("skipped_empty_blocks", 0) for item in task_metrics
                ),
                "skipped_empty_partitions": sum(
                    item.get("file_count", 0) == 0 for item in task_metrics
                ),
                "split_count": sum(item.get("split_count", 0) for item in task_metrics),
                "peak_row_group_decoded_bytes": max(
                    (
                        item.get("peak_row_group_decoded_bytes", 0)
                        for item in task_metrics
                    ),
                    default=0,
                ),
                "peak_file_decoded_bytes": max(
                    (item.get("peak_file_decoded_bytes", 0) for item in task_metrics),
                    default=0,
                ),
                "write_started_at_ns": self._write_started_at_ns,
                "first_writer_work_started_at_ns": first_writer_work_started_at_ns,
                "first_parquet_file_completed_at_ns": (
                    first_parquet_file_completed_at_ns
                ),
                "last_parquet_file_completed_at_ns": (
                    last_parquet_file_completed_at_ns
                ),
                "commit_started_at_ns": commit_started_at_ns,
                "manifest_closed_at_ns": manifest_closed_at_ns,
                "success_closed_at_ns": success_closed_at_ns,
                "drain_completed_at_ns": drain_completed_at_ns,
                "rollback_errors": [],
            }
            self._result = {
                "telemetry": dict(self._telemetry),
                "writer_tasks": writer_tasks,
                "manifest": manifest,
                "success": json.loads(success_payload),
            }
            self._committed = True
            for owned in self._owned_final_paths:
                _release_owned_file(owned)
            self._owned_final_paths = []
        except BaseException as error:
            if isinstance(error, _PrivateRollbackRootUnsafe):
                self._rollback_root_unsafe_reason = str(error)
            rollback_errors = self._rollback_driver()
            self._committed = False
            self._result = None
            self._telemetry = {
                "transaction_committed": False,
                "rollback_errors": rollback_errors,
                "commit_elapsed_s": time.perf_counter() - commit_started,
            }
            if rollback_errors:
                raise StreamingParquetSinkRollbackError(
                    "commit failure left transaction-owned files: "
                    + "; ".join(rollback_errors)
                ) from error
            normalized = _normalize_failure(
                error, phase="Parquet transaction commit", path=self.output_directory
            )
            if normalized is error:
                raise
            raise normalized from error

    def on_write_failed(self, error: Exception) -> None:
        if self._committed:
            return
        rollback_errors = self._rollback_driver()
        self._result = None
        self._telemetry = {
            "transaction_committed": False,
            "failure": repr(error),
            "rollback_errors": rollback_errors,
        }


__all__ = [
    "DEFAULT_DATA_PAGE_BYTES",
    "DEFAULT_MAX_DECODED_FILE_BYTES",
    "DEFAULT_MAX_WRITERS",
    "DEFAULT_ROW_GROUP_DECODED_BYTES",
    "DEFAULT_TARGET_BYTES_PER_WRITE",
    "StreamingParquetDatasink",
    "StreamingParquetSinkCapacityError",
    "StreamingParquetSinkError",
    "StreamingParquetSinkRollbackError",
]
