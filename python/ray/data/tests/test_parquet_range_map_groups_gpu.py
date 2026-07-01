"""Conditional end-to-end coverage for Parquet range ``map_groups``.

This module is importable on CPU-only hosts.  GPU libraries and hardware are
checked by fixtures at runtime so normal test collection remains unaffected.
"""

import importlib.util
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray import cloudpickle
from ray.data import ActorPoolStrategy, TaskPoolStrategy
from ray.data.context import DataContext, ShuffleStrategy


pytestmark = pytest.mark.gpu

# Test modules aren't importable from worker processes in every invocation mode.
# Serialize these tiny test UDFs by value so the conditional GPU suite works both
# under Bazel and when this file is selected directly with pytest.
cloudpickle.register_pickle_by_value(sys.modules[__name__])

_RANGE_BACKEND_CONFIG = "parquet_range_map_groups_enabled"
_PROJECTION = ["User", "Card", "Amount"]
_GROUP_KEYS = ["User", "Card"]


class _GpuAssignmentCollector:
    def __init__(self):
        self._records = []

    def record(self, record):
        self._records.append(record)

    def records(self, run_label):
        return [record for record in self._records if record["run_label"] == run_label]


class _CudfTokenizer:
    __ray_data_preserves_partitioning__ = ("User",)

    def __init__(
        self,
        collector,
        constructor_multiplier,
        *,
        constructor_offset,
        constructor_tag,
    ):
        self._collector = collector
        self._constructor_multiplier = constructor_multiplier
        self._constructor_offset = constructor_offset
        self._constructor_tag = constructor_tag

    def __call__(
        self,
        batch,
        call_multiplier,
        *,
        call_offset,
        run_label,
    ):
        output = batch.copy(deep=True)
        output["Token"] = (
            output["Amount"] * self._constructor_multiplier + self._constructor_offset
        ) * call_multiplier + call_offset
        accelerator_ids = tuple(
            str(value)
            for value in ray.get_runtime_context().get_accelerator_ids().get("GPU", ())
        )
        ray.get(
            self._collector.record.remote(
                {
                    "run_label": run_label,
                    "user_min": int(output["User"].min()),
                    "user_max": int(output["User"].max()),
                    "rows": len(output),
                    "accelerator_ids": accelerator_ids,
                    "constructor_tag": self._constructor_tag,
                    "call_multiplier": call_multiplier,
                }
            )
        )
        return output


def _summarize_group(group, *, summary_offset):
    output = group[["User", "Card"]].iloc[:1].reset_index(drop=True)
    output["Rows"] = len(group)
    output["TokenSum"] = group["Token"].sum() + summary_offset
    return output


@pytest.fixture(scope="module")
def ray_with_two_gpus():
    pytest.importorskip("cudf", reason="cudf (GPU DataFrame library) is not installed")
    cupy = pytest.importorskip("cupy", reason="cupy is not installed")
    try:
        visible_gpus = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        pytest.skip(f"CUDA runtime is unavailable: {exc}")
    if visible_gpus < 2:
        pytest.skip("Parquet range map_groups requires at least two visible GPUs")

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(num_cpus=4, num_gpus=2, include_dashboard=False)
    try:
        cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
        if cluster_gpus < 2:
            pytest.skip("Ray cluster exposes fewer than two GPU resources")
        yield cluster_gpus
    finally:
        if started_ray:
            ray.shutdown()


@pytest.fixture
def sorted_parquet_path(tmp_path):
    path = tmp_path / "sorted-ranges.parquet"
    table = pa.table(
        {
            "User": pa.array([0, 0, 1, 1, 100, 100, 101, 101], type=pa.int64()),
            "Card": pa.array([10, 10, 11, 11, 20, 20, 21, 21], type=pa.int64()),
            "Amount": pa.array(range(1, 9), type=pa.int32()),
        }
    )
    pq.write_table(
        table,
        path,
        row_group_size=4,
        write_statistics=True,
        compression="NONE",
    )

    metadata = pq.ParquetFile(path).metadata
    assert metadata.num_row_groups == 2
    user_index = table.schema.get_field_index("User")
    assert [
        (
            metadata.row_group(index).column(user_index).statistics.min,
            metadata.row_group(index).column(user_index).statistics.max,
        )
        for index in range(2)
    ] == [(0, 1), (100, 101)]
    yield path


@pytest.fixture
def gpu_assignment_collector(ray_with_two_gpus):
    collector = ray.remote(_GpuAssignmentCollector).remote()
    try:
        yield collector
    finally:
        ray.kill(collector)


def _run_natural_pipeline(
    path,
    collector,
    *,
    run_label,
    shuffle_strategy,
    range_backend_enabled,
):
    context = DataContext.get_current()
    context.use_datasource_v2 = False
    context.shuffle_strategy = shuffle_strategy
    context.gpu_shuffle_num_actors = 2
    context.set_config(_RANGE_BACKEND_CONFIG, range_backend_enabled)
    context.set_config("parquet_range_map_groups_gpu_memory_bytes", 8 * 1024**3)

    return (
        ray.data.read_parquet(str(path), columns=_PROJECTION)
        .map_batches(
            _CudfTokenizer,
            batch_size=2,
            batch_format="cudf",
            zero_copy_batch=False,
            fn_args=(2,),
            fn_kwargs={"call_offset": 7, "run_label": run_label},
            fn_constructor_args=(collector, 3),
            fn_constructor_kwargs={
                "constructor_offset": 5,
                "constructor_tag": "constructor-args-observed",
            },
            compute=ActorPoolStrategy(size=2),
            num_gpus=1,
            udf_modifying_row_count=False,
        )
        .groupby(_GROUP_KEYS, num_partitions=2)
        .map_groups(
            _summarize_group,
            batch_format="cudf",
            zero_copy_batch=False,
            fn_kwargs={"summary_offset": 11},
            compute=TaskPoolStrategy(size=2),
            num_gpus=1,
        )
        .materialize()
    )


def _logical_output(dataset):
    return dataset.to_pandas().sort_values(_GROUP_KEYS).reset_index(drop=True)


def _plan_stats(dataset):
    # The public summary nests upstream operators under the final logical stage
    # for some all-to-all plans. The rendered stats string includes every
    # physical suboperator and is therefore the stable assertion surface here.
    return dataset.stats()


def _assert_expected_groups(frame):
    assert len(frame) == 4
    assert frame[_GROUP_KEYS].drop_duplicates().shape[0] == 4
    expected = pd.DataFrame(
        {
            "User": np.array([0, 1, 100, 101], dtype=np.int64),
            "Card": np.array([10, 11, 20, 21], dtype=np.int64),
            "Rows": np.array([2, 2, 2, 2], dtype=np.int64),
            # Token = ((Amount * 3 + 5) * 2 + 7); add 11 per group.
            "TokenSum": np.array([63, 87, 111, 135], dtype=np.int64),
        }
    )
    # Exact dtype equality is asserted between the ordinary and fast outputs
    # above. Here, compare expected values without coupling to whether this Ray
    # build asks pandas to use NumPy or Arrow-backed integer dtypes.
    pd.testing.assert_frame_equal(frame, expected, check_dtype=False)


def _assert_fast_range_gpu_assignments(records):
    assert len(records) == 4
    assert sum(record["rows"] for record in records) == 8
    assert all(record["user_max"] < 50 or record["user_min"] > 50 for record in records)
    assert all(
        record["constructor_tag"] == "constructor-args-observed" for record in records
    )
    assert all(record["call_multiplier"] == 2 for record in records)
    assert all(len(record["accelerator_ids"]) == 1 for record in records)

    lower_ids = {
        record["accelerator_ids"][0] for record in records if record["user_max"] < 50
    }
    upper_ids = {
        record["accelerator_ids"][0] for record in records if record["user_min"] > 50
    }
    assert len(lower_ids) == 1
    assert len(upper_ids) == 1
    assert lower_ids.isdisjoint(upper_ids)


def test_natural_api_hash_shuffle_matches_selected_range_backend(
    ray_with_two_gpus,
    restore_data_context,
    sorted_parquet_path,
    gpu_assignment_collector,
):
    hash_output = _run_natural_pipeline(
        sorted_parquet_path,
        gpu_assignment_collector,
        run_label="hash-disabled",
        shuffle_strategy=ShuffleStrategy.HASH_SHUFFLE,
        range_backend_enabled=False,
    )
    fast_output = _run_natural_pipeline(
        sorted_parquet_path,
        gpu_assignment_collector,
        run_label="hash-fast",
        shuffle_strategy=ShuffleStrategy.HASH_SHUFFLE,
        range_backend_enabled=True,
    )

    hash_frame = _logical_output(hash_output)
    fast_frame = _logical_output(fast_output)
    pd.testing.assert_frame_equal(hash_frame, fast_frame, check_dtype=True)
    _assert_expected_groups(fast_frame)

    hash_stats = _plan_stats(hash_output)
    fast_stats = _plan_stats(fast_output)
    assert "HashShuffle" in hash_stats, hash_stats
    assert "ParquetRangeMapGroups" in fast_stats, fast_stats
    assert "Shuffle" not in fast_stats, fast_stats

    extra_metrics = fast_output.get_stats_summary().extra_metrics
    plan_metrics = extra_metrics["parquet_range_map_groups_plan"]
    worker_metrics = extra_metrics["parquet_range_map_groups_workers"]
    assert plan_metrics["selected"] is True
    assert plan_metrics["num_partitions"] == 2
    assert len(plan_metrics["ranges"]) == 2
    assert len(worker_metrics) == 2
    assert sum(item["groups_invoked"] for item in worker_metrics) == 4
    assert all(item["gpu_peak_memory_bytes"] > 0 for item in worker_metrics)
    assert all(item["host_transfer_bytes"] > 0 for item in worker_metrics)
    assert all(item["rmm_pool_initial_bytes"] > 0 for item in worker_metrics)
    assert all(
        item["rmm_pool_maximum_bytes"] >= item["rmm_pool_initial_bytes"]
        for item in worker_metrics
    )

    records = ray.get(gpu_assignment_collector.records.remote("hash-fast"))
    _assert_fast_range_gpu_assignments(records)


def test_group_boundaries_keep_float_nan_keys_together(ray_with_two_gpus, tmp_path):
    import cudf
    import cupy

    from ray.data._internal.planner.parquet_range_map_groups import _group_boundaries

    path = tmp_path / "nan-group-key.parquet"
    pq.write_table(
        pa.table(
            {
                "User": pa.array([1, 1, 1], type=pa.int64()),
                "Card": pa.array([1.0, np.nan, np.nan], type=pa.float64()),
            }
        ),
        path,
    )
    frame = cudf.read_parquet(path).sort_values(["User", "Card"], ignore_index=True)

    assert _group_boundaries(frame, ("User", "Card"), cupy) == [0, 1, 3]


def test_compiled_group_key_order_check(ray_with_two_gpus):
    import cudf

    from ray.data._internal.planner.parquet_range_map_groups import (
        _is_sorted_by_group_keys,
    )

    sorted_frame = cudf.DataFrame({"User": [1, 1, 1, 2], "Card": [1.0, 2.0, None, 0.0]})
    unsorted_frame = cudf.DataFrame({"User": [1, 2, 1], "Card": [1, 0, 2]})

    assert _is_sorted_by_group_keys(sorted_frame, ("User", "Card"))
    assert not _is_sorted_by_group_keys(unsorted_frame, ("User", "Card"))


def test_natural_api_gpu_shuffle_matches_selected_range_backend_when_available(
    ray_with_two_gpus,
    restore_data_context,
    sorted_parquet_path,
    gpu_assignment_collector,
):
    pytest.importorskip("rapidsmpf", reason="rapidsmpf is not installed")
    if (
        importlib.util.find_spec("ucxx") is None
        and importlib.util.find_spec("ucp") is None
    ):
        pytest.skip("UCXX Python bindings are not installed")

    gpu_shuffle_output = _run_natural_pipeline(
        sorted_parquet_path,
        gpu_assignment_collector,
        run_label="gpu-shuffle-disabled",
        shuffle_strategy=ShuffleStrategy.GPU_SHUFFLE,
        range_backend_enabled=False,
    )
    fast_output = _run_natural_pipeline(
        sorted_parquet_path,
        gpu_assignment_collector,
        run_label="gpu-shuffle-fast",
        shuffle_strategy=ShuffleStrategy.GPU_SHUFFLE,
        range_backend_enabled=True,
    )

    gpu_shuffle_frame = _logical_output(gpu_shuffle_output)
    fast_frame = _logical_output(fast_output)
    pd.testing.assert_frame_equal(
        gpu_shuffle_frame,
        fast_frame,
        check_dtype=True,
    )
    _assert_expected_groups(fast_frame)

    gpu_stats = _plan_stats(gpu_shuffle_output)
    fast_stats = _plan_stats(fast_output)
    assert "GPUShuffle" in gpu_stats, gpu_stats
    assert "ParquetRangeMapGroups" in fast_stats, fast_stats
    assert "Shuffle" not in fast_stats, fast_stats

    records = ray.get(gpu_assignment_collector.records.remote("gpu-shuffle-fast"))
    _assert_fast_range_gpu_assignments(records)


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-v", __file__]))
