import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data import (
    ActorPoolStrategy,
    ParquetCudfShuffleElisionConfig,
    TaskPoolStrategy,
)
from ray.data.context import DataContext, ShuffleStrategy
from ray.data.tests.conftest import *  # noqa: F403

cudf = pytest.importorskip("cudf")
cupy = pytest.importorskip("cupy")

GIB = 1024**3


@pytest.fixture
def ray_with_one_gpu(shutdown_only):  # noqa: F811
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            pytest.skip("No CUDA device is visible")
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA runtime is unavailable")
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=1)
    if ray.cluster_resources().get("GPU", 0) < 1:
        pytest.skip("Ray has no GPU resource")


@pytest.mark.gpu
def test_selected_path_matches_map_groups_without_shuffle(
    ray_with_one_gpu,
    restore_data_context,
    tmp_path,  # noqa: F811
):
    class IdentityTokenizer:
        def __init__(self):
            self.instance_id = uuid.uuid4().hex

        def __call__(self, batch):
            return batch.assign(tokenizer_instance=self.instance_id)

    def identity_group(batch):
        return batch

    def identity_partition(batch, context):
        assert context.group_keys == ("User", "Card")
        assert context.input_group_boundaries[0] == 0
        assert context.input_group_boundaries[-1] == len(batch)
        return batch

    DataContext.get_current().shuffle_strategy = ShuffleStrategy.HASH_SHUFFLE
    path = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "User": [0, 0, 1, 1],
                "Card": [1, 2, 1, 2],
                "value": [10, 20, 30, 40],
            }
        ),
        path,
        row_group_size=2,
    )

    tokenized = ray.data.read_parquet(
        str(path), columns=["User", "Card", "value"]
    ).map_batches(
        IdentityTokenizer,
        batch_size=2,
        batch_format="cudf",
        zero_copy_batch=True,
        compute=ActorPoolStrategy(size=1, max_tasks_in_flight_per_actor=1),
        num_cpus=1,
        num_gpus=1,
        udf_modifying_row_count=False,
    )
    config = ParquetCudfShuffleElisionConfig(
        partition_fn=identity_partition,
        shuffle_bytes_per_input_row=GIB,
        peak_gpu_bytes_per_input_row=1,
        gpu_memory_bytes=32 * GIB,
    )
    output = tokenized.groupby(["User", "Card"], num_partitions=2).map_groups(
        identity_group,
        batch_format="cudf",
        zero_copy_batch=True,
        compute=TaskPoolStrategy(size=1),
        num_cpus=1,
        num_gpus=1,
        parquet_cudf_shuffle_elision=config,
    )

    rows = output.take_all()
    assert len({row.pop("tokenizer_instance") for row in rows}) == 1
    rows.sort(key=lambda row: (row["User"], row["Card"]))
    assert rows == [
        {"User": 0, "Card": 1, "value": 10},
        {"User": 0, "Card": 2, "value": 20},
        {"User": 1, "Card": 1, "value": 30},
        {"User": 1, "Card": 2, "value": 40},
    ]
    stats = output.stats()
    assert "ParquetCudfShuffleElision" in stats
    assert "ranges=2" in stats
    assert "Repartition" not in stats
