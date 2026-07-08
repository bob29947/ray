import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ray
from ray.data.tests.conftest import *  # noqa: F403

cudf = pytest.importorskip("cudf")
cupy = pytest.importorskip("cupy")


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
def test_minimal_default_path_matches_expected_rows_without_shuffle(
    ray_with_one_gpu,
    restore_data_context,
    tmp_path,  # noqa: F811
):
    class IdentityTokenizer:
        def __init__(self):
            self.instance_id = uuid.uuid4().hex

        def __call__(self, batch):
            return batch.assign(tokenizer_instance=self.instance_id)

    def identity_partition(batch, context):
        assert context.group_keys == ("User", "Card")
        assert context.input_group_boundaries[0] == 0
        assert context.input_group_boundaries[-1] == len(batch)
        return batch

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

    tokenized = ray.data.read_parquet(str(path)).map_batches(
        IdentityTokenizer,
        batch_size=2,
        batch_format="cudf",
        num_gpus=1,
    )
    output = tokenized.groupby(["User", "Card"]).map_group_partitions(
        identity_partition,
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
    assert "ParquetCudfMapGroupPartitions" in stats
    assert "ranges=2" in stats
    assert "Repartition" not in stats
