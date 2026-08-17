from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

WORKER_PATH = Path(__file__).with_name("worker.py")
SPEC = importlib.util.spec_from_file_location("cloud_read_fusion_worker", WORKER_PATH)
assert SPEC is not None and SPEC.loader is not None
worker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = worker
SPEC.loader.exec_module(worker)


def _manifest() -> dict:
    return {
        "columns": list(worker.TOKENIZER_COLUMNS),
        "cohorts": {
            "8x": {
                "objects": [
                    {
                        "uri": "s3://benchmark/run-8x/part-000.parquet",
                        "bytes": 11,
                        "etag": "abc",
                    },
                    {
                        "uri": "s3://benchmark/run-8x/part-001.parquet",
                        "bytes": 13,
                        "etag": "def",
                    },
                ],
                "rows": 42,
                "compressed_bytes": 24,
                "expected_tokenizer_checksum": {
                    "algorithm": worker.CHECKSUM_ALGORITHM,
                    "rows": 42,
                    "sum1": "0123456789abcdef",
                    "sum2": "fedcba9876543210",
                },
            }
        },
    }


def _write_manifest(tmp_path: Path, document: dict | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document or _manifest()), encoding="utf-8")
    return path


def test_load_cohort_needs_only_the_timed_s3_inventory(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)

    cohort = worker.load_cohort(path, "8x")

    assert cohort.paths == (
        "s3://benchmark/run-8x/part-000.parquet",
        "s3://benchmark/run-8x/part-001.parquet",
    )
    assert cohort.rows == 42
    assert cohort.columns == worker.TOKENIZER_COLUMNS
    assert cohort.compressed_bytes == 24
    assert cohort.object_count == 2
    assert not hasattr(cohort, "smoke_uri")


@pytest.mark.parametrize(
    "change,match",
    [
        (
            lambda document: document["cohorts"]["8x"].update(
                paths=[
                    "s3://benchmark/run-8x/same.parquet",
                    "s3://benchmark/run-8x/same.parquet",
                ]
            ),
            "repeats",
        ),
        (
            lambda document: document["cohorts"]["8x"].update(rows=0),
            "positive integer row count",
        ),
        (
            lambda document: document["cohorts"]["8x"]["objects"][0].update(
                uri="https://benchmark/run-8x/part-000.parquet"
            ),
            "plain S3 object URI",
        ),
    ],
)
def test_load_cohort_rejects_bad_timed_inventory(
    tmp_path: Path, change, match: str
) -> None:
    document = _manifest()
    change(document)
    with pytest.raises(ValueError, match=match):
        worker.load_cohort(_write_manifest(tmp_path, document), "8x")


class _Dataset:
    def __init__(self) -> None:
        self.map_function = None
        self.map_kwargs = None
        self.materialized = object()

    def map_batches(self, function, **kwargs):
        self.map_function = function
        self.map_kwargs = kwargs
        return self

    def materialize(self):
        return self.materialized


class _Data:
    def __init__(self, dataset: _Dataset) -> None:
        self.dataset = dataset
        self.read_args = None
        self.pool_args = None

    def read_parquet(self, paths, *, columns):
        self.read_args = (paths, columns)
        return self.dataset

    def ActorPoolStrategy(self, **kwargs):
        self.pool_args = kwargs
        return ("pool", kwargs)


@pytest.mark.parametrize(
    ("num_gpus_per_actor", "expected_actors"),
    ((1, 8), (0.5, 16)),
)
def test_actor_geometry_consumes_all_eight_cluster_gpus(
    num_gpus_per_actor: float, expected_actors: int
) -> None:
    assert worker._actor_geometry(8, num_gpus_per_actor) == {
        "min_actors": expected_actors,
        "max_actors": expected_actors,
        "num_gpus_per_actor": num_gpus_per_actor,
        "total_reserved_gpus": 8,
    }


@pytest.mark.parametrize("num_gpus_per_actor", (True, 0.25, 2))
def test_actor_geometry_rejects_every_unapproved_fraction(
    num_gpus_per_actor: float,
) -> None:
    with pytest.raises(ValueError, match="unsupported GPUs per actor"):
        worker._actor_geometry(8, num_gpus_per_actor)


@pytest.mark.parametrize("batch_size", worker.APPROVED_BATCH_SIZES)
def test_timed_pipeline_keeps_ray_actor_defaults_omitted(
    monkeypatch: pytest.MonkeyPatch,
    batch_size: int,
) -> None:
    tokenizer = type("GPUTokenizer", (), {})
    src = ModuleType("src")
    module = ModuleType("src.ray_tokenize")
    module.GPUTokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.ray_tokenize", module)
    dataset = _Dataset()
    data = _Data(dataset)
    ray = SimpleNamespace(data=data)
    cohort = SimpleNamespace(
        paths=("s3://bucket/a.parquet",),
        columns=("a",),
    )
    ticks = iter((100, 400))

    monkeypatch.setattr(
        worker,
        "_temporary_fused_actor_concurrency",
        lambda _value: worker.contextlib.nullcontext({}),
    )
    planned, materialized, timing, resolved = worker.timed_pipeline(
        ray,
        cohort,
        nodes=8,
        batch_size=batch_size,
        clock=lambda: next(ticks),
    )

    assert planned is dataset
    assert materialized is dataset.materialized
    assert timing == {"started_ns": 100, "finished_ns": 400, "elapsed_ns": 300}
    assert resolved == {}
    assert data.read_args == ([cohort.paths[0]], ["a"])
    assert dataset.map_function is tokenizer
    assert dataset.map_kwargs == {
        "fn_constructor_kwargs": {"merchant_hash_size": 2_000},
        "batch_format": "cudf",
        "batch_size": batch_size,
        "zero_copy_batch": True,
        "compute": (
            "pool",
            {
                "size": 8,
            },
        ),
        "num_gpus": 1,
    }
    assert "max_concurrency" not in dataset.map_kwargs


@pytest.mark.parametrize("max_concurrency", worker.APPROVED_MAP_BATCHES_MAX_CONCURRENCY)
def test_timed_pipeline_passes_public_concurrency_without_actor_queue(
    monkeypatch: pytest.MonkeyPatch, max_concurrency: int
) -> None:
    tokenizer = type("GPUTokenizer", (), {})
    src = ModuleType("src")
    module = ModuleType("src.ray_tokenize")
    module.GPUTokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.ray_tokenize", module)
    dataset = _Dataset()
    data = _Data(dataset)
    ray = SimpleNamespace(data=data)
    cohort = SimpleNamespace(paths=("s3://bucket/a.parquet",), columns=("a",))
    ticks = iter((100, 400))

    planned, materialized, _, resolved = worker.timed_pipeline(
        ray,
        cohort,
        nodes=8,
        batch_size=16_777_216,
        map_batches_max_concurrency=max_concurrency,
        clock=lambda: next(ticks),
    )

    assert planned is dataset
    assert materialized is dataset.materialized
    assert resolved == {}
    assert dataset.map_kwargs["max_concurrency"] == max_concurrency
    assert data.pool_args == {"size": 8}
    assert "max_tasks_in_flight_per_actor" not in data.pool_args


def test_timed_pipeline_uses_sixteen_half_gpu_actors_without_actor_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = type("GPUTokenizer", (), {})
    src = ModuleType("src")
    module = ModuleType("src.ray_tokenize")
    module.GPUTokenizer = tokenizer
    monkeypatch.setitem(sys.modules, "src", src)
    monkeypatch.setitem(sys.modules, "src.ray_tokenize", module)
    dataset = _Dataset()
    data = _Data(dataset)
    ray = SimpleNamespace(data=data)
    cohort = SimpleNamespace(paths=("s3://bucket/a.parquet",), columns=("a",))
    observed = []

    @worker.contextlib.contextmanager
    def fractional_probe(value):
        observed.append(value)
        yield

    monkeypatch.setattr(worker, "_temporary_fractional_gpu_fusion", fractional_probe)
    worker.timed_pipeline(
        ray,
        cohort,
        nodes=8,
        batch_size=16_777_216,
        num_gpus_per_actor=0.5,
        map_batches_max_concurrency=3,
        clock=iter((100, 400)).__next__,
    )

    assert observed == [0.5]
    assert data.pool_args == {"size": 16}
    assert dataset.map_kwargs["num_gpus"] == 0.5
    assert dataset.map_kwargs["max_concurrency"] == 3
    assert "max_tasks_in_flight_per_actor" not in data.pool_args


@pytest.mark.parametrize(
    "parquet_row_group_chunk_multiplier",
    worker.APPROVED_PARQUET_ROW_GROUP_CHUNK_MULTIPLIERS,
)
def test_fairness_records_selected_batch_and_one_gpu_actor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parquet_row_group_chunk_multiplier: int,
) -> None:
    cohort = worker.load_cohort(_write_manifest(tmp_path), "8x")
    monkeypatch.setenv("CUDF_KVIKIO_REMOTE_IO", "1")
    monkeypatch.setenv("KVIKIO_NTHREADS", "32")
    monkeypatch.setenv("KVIKIO_TASK_SIZE", "8388608")

    value = worker._fairness_configuration(
        cohort,
        8,
        33_554_432,
        parquet_row_group_chunk_multiplier,
    )

    assert value["batch_size"] == 33_554_432
    assert value["actors"] == value["nodes"] == 8
    assert value["num_gpus_per_actor"] == 1
    assert value["max_tasks_in_flight_per_actor"] == "Ray default (omitted)"
    assert (
        value["parquet_row_group_chunk_multiplier"]
        == parquet_row_group_chunk_multiplier
    )
    assert value["max_concurrency"] == "Ray default (omitted)"


@pytest.mark.parametrize("max_concurrency", worker.APPROVED_MAP_BATCHES_MAX_CONCURRENCY)
def test_fairness_records_explicit_public_concurrency(
    tmp_path: Path, max_concurrency: int
) -> None:
    cohort = worker.load_cohort(_write_manifest(tmp_path), "8x")

    value = worker._fairness_configuration(
        cohort,
        8,
        16_777_216,
        map_batches_max_concurrency=max_concurrency,
    )

    assert value["max_concurrency"] == max_concurrency
    assert value["max_tasks_in_flight_per_actor"] == "Ray default (omitted)"


def test_fairness_records_fractional_actor_geometry(tmp_path: Path) -> None:
    cohort = worker.load_cohort(_write_manifest(tmp_path), "8x")

    value = worker._fairness_configuration(
        cohort,
        8,
        16_777_216,
        map_batches_max_concurrency=3,
        num_gpus_per_actor=0.5,
    )

    assert value["nodes"] == 8
    assert value["actors"] == 16
    assert value["actors_per_node"] == 2
    assert value["num_gpus_per_actor"] == 0.5
    assert value["total_reserved_gpus"] == 8
    assert value["max_concurrency"] == 3
    assert value["max_tasks_in_flight_per_actor"] == "Ray default (omitted)"


def test_data_context_changes_only_read_fusion() -> None:
    context = SimpleNamespace(
        enable_progress_bars=True,
        use_datasource_v2=False,
        enable_cudf_parquet_read_fusion=False,
        parquet_chunker_target_chunk_size=123,
    )

    worker.configure_data_context(context, fused=True)

    assert context.enable_progress_bars is False
    assert context.use_datasource_v2 is True
    assert context.enable_cudf_parquet_read_fusion is True
    assert context.parquet_chunker_target_chunk_size is None


@dataclass(frozen=True)
class _LogicalConfig:
    batch_size: int
    marker: str


@pytest.mark.parametrize("multiplier", (2, 4))
def test_chunk_multiplier_probe_changes_only_temporary_logical_config(
    multiplier: int,
) -> None:
    def original(eligible: bool) -> _LogicalConfig | None:
        return _LogicalConfig(16, "kept") if eligible else None

    fusion_module = SimpleNamespace(_logical_config_if_eligible=original)

    with worker._temporary_fused_parquet_chunk_multiplier(
        multiplier, fusion_module=fusion_module
    ):
        assert fusion_module._logical_config_if_eligible(True) == _LogicalConfig(
            16 * multiplier, "kept"
        )
        assert fusion_module._logical_config_if_eligible(False) is None
        assert fusion_module._logical_config_if_eligible is not original

    assert fusion_module._logical_config_if_eligible is original


def test_chunk_multiplier_one_does_not_patch_logical_config() -> None:
    def original() -> None:
        return None

    fusion_module = SimpleNamespace(_logical_config_if_eligible=original)

    with worker._temporary_fused_parquet_chunk_multiplier(
        1, fusion_module=fusion_module
    ):
        assert fusion_module._logical_config_if_eligible is original

    assert fusion_module._logical_config_if_eligible is original


@dataclass(frozen=True)
class _DownstreamLogical:
    ray_remote_args: dict


def test_fractional_probe_changes_only_the_temporary_eligibility_view() -> None:
    observed = []

    def original(read, downstream, context):
        observed.append((read, downstream, context))
        return downstream.ray_remote_args

    fusion_module = SimpleNamespace(_logical_config_if_eligible=original)
    downstream = _DownstreamLogical(
        ray_remote_args={"num_gpus": 0.5, "max_concurrency": 3}
    )

    with worker._temporary_fractional_gpu_fusion(0.5, fusion_module=fusion_module):
        result = fusion_module._logical_config_if_eligible(
            "read", downstream, "context"
        )

    assert result == {"num_gpus": 1, "max_concurrency": 3}
    assert observed[0][1].ray_remote_args == {
        "num_gpus": 1,
        "max_concurrency": 3,
    }
    assert downstream.ray_remote_args == {
        "num_gpus": 0.5,
        "max_concurrency": 3,
    }
    assert fusion_module._logical_config_if_eligible is original


@pytest.mark.parametrize("max_concurrency", worker.APPROVED_FUSED_ACTOR_MAX_CONCURRENCY)
def test_fused_concurrency_probe_changes_only_the_synthetic_actor(
    max_concurrency: int,
) -> None:
    captured = {}

    class ActorPool:
        def max_actor_concurrency(self):
            return max_concurrency

        def max_tasks_in_flight_per_actor(self):
            return max_concurrency * 2

    physical = SimpleNamespace(_actor_pool=ActorPool())

    class FusionRule:
        @staticmethod
        def _create_fused_operator(*args):
            captured["downstream"] = args[3]
            return physical, "logical"

    original = FusionRule._create_fused_operator
    downstream = _DownstreamLogical(ray_remote_args={"num_gpus": 1})

    with worker._temporary_fused_actor_concurrency(
        max_concurrency, fusion_rule=FusionRule
    ) as receipt:
        result = FusionRule._create_fused_operator(
            "read-physical",
            "read-logical",
            "map-physical",
            downstream,
            "input",
            "plan",
            "config",
        )

    assert result == (physical, "logical")
    assert downstream.ray_remote_args == {"num_gpus": 1}
    assert captured["downstream"].ray_remote_args == {
        "num_gpus": 1,
        "max_concurrency": max_concurrency,
    }
    assert receipt == {
        "max_concurrency": max_concurrency,
        "max_tasks_in_flight_per_actor": max_concurrency * 2,
    }
    assert FusionRule._create_fused_operator is original


class _RemoteMethod:
    def __init__(self, owner, method, ray) -> None:
        self.owner = owner
        self.method = method
        self.ray = ray

    def remote(self):
        self.ray.active_node = self.owner.node_id
        return self.method()


class _Actor:
    def __init__(self, value, node_id, ray) -> None:
        self.value = value
        self.node_id = node_id
        self.ray = ray

    def __getattr__(self, name):
        return _RemoteMethod(self, getattr(self.value, name), self.ray)


class _RemoteClass:
    def __init__(self, cls, ray, node_id=None) -> None:
        self.cls = cls
        self.ray = ray
        self.node_id = node_id

    def options(self, *, scheduling_strategy):
        return _RemoteClass(self.cls, self.ray, scheduling_strategy)

    def remote(self):
        return _Actor(self.cls(), self.node_id, self.ray)


class _FakeRay(ModuleType):
    def __init__(self) -> None:
        super().__init__("ray")
        self.active_node = None

    def remote(self, **options):
        return lambda cls: _RemoteClass(cls, self)

    @staticmethod
    def get(values):
        return values

    def get_runtime_context(self):
        return SimpleNamespace(get_node_id=lambda: self.active_node)


def test_post_timing_transport_probe_reads_cudf_option_and_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ray = _FakeRay()
    fake_cudf = ModuleType("cudf")
    fake_cudf.get_option = lambda name: name == "kvikio_remote_io"
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "cudf", fake_cudf)
    monkeypatch.delitem(sys.modules, "kvikio", raising=False)
    monkeypatch.setattr(worker, "_node_affinity", lambda node_id: node_id)
    monkeypatch.setenv("CUDF_KVIKIO_REMOTE_IO", "1")
    monkeypatch.setenv("KVIKIO_NTHREADS", "32")
    monkeypatch.setenv("KVIKIO_TASK_SIZE", str(8 * 1024 * 1024))
    alive = [{"NodeID": "node-a"}, {"NodeID": "node-b"}]

    result = worker._transport_realization(fake_ray, alive)

    assert result["valid"] is True
    assert result["expected_environment"] == {
        "CUDF_KVIKIO_REMOTE_IO": "1",
        "KVIKIO_NTHREADS": "32",
        "KVIKIO_TASK_SIZE": str(8 * 1024 * 1024),
    }
    assert [item["node_id"] for item in result["nodes"]] == ["node-a", "node-b"]
    assert all(
        item["process_environment"] == result["expected_environment"]
        and item["cudf_kvikio_remote_io"] is True
        and item["matches"] is True
        for item in result["nodes"]
    )


@pytest.mark.parametrize("batch_size", (1, 16_777_215, 33_554_433))
def test_cli_rejects_unapproved_batch_size(tmp_path: Path, batch_size: int) -> None:
    with pytest.raises(SystemExit, match="batch-size must be one of"):
        worker.main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                str(batch_size),
                "--spill-directory",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_cli_does_not_expose_ray_queue_override() -> None:
    with pytest.raises(SystemExit):
        worker.parser().parse_args(
            [
                "--manifest",
                "manifest.json",
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--max-tasks-in-flight-per-actor",
                "4",
                "--spill-directory",
                "spill",
                "--output",
                "result.json",
            ]
        )
    assert "--max-tasks-in-flight-per-actor" not in worker.parser().format_help()


@pytest.mark.parametrize("multiplier", (0, 3, 5))
def test_cli_rejects_unapproved_parquet_chunk_multiplier(
    tmp_path: Path, multiplier: int
) -> None:
    with pytest.raises(
        SystemExit, match="parquet-row-group-chunk-multiplier must be one of"
    ):
        worker.main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--parquet-row-group-chunk-multiplier",
                str(multiplier),
                "--spill-directory",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


@pytest.mark.parametrize("fused_actor_max_concurrency", (1, 2, 3))
@pytest.mark.parametrize(
    "parquet_row_group_chunk_multiplier",
    worker.APPROVED_PARQUET_ROW_GROUP_CHUNK_MULTIPLIERS,
)
def test_cli_carries_approved_settings_into_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fused_actor_max_concurrency: int,
    parquet_row_group_chunk_multiplier: int,
) -> None:
    manifest = _write_manifest(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()
    output = tmp_path / "result.json"
    observed = {}

    def fake_run(args):
        observed["batch_size"] = args.batch_size
        observed["fused_actor_max_concurrency"] = args.fused_actor_max_concurrency
        observed[
            "parquet_row_group_chunk_multiplier"
        ] = args.parquet_row_group_chunk_multiplier
        return worker._self_hash({"valid": True, "status": "accepted"})

    monkeypatch.setattr(worker, "run_observation", fake_run)

    assert (
        worker.main(
            [
                "--manifest",
                str(manifest),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "33554432",
                "--fused-actor-max-concurrency",
                str(fused_actor_max_concurrency),
                "--parquet-row-group-chunk-multiplier",
                str(parquet_row_group_chunk_multiplier),
                "--spill-directory",
                str(spill),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert observed["batch_size"] == 33_554_432
    assert observed["fused_actor_max_concurrency"] == fused_actor_max_concurrency
    assert (
        observed["parquet_row_group_chunk_multiplier"]
        == parquet_row_group_chunk_multiplier
    )
    assert worker._artifact_hash_is_valid(json.loads(output.read_text()))


def test_failure_artifact_records_parquet_chunk_multiplier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(tmp_path)
    spill = tmp_path / "spill"
    spill.mkdir()
    output = tmp_path / "result.json"

    def fail(_args):
        raise RuntimeError("planned failure")

    monkeypatch.setattr(worker, "run_observation", fail)

    assert (
        worker.main(
            [
                "--manifest",
                str(manifest),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--parquet-row-group-chunk-multiplier",
                "4",
                "--spill-directory",
                str(spill),
                "--output",
                str(output),
            ]
        )
        == 1
    )
    result = json.loads(output.read_text())
    assert result["status"] == "failed"
    assert result["parquet_row_group_chunk_multiplier"] == 4
    assert worker._artifact_hash_is_valid(result)


def test_cli_defaults_fused_probe_and_parquet_chunk_multiplier_to_one() -> None:
    args = worker.parser().parse_args(
        [
            "--manifest",
            "manifest.json",
            "--cohort",
            "8x",
            "--mode",
            "fused",
            "--nodes",
            "8",
            "--batch-size",
            "16777216",
            "--spill-directory",
            "spill",
            "--output",
            "result.json",
        ]
    )

    assert args.map_batches_max_concurrency is None
    assert args.fused_actor_max_concurrency is None
    assert args.num_gpus_per_actor == 1
    assert args.parquet_row_group_chunk_multiplier == 1
    assert args.critical_profile is False


@pytest.mark.parametrize("max_concurrency", worker.APPROVED_MAP_BATCHES_MAX_CONCURRENCY)
def test_cli_carries_public_map_batches_concurrency(
    tmp_path: Path, max_concurrency: int
) -> None:
    args = worker.parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--cohort",
            "8x",
            "--mode",
            "fused",
            "--nodes",
            "8",
            "--batch-size",
            "16777216",
            "--map-batches-max-concurrency",
            str(max_concurrency),
            "--spill-directory",
            str(tmp_path),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.map_batches_max_concurrency == max_concurrency
    assert args.fused_actor_max_concurrency is None


@pytest.mark.parametrize("mode", ("isolated", "fused"))
def test_cli_applies_the_same_explicit_fractional_geometry_to_both_arms(
    tmp_path: Path, mode: str
) -> None:
    args = worker.parser().parse_args(
        [
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--cohort",
            "8x",
            "--mode",
            mode,
            "--nodes",
            "8",
            "--batch-size",
            "16777216",
            "--num-gpus-per-actor",
            "0.5",
            "--map-batches-max-concurrency",
            "3",
            "--spill-directory",
            str(tmp_path),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.num_gpus_per_actor == 0.5
    assert args.map_batches_max_concurrency == 3


def test_cli_rejects_unapproved_fractional_geometry(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        worker.parser().parse_args(
            [
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--num-gpus-per-actor",
                "0.25",
                "--spill-directory",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_cli_requires_public_concurrency_for_fractional_geometry(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="require explicit"):
        worker.main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--num-gpus-per-actor",
                "0.5",
                "--spill-directory",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_public_and_physical_concurrency_are_mutually_exclusive(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit, match="mutually exclusive"):
        worker.main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--cohort",
                "8x",
                "--mode",
                "fused",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--map-batches-max-concurrency",
                "2",
                "--fused-actor-max-concurrency",
                "2",
                "--spill-directory",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_cli_exposes_only_explicit_critical_profile_surface() -> None:
    help_text = worker.parser().format_help()
    assert "--critical-profile" in help_text
    assert "smoke" not in help_text


def test_critical_profile_requires_fused_mode(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="requires --mode fused"):
        worker.main(
            [
                "--manifest",
                str(tmp_path / "missing.json"),
                "--cohort",
                "8x",
                "--mode",
                "isolated",
                "--nodes",
                "8",
                "--batch-size",
                "16777216",
                "--critical-profile",
                "--spill-directory",
                str(tmp_path),
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_critical_profile_runtime_env_ships_lazy_setup_hook() -> None:
    support = Path(__file__).with_name("fusion_critical_profile_support.py")
    registered = []
    fake_ray = SimpleNamespace(
        cloudpickle=SimpleNamespace(register_pickle_by_value=registered.append)
    )

    runtime_env = worker._critical_profile_runtime_env(
        fake_ray,
        "run-123",
        support_path=support,
    )

    assert len(registered) == 1
    module = registered[0]
    assert runtime_env == {
        "env_vars": {
            module.PROFILE_ENV: "1",
            module.LAZY_PROFILE_ENV: "1",
            module.RUN_ID_ENV: "run-123",
        },
        "worker_process_setup_hook": module.install_profile_hooks,
    }


def test_profile_log_decode_filters_run_and_reports_bad_json() -> None:
    marker = worker.CRITICAL_PROFILE_MARKER
    matching = {
        "event": "actor_ready",
        "profile_run_id": "selected",
        "node_id": "node-b",
        "task_idx": 1,
    }
    earlier = {
        "event": "actor_ready",
        "profile_run_id": "selected",
        "node_id": "node-a",
        "task_idx": 0,
    }
    other = {"event": "actor_ready", "profile_run_id": "other"}

    events, errors = worker._decode_critical_profile_lines(
        [
            "ordinary worker output",
            marker + json.dumps(matching),
            marker + "{broken",
            marker + json.dumps(other),
            marker + json.dumps(earlier),
        ],
        "selected",
    )

    assert events == [earlier, matching]
    assert len(errors) == 1
    assert errors[0].startswith("line 3:")


def test_critical_profile_summary_keeps_required_phases_separate() -> None:
    stages = {
        "cudf_read": {"calls": 2, "elapsed_s": 4.0, "rows": 20, "bytes": 0},
        "udf": {"calls": 2, "elapsed_s": 2.0, "rows": 20, "bytes": 200},
        "batch_to_block": {
            "calls": 2,
            "elapsed_s": 1.0,
            "rows": 20,
            "bytes": 200,
        },
        "output_buffer_add_batch": {
            "calls": 2,
            "elapsed_s": 1.5,
            "rows": 20,
            "bytes": 200,
        },
        "output_buffer_next": {
            "calls": 2,
            "elapsed_s": 0.5,
            "rows": 20,
            "bytes": 200,
        },
        "block_yield_suspended": {
            "calls": 2,
            "elapsed_s": 3.0,
            "rows": 0,
            "bytes": 0,
        },
        "metadata_yield_suspended": {
            "calls": 2,
            "elapsed_s": 0.2,
            "rows": 0,
            "bytes": 0,
        },
    }
    events = [
        {
            "event": "actor_ready",
            "node_id": node,
            "success": True,
            "init_elapsed_s": 0.5,
        }
        for node in ("node-a", "node-b")
    ]
    events.extend(
        {
            "event": "tokenizer_initialized",
            "node_id": node,
            "init_elapsed_s": 0.25,
        }
        for node in ("node-a", "node-b")
    )
    events.append(
        {
            "event": "task_complete",
            "node_id": "node-a",
            "success": True,
            "task_elapsed_s": 6.0,
            "stages": stages,
        }
    )

    summary = worker._critical_profile_summary(
        events,
        expected_actor_count=2,
        expected_node_ids={"node-a", "node-b"},
    )

    assert summary["valid"] is True
    assert summary["phases"]["cudf_s3_read_decode"]["actor_time_s"] == 4.0
    assert summary["phases"]["gpu_tokenizer"]["actor_time_s"] == 2.0
    assert summary["phases"]["batch_to_block"]["actor_time_s"] == 1.0
    assert summary["phases"]["output_buffer"]["actor_time_s"] == 2.0
    assert summary["phases"]["yield_backpressure"]["actor_time_s"] == 3.2
