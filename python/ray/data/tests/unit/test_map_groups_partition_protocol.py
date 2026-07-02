import asyncio

import ray.cloudpickle as ray_pickle

from ray.data._internal.planner.map_groups_partition_protocol import (
    MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG,
    MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE,
    MAP_GROUPS_PARTITION_UDF_ATTRIBUTE,
    MapGroupsPartitionContext,
    resolve_map_groups_partition_contract,
)
from ray.data.context import DataContext


def _partition(batch, context):
    return batch


def _declare(fn, partition_fn=_partition, **overrides):
    descriptor = {
        "version": 1,
        "batch_format": "cudf",
        "side_effect_free": True,
        "equivalent_to_per_group": True,
    }
    descriptor.update(overrides)
    setattr(fn, MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE, descriptor)
    setattr(fn, MAP_GROUPS_PARTITION_UDF_ATTRIBUTE, partition_fn)
    return fn


def _context(enabled=True):
    context = DataContext()
    if enabled:
        context.set_config(MAP_GROUPS_PARTITION_EXECUTION_ENABLED_CONFIG, True)
    return context


def test_partition_protocol_is_flagged_off_and_optional():
    fn = _declare(lambda batch: batch)
    contract, reason = resolve_map_groups_partition_contract(
        fn, _context(False), batch_format="cudf"
    )
    assert contract is None
    assert reason is None

    contract, reason = resolve_map_groups_partition_contract(
        lambda batch: batch, _context(), batch_format="cudf"
    )
    assert contract is None
    assert reason is None


def test_partition_protocol_resolves_and_serializes():
    fn = _declare(lambda batch: batch)
    contract, reason = resolve_map_groups_partition_contract(
        fn, _context(), batch_format="cudf"
    )
    assert reason is None
    assert contract is not None
    assert contract.udf is _partition
    assert contract.batch_format == "cudf"
    restored = ray_pickle.loads(ray_pickle.dumps(contract))
    assert restored.batch_format == "cudf"
    assert restored.udf("batch", object()) == "batch"


def test_partition_context_reports_immutable_group_layout():
    context = MapGroupsPartitionContext(
        group_keys=("User", "Card"),
        input_group_boundaries=(0, 2, 5),
    )
    assert context.num_groups == 2
    assert context.group_keys == ("User", "Card")


def test_partition_protocol_rejects_malformed_descriptors():
    cases = [
        ({"version": 1}, "group_partition_protocol_invalid"),
        (
            {
                "version": 2,
                "batch_format": "cudf",
                "side_effect_free": True,
                "equivalent_to_per_group": True,
            },
            "group_partition_protocol_version_unsupported",
        ),
        (
            {
                "version": 1,
                "batch_format": "pandas",
                "side_effect_free": True,
                "equivalent_to_per_group": True,
            },
            "group_partition_batch_format_mismatch",
        ),
        (
            {
                "version": 1,
                "batch_format": "cudf",
                "side_effect_free": False,
                "equivalent_to_per_group": True,
            },
            "group_partition_side_effect_contract_missing",
        ),
        (
            {
                "version": 1,
                "batch_format": "cudf",
                "side_effect_free": True,
                "equivalent_to_per_group": False,
            },
            "group_partition_equivalence_contract_missing",
        ),
    ]
    for descriptor, expected_reason in cases:

        def fn(batch):
            return batch

        setattr(fn, MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE, descriptor)
        setattr(fn, MAP_GROUPS_PARTITION_UDF_ATTRIBUTE, _partition)
        contract, reason = resolve_map_groups_partition_contract(
            fn, _context(), batch_format="cudf"
        )
        assert contract is None
        assert reason == expected_reason


def test_partition_protocol_rejects_async_and_nonfunction_udfs():
    async def async_partition(batch, context):
        await asyncio.sleep(0)
        return batch

    for partition_fn in (async_partition, object()):
        fn = _declare(lambda batch: batch, partition_fn)
        contract, reason = resolve_map_groups_partition_contract(
            fn, _context(), batch_format="cudf"
        )
        assert contract is None
        assert reason == "group_partition_udf_invalid"

    class CallableGroup:
        __ray_data_map_groups_partition_protocol__ = {
            "version": 1,
            "batch_format": "cudf",
            "side_effect_free": True,
            "equivalent_to_per_group": True,
        }
        __ray_data_map_groups_partition__ = _partition

        def __call__(self, batch):
            return batch

    contract, reason = resolve_map_groups_partition_contract(
        CallableGroup, _context(), batch_format="cudf"
    )
    assert contract is None
    assert reason == "group_partition_callable_unsupported"


def test_partition_protocol_does_not_inspect_nonfunction_callable_attributes():
    class AttributeRaisingCallable:
        def __getattribute__(self, name):
            if name in {
                MAP_GROUPS_PARTITION_PROTOCOL_ATTRIBUTE,
                MAP_GROUPS_PARTITION_UDF_ATTRIBUTE,
            }:
                raise AssertionError("application attribute access during planning")
            return super().__getattribute__(name)

        def __call__(self, batch):
            return batch

    contract, reason = resolve_map_groups_partition_contract(
        AttributeRaisingCallable(), _context(), batch_format="cudf"
    )
    assert contract is None
    assert reason == "group_partition_callable_unsupported"


def test_partition_protocol_validates_signature_and_zero_copy_before_execution():
    def incompatible(batch):
        return batch

    fn = _declare(lambda batch: batch, incompatible)
    contract, reason = resolve_map_groups_partition_contract(
        fn,
        _context(),
        batch_format="cudf",
        fn_args=(1,),
        fn_kwargs={"offset": 2},
    )
    assert contract is None
    assert reason == "group_partition_udf_signature_incompatible"

    fn = _declare(lambda batch: batch)
    contract, reason = resolve_map_groups_partition_contract(
        fn,
        _context(),
        batch_format="cudf",
        zero_copy_batch=False,
    )
    assert contract is None
    assert reason == "group_partition_zero_copy_required"
