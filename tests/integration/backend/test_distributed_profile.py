"""Cross-provider conformance for the eager distributed runtime profile."""

from __future__ import annotations

from typing import Any

import pytest
from tests.integration.backend._providers import provider_case

from opaque.api.engine import runtime
from opaque.api.engine.backend import clear_backend, use_backend
from opaque.distributed import (
    all_reduce,
    barrier,
    gather_for_metrics,
    get_rank,
    get_world_size,
    is_distributed,
)


def _tolist(value: Any) -> list[float]:
    return value.tolist()


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_complete_distributed_profile_and_public_singleton_helpers(
    provider_name: str,
) -> None:
    case = provider_case(provider_name)

    assert runtime.RuntimeProfile.DISTRIBUTED.supports(case.backend)
    assert all(
        primitive.supports(case.backend)
        for primitive in runtime.RuntimeProfile.DISTRIBUTED.primitives
    )

    with use_backend(case.backend):
        assert get_rank() == 0
        assert get_world_size() == 1
        assert not is_distributed()
        for operation in runtime.ReduceOp:
            reduced = all_reduce(case.value, operation.value)
            assert isinstance(reduced, case.array_type)
            assert reduced is not case.value
            assert _tolist(reduced) == [1.0, 2.0]

        gathered = gather_for_metrics(case.value)
        objects = runtime.distributed_all_gather_object({"provider": provider_name})
        assert barrier() is None

    assert isinstance(gathered, case.array_type)
    assert gathered is not case.value
    assert _tolist(gathered) == [1.0, 2.0]
    assert objects == [{"provider": provider_name}]
