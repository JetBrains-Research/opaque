"""Torch execution of the inherited portable core contract."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from opaque_engine_testkit.contract import BackendContractTests

if TYPE_CHECKING:
    from opaque_engine_testkit.matrix import BackendCase

pytestmark = pytest.mark.provider("torch")


class TestTorchBackendContract(BackendContractTests):
    """Torch's production implementation of the engine portable core."""

    __test__ = True
    provider_name = "torch"

    def array(self, case: BackendCase, value: Any, *, dtype: Any | None = None) -> Any:
        return case.array(value, dtype=dtype)

    def dtype(self, case: BackendCase, name: str) -> Any:
        return case.dtype(name)

    def to_host(self, case: BackendCase, value: Any):
        return case.to_host(value)

    def capabilities(self) -> dict[str, bool]:
        return {"unified_memory": False}
