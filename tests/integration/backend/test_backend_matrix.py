"""Portable backend-matrix configuration and fixture behavior."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from opaque_engine_testkit import matrix as backend_matrix
from opaque_engine_testkit.contract import BackendContractTests
from opaque_engine_testkit.validation import (
    BackendContractValidationError,
    contract_test_methods,
    semantic_exception,
    validate_backend_contract,
)

from opaque.api.engine.backend import active_backend

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_default_provider_matrix_is_platform_gated() -> None:
    assert backend_matrix.expected_provider_names(
        {}, system="Linux", machine="x86_64"
    ) == ("torch",)
    assert backend_matrix.expected_provider_names(
        {}, system="Darwin", machine="arm64"
    ) == ("torch", "mlx")


def test_explicit_provider_matrix_preserves_parameter_ids() -> None:
    names = backend_matrix.expected_provider_names(
        {backend_matrix.EXPECTED_PROVIDERS_ENV: "mlx, torch"}
    )

    assert names == ("mlx", "torch")
    assert names == tuple(
        case.id
        for case in (
            backend_matrix.BackendCase(name, ModuleType(name)) for name in names
        )
    )


def test_missing_expected_provider_is_a_configuration_error() -> None:
    def missing_import(_module: str):
        raise ModuleNotFoundError("provider is not installed")

    with pytest.raises(backend_matrix.BackendMatrixConfigurationError, match="torch"):
        backend_matrix.load_backend_case("torch", importer=missing_import)


def test_pytest_configuration_fails_for_a_missing_expected_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        def addinivalue_line(self, _name: str, _value: str) -> None:
            pass

    def missing_provider():
        raise backend_matrix.BackendMatrixConfigurationError("missing provider")

    monkeypatch.setattr(backend_matrix, "load_expected_backend_cases", missing_provider)

    with pytest.raises(pytest.UsageError, match="missing provider"):
        backend_matrix.pytest_configure(Config())


def test_backend_case_activates_and_cleans_up_registry() -> None:
    case = backend_matrix.load_backend_case("torch")

    with backend_matrix.activate_backend_case(case):
        assert active_backend() is not None
        assert active_backend().name == "torch"

    assert active_backend() is None


def test_backend_case_fixture_uses_stable_parameter_id(backend_case) -> None:
    assert active_backend() is not None
    assert active_backend().name == backend_case.id


def test_backend_case_adapter_constructs_and_observes_native_values(
    backend_case,
) -> None:
    value = backend_case.array([1.0, 2.0], dtype=backend_case.dtype("float32"))

    assert backend_case.to_host(value).dtype.name == "float32"
    backend_case.assert_allclose(value, [1.0, 2.0])


class _CompleteContract(BackendContractTests):
    provider_name = "torch"


def test_inherited_contract_maps_every_core_primitive() -> None:
    validate_backend_contract(_CompleteContract)
    inherited = contract_test_methods(_CompleteContract)
    assert set(inherited) == set(contract_test_methods(BackendContractTests))
    assert len(inherited) > 1


@pytest.mark.parametrize(
    ("provider", "relative_path", "class_name"),
    [
        (
            "torch",
            "packages/opaque-torch/tests/backend/test_portable_contract.py",
            "TestTorchBackendContract",
        ),
        (
            "mlx",
            "packages/opaque-mlx/tests/backend/test_portable_contract.py",
            "TestMLXBackendContract",
        ),
    ],
)
def test_provider_contract_subclasses_preserve_the_inherited_core(
    provider: str, relative_path: str, class_name: str
) -> None:
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(class_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = getattr(module, class_name)

    assert contract.__test__ is True
    assert contract.provider_name == provider
    validate_backend_contract(contract)
    assert set(contract_test_methods(contract)) == set(
        contract_test_methods(BackendContractTests)
    )


def test_contract_rejects_an_unannotated_override() -> None:
    class UnannotatedOverride(_CompleteContract):
        def test_keyed_randomness_is_deterministic(self, provider_case):
            pass

    with pytest.raises(
        BackendContractValidationError, match="without @semantic_exception"
    ):
        validate_backend_contract(UnannotatedOverride)


def test_contract_rejects_a_removed_inherited_test() -> None:
    class RemovedTest(_CompleteContract):
        test_keyed_randomness_is_deterministic = None

    with pytest.raises(
        BackendContractValidationError, match="without @semantic_exception"
    ):
        validate_backend_contract(RemovedTest)


def test_contract_rejects_an_override_that_removes_primitive_coverage() -> None:
    class WeakOverride(_CompleteContract):
        @semantic_exception("native random source differs", covers=())
        def test_keyed_randomness_is_deterministic(self, provider_case):
            pass

    with pytest.raises(
        BackendContractValidationError, match="weakens inherited coverage"
    ):
        validate_backend_contract(WeakOverride)


def test_contract_rejects_a_stale_core_profile_version() -> None:
    class StaleProfile(_CompleteContract):
        core_profile_version = -1

    with pytest.raises(BackendContractValidationError, match="declares core profile"):
        validate_backend_contract(StaleProfile)
