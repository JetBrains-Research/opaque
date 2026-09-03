"""Portable backend test matrix and native-value adapters."""

from __future__ import annotations

import importlib
import os
import platform
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from types import ModuleType

EXPECTED_PROVIDERS_ENV = "OPAQUE_EXPECTED_PROVIDERS"


class BackendMatrixConfigurationError(RuntimeError):
    """Raised when a required provider cannot participate in portable tests."""


@dataclass(frozen=True)
class _ProviderSpec:
    runtime_module: str
    provider_module: str


_PROVIDER_SPECS = {
    "torch": _ProviderSpec("torch", "opaque.api.torch"),
    "mlx": _ProviderSpec("mlx.core", "opaque.api.mlx"),
}


@dataclass(frozen=True)
class BackendCase:
    """Native construction and observation helpers for one production backend."""

    name: str
    runtime: ModuleType

    @property
    def id(self) -> str:
        """Return the stable pytest parameter identifier for this backend."""
        return self.name

    def dtype(self, name: str) -> Any:
        """Return a backend-native dtype by its shared name."""
        try:
            return getattr(self.runtime, name)
        except AttributeError as exc:
            raise ValueError(f"{self.name} has no dtype {name!r}") from exc

    def array(self, value: Any, *, dtype: Any | None = None) -> Any:
        """Construct a native array without embedding algorithm behavior."""
        if self.name == "torch":
            return self.runtime.as_tensor(value, dtype=dtype)
        return self.runtime.array(value, dtype=dtype)

    def to_host(self, value: Any) -> np.ndarray:
        """Copy a native value to a NumPy array for portable assertions."""
        if self.name == "torch":
            return value.detach().cpu().numpy().copy()
        return np.array(value, copy=True)

    def assert_allclose(
        self,
        actual: Any,
        expected: Any,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Compare a native value with a host value using shared tolerances."""
        np.testing.assert_allclose(self.to_host(actual), expected, rtol=rtol, atol=atol)


def expected_provider_names(
    environ: Mapping[str, str] | None = None,
    *,
    system: str | None = None,
    machine: str | None = None,
) -> tuple[str, ...]:
    """Resolve the production providers required for this test invocation."""
    environ = os.environ if environ is None else environ
    configured = environ.get(EXPECTED_PROVIDERS_ENV, "").strip()
    if configured:
        names = tuple(name.strip() for name in configured.split(",") if name.strip())
    else:
        system = platform.system() if system is None else system
        machine = platform.machine() if machine is None else machine
        names = (
            ("torch", "mlx") if (system, machine) == ("Darwin", "arm64") else ("torch",)
        )

    unknown = sorted(set(names).difference(_PROVIDER_SPECS))
    if unknown:
        known = ", ".join(sorted(_PROVIDER_SPECS))
        raise BackendMatrixConfigurationError(
            f"{EXPECTED_PROVIDERS_ENV} contains unknown provider(s): "
            f"{', '.join(unknown)}. Expected one of: {known}."
        )
    if len(set(names)) != len(names):
        raise BackendMatrixConfigurationError(
            f"{EXPECTED_PROVIDERS_ENV} must not contain duplicate providers."
        )
    return names


def load_backend_case(
    name: str, *, importer: Callable[[str], ModuleType] = importlib.import_module
) -> BackendCase:
    """Import one required runtime and its Opaque provider package."""
    try:
        spec = _PROVIDER_SPECS[name]
    except KeyError as exc:
        raise BackendMatrixConfigurationError(
            f"Unknown backend provider {name!r}."
        ) from exc

    try:
        runtime = importer(spec.runtime_module)
        importer(spec.provider_module)
    except ImportError as exc:
        raise BackendMatrixConfigurationError(
            f"Expected provider {name!r} is unavailable. "
            f"Install its runtime and {spec.provider_module!r}."
        ) from exc
    return BackendCase(name=name, runtime=runtime)


def load_expected_backend_cases(
    environ: Mapping[str, str] | None = None,
) -> tuple[BackendCase, ...]:
    """Load every provider declared as required for the invocation."""
    return tuple(load_backend_case(name) for name in expected_provider_names(environ))


@contextmanager
def activate_backend_case(case: BackendCase) -> Iterator[BackendCase]:
    """Select one provider and unconditionally restore a clean registry."""
    from opaque.api.engine.backend import _registry, clear_backend, ensure_backend

    clear_backend()
    _registry._reset_loaded_backends()
    try:
        backend = ensure_backend(case.array(0.0))
        if backend.name != case.name:
            raise BackendMatrixConfigurationError(
                f"Expected provider {case.name!r}, but activated {backend.name!r}."
            )
        yield case
    finally:
        clear_backend()
        _registry._reset_loaded_backends()


def pytest_configure(config: pytest.Config) -> None:
    """Require every configured provider before tests are collected."""
    config.addinivalue_line(
        "markers", "provider(name): run a provider-local conformance case"
    )
    try:
        config._opaque_backend_cases = load_expected_backend_cases()
    except BackendMatrixConfigurationError as exc:
        raise pytest.UsageError(str(exc)) from exc


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parameterize portable tests over configured production providers."""
    fixture_name = next(
        (
            name
            for name in ("backend_case", "provider_case")
            if name in metafunc.fixturenames
        ),
        None,
    )
    if fixture_name is None:
        return
    cases: tuple[BackendCase, ...] = metafunc.config._opaque_backend_cases
    if fixture_name == "provider_case":
        marker = metafunc.definition.get_closest_marker("provider")
        if (
            marker is None
            or len(marker.args) != 1
            or not isinstance(marker.args[0], str)
        ):
            raise pytest.UsageError(
                "provider_case requires one @pytest.mark.provider(<name>) marker."
            )
        cases = tuple(case for case in cases if case.name == marker.args[0])
        if not cases:
            raise pytest.UsageError(
                f"Provider-local test requires {marker.args[0]!r}, but it is not expected."
            )
    metafunc.parametrize(fixture_name, cases, ids=lambda case: case.id, indirect=True)


@pytest.fixture
def backend_case(request: pytest.FixtureRequest) -> Iterator[BackendCase]:
    """Activate one configured provider for a portable owner-package test."""
    case = request.param
    if not isinstance(case, BackendCase):
        raise TypeError("backend_case must be parameterized by the portable matrix")
    with activate_backend_case(case):
        yield case


@pytest.fixture
def provider_case(request: pytest.FixtureRequest) -> Iterator[BackendCase]:
    """Activate the provider named by a provider-local contract test."""
    case = request.param
    if not isinstance(case, BackendCase):
        raise TypeError("provider_case must be parameterized by the portable matrix")
    with activate_backend_case(case):
        yield case
