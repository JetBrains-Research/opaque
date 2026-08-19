"""Shared pytest configuration and fixtures for all Opaque packages.

Pytest discovers conftest.py by walking up from each test file, so every
package in the workspace inherits these fixtures automatically.
"""

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Auto-skip logic for marker-based gating
# ---------------------------------------------------------------------------


def pytest_runtest_setup(item):
    """Auto-skip tests based on the three orthogonal markers (cuda/mps/slow)."""
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if "mps" in item.keywords and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def get_default_device():
    """Get the default device for testing (CUDA > MPS > CPU)."""
    requested = os.environ.get("OPAQUE_TEST_DEVICE", "").strip().lower()
    if requested:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError(
                "OPAQUE_TEST_DEVICE=cuda requested but CUDA is unavailable"
            )
        if requested == "mps":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError(
                "OPAQUE_TEST_DEVICE=mps requested but MPS is unavailable"
            )
        raise RuntimeError(
            f"Invalid OPAQUE_TEST_DEVICE={requested!r}. Expected one of: cpu, cuda, mps"
        )

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_default_gpu_device():
    """Get the default GPU device (CUDA > MPS), or None."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _activate_torch_backend(request):
    """Select Torch explicitly for legacy Torch-facing package tests."""
    path = str(request.node.path).replace(os.sep, "/")
    neutral_test_roots = (
        "packages/opaque-accounting/tests/",
        "packages/opaque-base/tests/",
        "tests/contracts/",
        "tests/integration/backend/",
    )
    if any(root in path for root in neutral_test_roots):
        yield
        return

    from opaque.api.engine.backend import _registry, clear_backend, ensure_backend

    clear_backend()
    ensure_backend(torch.empty(0))
    # Forget throwaway backends registered by earlier tests so the
    # single-backend dispatch fast path stays representative.
    _registry._reset_loaded_backends()
    yield
    clear_backend()


@pytest.fixture
def device():
    """Provide device for tests (CUDA > MPS > CPU in priority order)."""
    return get_default_device()


@pytest.fixture(params=["cpu", "cuda", "mps"])
def all_devices(request):
    """Parametrize tests over all available devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device(request.param)


@pytest.fixture
def set_random_seed():
    """Fixture to set random seed for reproducibility."""

    def _set_seed(seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    return _set_seed


@pytest.fixture
def simple_pytree():
    """Provide a simple PyTree for testing."""
    return {
        "weight": torch.tensor([3.0, 4.0]),
        "bias": torch.tensor([0.0, 12.0]),
    }


@pytest.fixture
def nested_pytree():
    """Provide a nested PyTree for testing."""
    return {
        "layer1": {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, -0.5]),
        },
        "layer2": {"weight": torch.tensor([1.0]), "bias": torch.tensor([0.0])},
    }
