"""Pytest configuration and shared fixtures for Opaque tests."""

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "jax_validation: marks tests that require JAX and JAX-Privacy for cross-framework validation",
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")


@pytest.fixture
def device():
    """Provide device for tests (CUDA if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(params=["cpu", "cuda"])
def all_devices(request):
    """Parametrize tests over all available devices."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(request.param)


@pytest.fixture
def set_random_seed():
    """Fixture to set random seed for reproducibility."""

    def _set_seed(seed: int = 42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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
