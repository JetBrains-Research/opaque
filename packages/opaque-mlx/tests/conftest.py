"""MLX provider test configuration."""

from __future__ import annotations

import pytest

from opaque.backend import clear_backend, set_backend


@pytest.fixture(autouse=True)
def _activate_mlx_backend():
    clear_backend()
    set_backend("mlx")
    yield
    clear_backend()
