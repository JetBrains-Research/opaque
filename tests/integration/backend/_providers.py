"""Provider cases shared by cross-framework conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class ProviderCase:
    """Native value and backend identity for one first-party provider."""

    name: str
    backend: Any
    value: Any
    array_type: type


def provider_case(name: str) -> ProviderCase:
    """Construct a singleton-safe case for a first-party provider."""
    if name == "torch":
        torch = pytest.importorskip("torch")
        from opaque.torch import torch_backend

        return ProviderCase(
            name=name,
            backend=torch_backend(),
            value=torch.tensor([1.0, 2.0]),
            array_type=torch.Tensor,
        )
    if name == "jax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from opaque.jax import jax_backend

        return ProviderCase(
            name=name,
            backend=jax_backend(),
            value=jnp.array([1.0, 2.0]),
            array_type=jax.Array,
        )
    mx = pytest.importorskip("mlx.core")
    from opaque.mlx import mlx_backend

    return ProviderCase(
        name=name,
        backend=mlx_backend(),
        value=mx.array([1.0, 2.0]),
        array_type=mx.array,
    )


__all__ = ["ProviderCase", "provider_case"]
