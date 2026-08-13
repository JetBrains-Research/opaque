"""Provider cases shared by cross-framework conformance tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class ProviderCase:
    """Native construction and observation adapters for one provider."""

    name: str
    backend: Any
    value: Any
    array_type: type
    array: Callable[[Any, Any | None], Any]
    dtypes: Mapping[str, Any]
    evaluate: Callable[[Any], Any]
    to_numpy: Callable[[Any], np.ndarray]

    def dtype(self, name: str) -> Any:
        """Return a provider-native dtype by its portable test name."""
        return self.dtypes[name]


def provider_case(name: str) -> ProviderCase:
    """Construct a singleton-safe case for a first-party provider."""
    if name == "torch":
        torch = pytest.importorskip("torch")
        from opaque.torch import torch_backend

        def torch_array(value: Any, dtype: Any | None = None) -> Any:
            return torch.tensor(value, dtype=dtype)

        def torch_to_numpy(value: Any) -> np.ndarray:
            return value.detach().cpu().numpy()

        return ProviderCase(
            name=name,
            backend=torch_backend(),
            value=torch.tensor([1.0, 2.0]),
            array_type=torch.Tensor,
            array=torch_array,
            dtypes={
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
                "int64": torch.int64,
            },
            evaluate=lambda value: value,
            to_numpy=torch_to_numpy,
        )
    if name == "jax":
        jax = pytest.importorskip("jax")
        jnp = pytest.importorskip("jax.numpy")
        from opaque.jax import jax_backend

        def jax_array(value: Any, dtype: Any | None = None) -> Any:
            return jnp.asarray(value, dtype=dtype)

        return ProviderCase(
            name=name,
            backend=jax_backend(),
            value=jnp.array([1.0, 2.0]),
            array_type=jax.Array,
            array=jax_array,
            dtypes={
                "float16": jnp.float16,
                "bfloat16": jnp.bfloat16,
                "float32": jnp.float32,
                "int64": jnp.int64,
            },
            evaluate=jax.block_until_ready,
            to_numpy=np.asarray,
        )
    mx = pytest.importorskip("mlx.core")
    from opaque.mlx import mlx_backend

    def mlx_array(value: Any, dtype: Any | None = None) -> Any:
        return mx.array(value, dtype=dtype)

    def mlx_to_numpy(value: Any) -> np.ndarray:
        mx.eval(value)
        return np.asarray(value)

    return ProviderCase(
        name=name,
        backend=mlx_backend(),
        value=mx.array([1.0, 2.0]),
        array_type=mx.array,
        array=mlx_array,
        dtypes={
            "float16": mx.float16,
            "bfloat16": mx.bfloat16,
            "float32": mx.float32,
            "int64": mx.int64,
        },
        evaluate=mx.eval,
        to_numpy=mlx_to_numpy,
    )


__all__ = ["ProviderCase", "provider_case"]
