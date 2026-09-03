"""MLX unified-memory device behavior."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from opaque import ops
from opaque.mlx.device import DeviceCapabilities, device_capabilities


def test_mlx_capabilities_describe_the_metal_unified_memory_runtime() -> None:
    capabilities = device_capabilities()

    assert isinstance(capabilities, DeviceCapabilities)
    assert capabilities.is_accelerator
    assert capabilities.supports_bf16
    assert capabilities.supports_compile is hasattr(mx, "compile")
    assert capabilities.recommended_compile_backend is None
    assert not capabilities.supports_fused_kernels
    assert not capabilities.supports_pin_memory


def test_mlx_transfer_accepts_only_unified_memory_cpu_placement() -> None:
    value = mx.array([1.0, 2.0], dtype=mx.float32)

    transferred = ops.transfer(value, "cpu")

    assert transferred.dtype == mx.float32
    np.testing.assert_array_equal(np.array(transferred), [1.0, 2.0])
    with pytest.raises(TypeError, match="unified-memory 'cpu' device"):
        ops.transfer(value, "cuda")
