"""MLX device capability helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class DeviceCapabilities:
    """Capabilities MLX exposes through its unified-memory runtime."""

    supports_bf16: bool
    supports_compile: bool
    recommended_compile_backend: str | None
    supports_fused_kernels: bool
    peak_memory_trackable: bool
    supports_pin_memory: bool

    @property
    def is_accelerator(self) -> bool:
        return True


def device_capabilities(device: Any = None) -> DeviceCapabilities:
    """Resolve MLX capabilities for its unified-memory device."""
    del device
    return DeviceCapabilities(
        supports_bf16=True,
        supports_compile=hasattr(mx, "compile"),
        recommended_compile_backend=None,
        supports_fused_kernels=False,
        peak_memory_trackable=hasattr(mx, "get_peak_memory"),
        supports_pin_memory=False,
    )


__all__ = ["DeviceCapabilities", "device_capabilities"]
