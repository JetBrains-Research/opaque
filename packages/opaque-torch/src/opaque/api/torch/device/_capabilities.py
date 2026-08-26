"""Empirical PyTorch device capability probes."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch


@functools.cache
def _triton_importable() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def _probe_bf16(device_type: str) -> bool:
    try:
        if device_type == "cuda":
            return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if device_type == "mps":
            if not torch.backends.mps.is_available():
                return False
            probe = torch.ones(2, dtype=torch.bfloat16, device="mps")
            return (probe + probe).sum().item() == 4.0  # noqa: PLR2004 - bf16 probe
        if device_type == "cpu":
            return True
    except Exception:
        return False
    return False


def fused_kernels_available() -> bool:
    """Return whether CUDA and Triton are available on this host."""
    return torch.cuda.is_available() and _triton_importable()


@functools.cache
def sdpa_autocast_under_vmap_broken(device_type: str) -> bool:
    """Return whether the live PyTorch has the MPS SDPA autocast bug."""
    if device_type != "mps" or not torch.backends.mps.is_available():
        return False

    def loss(scale: Any, query: Any, key: Any, value: Any) -> Any:
        output = torch.nn.functional.scaled_dot_product_attention(
            query * scale, key, value
        )
        return output.float().sum()

    query = torch.randn(2, 1, 4, 8, device="mps")
    key = torch.randn(2, 1, 4, 8, device="mps")
    value = torch.randn(2, 1, 4, 8, device="mps", dtype=torch.bfloat16)
    scale = torch.tensor(1.0, device="mps")
    try:
        with torch.autocast(device_type="mps", dtype=torch.bfloat16):
            torch.vmap(torch.func.grad(loss), in_dims=(None, 0, 0, 0))(
                scale, query, key, value
            )
    except RuntimeError:
        return True
    return False


@dataclass(frozen=True)
class DeviceCapabilities:
    """PyTorch capabilities for one device type."""

    device_type: str
    supports_bf16: bool
    supports_compile: bool
    recommended_compile_backend: str | None
    supports_fused_kernels: bool
    peak_memory_trackable: bool
    supports_pin_memory: bool

    @property
    def is_accelerator(self) -> bool:
        return self.device_type in ("cuda", "mps")


def device_capabilities(device: Any) -> DeviceCapabilities:
    """Resolve PyTorch capabilities for ``device``."""
    device_type = torch.device(device).type
    supports_compile = device_type in ("cuda", "mps", "cpu")
    return DeviceCapabilities(
        device_type=device_type,
        supports_bf16=_probe_bf16(device_type),
        supports_compile=supports_compile,
        recommended_compile_backend="inductor" if supports_compile else None,
        supports_fused_kernels=(device_type == "cuda" and fused_kernels_available()),
        peak_memory_trackable=device_type == "cuda",
        supports_pin_memory=device_type == "cuda",
    )


__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]
