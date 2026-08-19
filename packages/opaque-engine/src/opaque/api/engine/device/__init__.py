"""Transitional shim: device capabilities moved to ``opaque.api.torch.device``.

Kept while downstream packages migrate their imports to the Torch
provider wheel; scheduled for removal once the migration completes.
Imports resolve lazily so the torch-free engine wheel never touches the
provider unless a caller actually reaches for these symbols.
"""

from __future__ import annotations

__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]


def __getattr__(name: str):
    if name in __all__:
        import opaque.api.torch.device as _torch_device

        return getattr(_torch_device, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
