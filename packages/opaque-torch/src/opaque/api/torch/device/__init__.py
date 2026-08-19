"""PyTorch device capability helpers."""

from opaque.api.torch.device._capabilities import (
    DeviceCapabilities,
    device_capabilities,
    fused_kernels_available,
    sdpa_autocast_under_vmap_broken,
)

__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]
