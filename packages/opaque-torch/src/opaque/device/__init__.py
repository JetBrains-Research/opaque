"""Transitional facade: ``opaque.device`` moved to ``opaque.torch.device``.

Shipped by the Torch provider wheel while downstream code migrates;
scheduled for removal once the migration completes.
"""

from opaque.torch.device import (
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
