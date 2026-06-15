"""Device capability helpers.

What bf16 / ``torch.compile`` / fused-kernel / peak-memory features a given
device actually supports, resolved in one place so call sites query a
capability instead of re-deriving it.  See :mod:`opaque.api.engine.device`
for probe details.
"""

from opaque.api.engine.device import (
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
