"""PyTorch device capability helpers.

The ``DeviceCapabilities`` record :func:`device_capabilities` returns
lives in :mod:`opaque.torch.device.types`.
"""

from opaque.api.torch.device import (
    device_capabilities,
    fused_kernels_available,
    sdpa_autocast_under_vmap_broken,
)
from opaque.torch.device import types

__all__ = [
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
    "types",
]
