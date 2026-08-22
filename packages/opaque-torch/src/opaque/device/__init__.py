"""Deprecated alias: ``opaque.device`` moved to :mod:`opaque.torch.device`.

Device capabilities are a property of the Torch provider, so the surface
belongs under ``opaque.torch``, not at the backend-neutral namespace
root. Every name this module ever exported still resolves here —
including ``DeviceCapabilities``, now at :mod:`opaque.torch.device.types`
— each with a ``DeprecationWarning`` naming its replacement, once per
name per process. The module is removed in the next minor release.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - resolved lazily at runtime
    from opaque.torch.device import (
        device_capabilities,
        fused_kernels_available,
        sdpa_autocast_under_vmap_broken,
    )
    from opaque.torch.device.types import DeviceCapabilities

__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]

# Names that moved further than the module itself did.
_RELOCATED = {"DeviceCapabilities": "opaque.torch.device.types"}


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Any:
    """Resolve against ``opaque.torch.device``, warning on first access.

    The resolved object is cached in ``globals()`` — the repo's PEP 562
    house style — so the warning fires once per name rather than once per
    lookup. ``from opaque.device import x`` probes the attribute twice.
    """
    if name not in __all__:
        raise AttributeError(f"module 'opaque.device' has no attribute {name!r}")

    from importlib import import_module

    target = _RELOCATED.get(name, "opaque.torch.device")
    warnings.warn(
        f"opaque.device is deprecated; use {target}.{name} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    value = getattr(import_module(target), name)
    globals()[name] = value
    return value
