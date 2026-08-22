"""Kernel selector constants — the activation kinds ``opaque_lora_mlp`` dispatches on."""

from opaque.api.kernels import (
    ACTIVATION_GEGLU_APPROX,
    ACTIVATION_GEGLU_EXACT,
    ACTIVATION_SWIGLU,
)

__all__ = [
    "ACTIVATION_GEGLU_APPROX",
    "ACTIVATION_GEGLU_EXACT",
    "ACTIVATION_SWIGLU",
]
