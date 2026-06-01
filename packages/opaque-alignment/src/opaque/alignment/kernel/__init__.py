"""opaque.alignment.kernel façade — fused-linear preference kernels.

The per-pair loss protocol lives in :mod:`opaque.alignment.kernel.types`.
"""

from opaque.api.alignment.kernel import (
    fused_linear_preference,
    fused_linear_unpaired_preference,
    opaque_fused_linear_dpo_loss,
    opaque_fused_linear_kto_loss,
)

__all__ = [
    "opaque_fused_linear_dpo_loss",
    "opaque_fused_linear_kto_loss",
    "fused_linear_preference",
    "fused_linear_unpaired_preference",
]
