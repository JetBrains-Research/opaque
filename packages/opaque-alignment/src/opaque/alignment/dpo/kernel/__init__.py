"""opaque.alignment.dpo.kernel façade — the fused-linear DPO preference kernel.

The per-pair loss protocol lives in :mod:`opaque.alignment.dpo.kernel.types`.
"""

from opaque.api.alignment.dpo.kernel import (
    fused_linear_preference,
    opaque_fused_linear_dpo_loss,
)

__all__ = [
    "opaque_fused_linear_dpo_loss",
    "fused_linear_preference",
]
