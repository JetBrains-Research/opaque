"""Alignment-specific fused-linear preference kernel impl (plan §7.10).

Chunked, self-contained pure-PyTorch (no Triton): peak logits memory is
``O(chunk_size · T · V)`` instead of ``O(B · T · V)``, and the path stays
``torch.func``-composable (vmap/grad). The headline dispatcher covers DPO; the
reusable chunked core is exported for power users.
"""

from opaque.api.alignment.dpo.kernel._dpo_dispatch import opaque_fused_linear_dpo_loss
from opaque.api.alignment.dpo.kernel._fused_linear_preference import (
    fused_linear_preference,
)

__all__ = [
    "opaque_fused_linear_dpo_loss",
    "fused_linear_preference",
]
