"""Alignment-specific fused-linear preference kernels impl (plan §7.10).

Chunked, self-contained pure-PyTorch (no Triton): peak logits memory is
``O(chunk_size · T · V)`` instead of ``O(B · T · V)``, and the path stays
``torch.func``-composable (vmap/grad). The headline dispatchers cover DPO and
KTO; the reusable chunked cores are exported for power users.
"""

from opaque.api.alignment.kernel._dpo_dispatch import opaque_fused_linear_dpo_loss
from opaque.api.alignment.kernel._fused_linear_preference import (
    fused_linear_preference,
)
from opaque.api.alignment.kernel._fused_linear_unpaired import (
    fused_linear_unpaired_preference,
)
from opaque.api.alignment.kernel._kto_dispatch import opaque_fused_linear_kto_loss

__all__ = [
    "opaque_fused_linear_dpo_loss",
    "opaque_fused_linear_kto_loss",
    "fused_linear_preference",
    "fused_linear_unpaired_preference",
]
