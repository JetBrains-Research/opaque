"""SFT loss family impl — ``nll`` and ``dft`` (DP-corrected per-example divisor),
each with a memory-efficient fused twin.

Direct functions only: there is no string registry / resolver / variant enum.
A name→function resolver is reintroduced only when a config-string consumer
(e.g. a trainer or CLI) actually needs one.

``nll_loss`` / ``dft_loss`` are **strict per-example** (swapping one example's
data changes only that example's gradient, enforced by the NaN-injection test in
``tests/sft/loss/test_sft.py``). ``fused_nll_loss`` / ``fused_dft_loss`` are
their memory-efficient drop-ins: same per-example math, but they take the hidden
states + ``lm_head`` weight and fuse the projection through the opaque-patches
linear-CE kernel (no ``(T, V)`` logits), with an eager fallback. The fused twins
are per-example — drive them with ``vmap(grad)``.
"""

from opaque.api.alignment.sft.loss._dft import dft_loss, fused_dft_loss
from opaque.api.alignment.sft.loss._nll import fused_nll_loss, nll_loss

__all__ = ["nll_loss", "dft_loss", "fused_nll_loss", "fused_dft_loss"]
