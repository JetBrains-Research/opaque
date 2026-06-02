"""SFT loss family impl — ``nll`` and ``dft`` (DP-corrected per-example divisor),
plus the opt-in chunked ``fused_linear_sft_loss``.

Direct functions only: there is no string registry / resolver / variant enum.
A name→function resolver is reintroduced only when a config-string consumer
(e.g. a trainer or CLI) actually needs one.

``nll_loss`` / ``dft_loss`` are **strict per-example**: swapping one
example's data changes only that example's gradient, enforced by the
NaN-injection test in ``tests/sft/loss/test_sft.py`` rather than carried as loss
metadata. ``fused_linear_sft_loss`` is the memory-efficient (opt-in) variant —
same math, chunked over the batch — selected by passing ``loss_fn=nll_loss`` or
``loss_fn=dft_loss``.
"""

from opaque.api.alignment.sft.loss._dft import dft_loss
from opaque.api.alignment.sft.loss._fused import fused_linear_sft_loss
from opaque.api.alignment.sft.loss._nll import nll_loss

__all__ = ["nll_loss", "dft_loss", "fused_linear_sft_loss"]
