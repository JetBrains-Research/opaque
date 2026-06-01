"""SFT loss family impl — ``nll`` and ``dft`` (DP-corrected per-example divisor).

Direct functions only: there is no string registry / resolver / variant enum.
A name→function resolver is reintroduced only when a config-string consumer
(e.g. a trainer or CLI) actually needs one.

Both losses are **strict per-example** (Tier 1): swapping one example's data
changes only that example's gradient. That property is enforced by the
NaN-injection test in ``tests/sft/loss/test_sft.py`` rather than carried as
loss metadata.
"""

from opaque.api.alignment.sft.loss._dft import dft_loss
from opaque.api.alignment.sft.loss._nll import nll_loss

__all__ = ["nll_loss", "dft_loss"]
