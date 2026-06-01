"""opaque.alignment.sft — functional SFT primitives (method-first façade).

Mirrors the ``opaque.dpsgd`` / ``opaque.dpftrl`` mechanism-namespaced layout:
the SFT method owns its loss math (``opaque.alignment.sft.loss``) and, as they
land, its collator (``opaque.alignment.sft.collator``). Shared primitives
(``logprob``, ``metric``, ``data``) stay in their concern modules and are
reimported where needed.
"""

from opaque.alignment.sft.loss import dft_loss, nll_loss

__all__ = ["nll_loss", "dft_loss"]
