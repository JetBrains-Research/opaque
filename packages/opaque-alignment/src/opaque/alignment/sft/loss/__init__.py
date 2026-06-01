"""opaque.alignment.sft.loss façade — re-exports the SFT loss functions."""

from opaque.api.alignment.sft.loss import dft_loss, nll_loss

__all__ = ["nll_loss", "dft_loss"]
