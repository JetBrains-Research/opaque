"""Kernel types — the per-pair loss callable protocol used by the chunked core."""

from opaque.api.alignment.dpo.kernel._fused_linear_preference import PerPairLossFn

__all__ = ["PerPairLossFn"]
