"""opaque.alignment.loss types façade — re-exports the DP-purity record.

``DPSpec`` declares each loss's DP compatibility — Tier 1 (strict per-example)
or Tier 3 (rejected: rank/sort/quantile across the batch). See plan §3.3, §8.
"""

from opaque.api.alignment.loss.types import DPSpec

__all__ = ["DPSpec"]
