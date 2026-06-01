"""opaque.alignment.loss types façade — re-exports the DP-purity records.

``DPSpec`` and ``LossAggregateSpec`` declare each loss's DP compatibility
(Tier 1/2/3) and any required cross-batch aggregate (see plan §7.4, §8).
"""

from opaque.api.alignment.loss.types import DPSpec, LossAggregateSpec

__all__ = ["DPSpec", "LossAggregateSpec"]
