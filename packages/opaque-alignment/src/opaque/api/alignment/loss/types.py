"""Shared loss-type records for the alignment loss families.

Defines the inert metadata records that every loss-family ``types.py``
imports to declare DP compatibility:

- :class:`DPSpec` — per-loss DP-purity tier declaration (§3.3, §8). Read by
  trainers and the audit harness to decide whether a loss is admissible
  (Tier 1 / Tier 2) or rejected (Tier 3), and how its cross-batch aggregate
  (if any) must be handled.
- :class:`LossAggregateSpec` — declares a Tier-2 loss's required cross-batch
  aggregate so the trainer knows whether to compute it pre-vmap and whether
  to all-reduce it across ranks.

These are pure metadata: no tensors, no torch import. AGENTS.md rule 9
permits frozen dataclasses for inert state of this kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["DPSpec", "LossAggregateSpec"]


@dataclass(frozen=True)
class DPSpec:
    """Per-loss DP compatibility declaration. Read by trainers + audit harness.

    Attributes:
        tier: DP-purity tier (§3.3). ``1`` = strict per-example;
            ``2`` = per-example + detached batch aggregate with bounded
            leverage; ``3`` = rejected (rank/sort/quantile across batch).
        cross_batch_aggregate: Name of the required cross-batch aggregate for
            a Tier-2 loss (e.g. ``"kl_mean"``, ``"softmax_partition"``), or
            ``None`` for Tier-1 losses that need no aggregate.
        aggregate_must_detach: Whether the aggregate must be ``.detach()``-ed
            before entering the per-example loss (verified by the
            aggregate-detach audit).
        aggregate_leverage: Single-example leverage on the aggregate —
            ``"O(1)"``, ``"O(1/n)"``, or ``"sort"`` — or ``None`` when there
            is no aggregate.
        dp_safe: Whether the loss is admissible under the DP-purity invariant.
            ``False`` for Tier-3 (rejected) variants.
        rejection_reason: Human-readable rationale when ``dp_safe`` is
            ``False``; ``None`` otherwise.
    """

    tier: Literal[1, 2, 3]
    cross_batch_aggregate: str | None = None
    aggregate_must_detach: bool = True
    aggregate_leverage: Literal["O(1)", "O(1/n)", "sort"] | None = None
    dp_safe: bool = True
    rejection_reason: str | None = None


@dataclass(frozen=True)
class LossAggregateSpec:
    """Declares a Tier-2 loss's required cross-batch aggregate.

    The trainer reads this to know whether to compute the aggregate pre-vmap
    (and whether to all-reduce it across ranks).

    Attributes:
        name: Aggregate identifier (e.g. ``"kl_mean"``).
        reduction: How the aggregate reduces across the batch — ``"mean"`` or
            ``"sum"``.
        detach: Whether the aggregate must be detached from the autograd graph
            before it is broadcast into the per-example loss (DP-purity).
        cross_rank: When ``True``, the trainer routes the aggregate through
            ``opaque.distributed.all_reduce`` so it is computed over the global
            batch under DDP.
    """

    name: str
    reduction: Literal["mean", "sum"] = "mean"
    detach: bool = True
    cross_rank: bool = False
