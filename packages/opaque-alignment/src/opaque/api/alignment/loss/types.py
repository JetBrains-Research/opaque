"""Shared loss-type record for the alignment loss families.

Defines the inert metadata record every loss-family ``types.py`` imports to
declare DP compatibility:

- :class:`DPSpec` — per-loss DP-purity tier declaration (§3.3, §8). Read by
  trainers and the audit harness to decide whether a loss is admissible
  (Tier 1, strict per-example) or rejected (Tier 3, rank/sort/quantile across
  the batch).

Pure metadata: no tensors, no torch import. AGENTS.md rule 9 permits frozen
dataclasses for inert state of this kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["DPSpec"]


@dataclass(frozen=True)
class DPSpec:
    """Per-loss DP compatibility declaration. Read by trainers + audit harness.

    Attributes:
        tier: DP-purity tier (§3.3). ``1`` = strict per-example;
            ``3`` = rejected (rank/sort/quantile across batch).
        dp_safe: Whether the loss is admissible under the DP-purity invariant.
            ``False`` for Tier-3 (rejected) variants.
        aggregate_leverage: Single-example leverage on a batch aggregate —
            ``"O(1)"``, ``"O(1/n)"``, or ``"sort"`` — or ``None`` when there is
            no aggregate (e.g. ``"sort"`` records why a Tier-3 variant is
            rejected).
        rejection_reason: Human-readable rationale when ``dp_safe`` is
            ``False``; ``None`` otherwise.
    """

    tier: Literal[1, 3]
    dp_safe: bool = True
    aggregate_leverage: Literal["O(1)", "O(1/n)", "sort"] | None = None
    rejection_reason: str | None = None
