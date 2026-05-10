"""Public type definitions for :mod:`opaque.dpftrl.sampling`.

Re-exports the partition-strategy enum used by
:class:`opaque.dpftrl.sampling.CyclicPoissonSampler` for type annotations
and explicit construction.
"""

from __future__ import annotations

from opaque.api.dpftrl.sampling._partitions import PartitionType

__all__ = ["PartitionType"]
