"""Federated DP primitives on IFED: cohort samplers, round loaders, and the
per-client clipped-gradient factory."""

from opaque.api.federated import (
    DataLoader,
    MinSepSampler,
    clipped_grad,
    make_clipping_aggregate,
)

__all__ = ["DataLoader", "MinSepSampler", "clipped_grad", "make_clipping_aggregate"]
