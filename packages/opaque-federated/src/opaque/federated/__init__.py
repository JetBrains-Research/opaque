"""Federated DP primitives on IFED: cohort samplers, round loaders, and the
per-client clipped-gradient factory."""

from opaque.api.federated import (
    Cohort,
    DataLoader,
    MinSepSampler,
    Population,
    clipped_grad,
    make_clipping_aggregate,
)

__all__ = ["Cohort", "DataLoader", "MinSepSampler", "Population", "clipped_grad", "make_clipping_aggregate"]
