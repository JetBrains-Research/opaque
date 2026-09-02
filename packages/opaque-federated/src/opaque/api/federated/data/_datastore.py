"""Federated datastore built from a cohort sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opaque.exceptions import InputTypeError

if TYPE_CHECKING:
    from ifed import FederatedDatastore

    from opaque.api.federated.sampling import MinSepSampler

_OWNED = ("population", "version", "cardinality")


def datastore(sampler: MinSepSampler, **requirements: Any) -> FederatedDatastore:
    """Build the ``ifed.FederatedDatastore`` a sampler's cohorts describe.

    Population, version and cardinality come from the sampler, so the three
    numbers the accounting is computed from cannot drift from the ones the
    round actually runs with. Everything else — ``server``, ``datasets``, the
    hardware requirements — is passed through to
    :class:`ifed.FederatedDatastore` unchanged.

    Args:
        sampler: The :class:`~opaque.api.federated.sampling.MinSepSampler`
            whose population and ``batch_size`` bind the store.
        **requirements: Any other :class:`ifed.FederatedDatastore` field.

    Returns:
        A plain :class:`ifed.FederatedDatastore`, ready for ``ifed.session``.
    """
    from ifed import FederatedDatastore

    owned = sorted(name for name in _OWNED if name in requirements)
    if owned:
        raise InputTypeError(
            *(
                f"datastore() takes {owned} from the sampler, not as keywords: "
                f"population={sampler.population.name!r}, "
                f"version={sampler.population.version!r}, "
                f"cardinality={sampler.batch_size}. Change the sampler instead, "
                f"so the accounting and the round agree.",
            )
        )
    return FederatedDatastore(
        population=sampler.population.name,
        version=sampler.population.version,
        cardinality=sampler.batch_size,
        **requirements,
    )


__all__ = ["datastore"]
