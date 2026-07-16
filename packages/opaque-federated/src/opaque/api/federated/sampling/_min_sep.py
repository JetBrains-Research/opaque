"""Minimum-separation cohort sampler — the federated ``BMinSepSampler``."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Mapping

from opaque.api.federated.data.types import Cohort, Population


class MinSepSampler:
    """Yield fixed-size cohort specs with a minimum-separation guarantee.

    The federated twin of :class:`opaque.dpftrl.sampling.BMinSepSampler`, with
    the *client* as the privacy unit and IFED as the enforcement mechanism:
    every yielded :class:`Cohort` carries ``separation = bands - 1``,
    which the consumer compiles onto IFED's ``assignSeparation`` policy — an
    agent that served round ``r`` is platform-ineligible until round
    ``r + bands``.

    Two central parameters are deliberately absent, because federation cannot
    honestly claim them today:

    - ``sampling_prob`` / ``key`` — IFED assigns greedily among live eligible
      agents; there is no randomized-selection guarantee, hence no
      subsampling-amplification claim. Pair this sampler with
      **non-amplified** BandMF accounting (``opaque.dpftrl.accounting``:
      ``min_sep = bands``, ``max_participations = ceil(rounds / bands)``).
    - ``n_steps`` — the round horizon lives on the
      :class:`~opaque.api.federated.data.DataLoader` (``rounds=``), which
      bounds this unbounded spec stream.

    Cohorts are exact: IFED collects contributions from exactly
    ``batch_size`` clients per round (the round blocks until it has them), so
    the batch size is a constant, not a random variable.

    Args:
        population: The Opaque :class:`Population` to draw cohorts from.
        batch_size: Exact clients per cohort (IFED ``cardinality``).
        bands: Minimum-separation parameter ``b`` (same as BandMF bandwidth);
            ``bands=1`` allows every agent every round.

    Example::

        population = opaque.federated.population("/hive")
        sampler = MinSepSampler(population, batch_size=8, bands=4)
        loader = DataLoader(population, batch_sampler=sampler, rounds=60)
    """

    def __init__(self, population: Population, batch_size: int, bands: int):
        if not isinstance(population, Population):
            raise TypeError(
                f"population must be an opaque.federated.Population, got {type(population).__name__}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if bands < 1:
            raise ValueError(f"bands must be >= 1, got {bands}")
        self.population = population
        self.batch_size = batch_size
        self.bands = bands
        self._consumed = 0

    @property
    def consumed(self) -> int:
        """Number of cohort specs yielded so far (resume cursor)."""
        return self._consumed

    @property
    def separation(self) -> int:
        """IFED ``assignSeparation.iterationDelta`` this sampler compiles to."""
        return self.bands - 1

    def __iter__(self) -> Iterator[Cohort]:
        while True:
            spec = Cohort(
                round=self._consumed,
                size=self.batch_size,
                separation=self.separation,
            )
            self._consumed += 1
            yield spec

    def __repr__(self) -> str:
        return (
            f"MinSepSampler(population={self.population.name!r}, "
            f"batch_size={self.batch_size}, bands={self.bands})"
        )


def _state_dict_min_sep(s: MinSepSampler) -> dict[str, Any]:
    return {
        "population_name": s.population.name,
        "population_version": s.population.version,
        "batch_size": int(s.batch_size),
        "bands": int(s.bands),
        "consumed": int(s.consumed),
    }


def _from_state_dict_min_sep(
    template: MinSepSampler, sd: Mapping[str, Any]
) -> MinSepSampler:
    """Rebuild a ``MinSepSampler`` at the saved cursor.

    The population comes from ``template`` and must match the snapshot —
    resuming against a different population would silently change the
    participation structure being accounted.
    """
    saved = Population(
        name=str(sd["population_name"]),
        version=str(sd.get("population_version", "*")),
    )
    have = template.population
    if saved != have:
        raise ValueError(
            f"MinSepSampler.from_state_dict: template population {have!r} does "
            f"not match snapshot {saved!r}."
        )
    sampler = MinSepSampler(
        template.population,
        batch_size=int(sd["batch_size"]),
        bands=int(sd["bands"]),
    )
    sampler._consumed = int(sd["consumed"])
    return sampler


def _register_min_sep_sampler_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(MinSepSampler, _state_dict_min_sep, _from_state_dict_min_sep)


_register_min_sep_sampler_serializer()
