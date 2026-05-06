"""Private second-moment support for MF noise mechanisms.

The public entry point is :func:`opaque.dpftrl.noise.mf_noise` with
``second_moment=True`` (or a custom first-stream overhead).  This module
only defines the paired-stream state type used by that dispatcher.
"""

from __future__ import annotations

import dataclasses

from opaque.types import NoiseState
from opaque.random import RngKey

from ._engine import MFNoiseState


@dataclasses.dataclass(frozen=True)
class SecondMomentMFNoiseState(NoiseState):
    """Internal state for MF noise with private second moments."""

    _first_state: MFNoiseState
    _second_state: MFNoiseState

    @property
    def _step_counter(self) -> int:  # type: ignore[override]
        return self._first_state._step_counter

    @property
    def _rng_key(self) -> RngKey:  # type: ignore[override]
        return self._first_state._rng_key


__all__ = ["SecondMomentMFNoiseState"]
