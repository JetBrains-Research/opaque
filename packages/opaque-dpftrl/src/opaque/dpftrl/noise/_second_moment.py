"""Internal state for paired-stream MF noise.

The runtime allocation lives in
``opaque._noise_allocation.paired_noise_stddevs``; this module only carries
the joint state that wraps both stream noise states.
"""

from __future__ import annotations

import dataclasses

from opaque.random.types import RngKey
from opaque.types import NoiseState

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
