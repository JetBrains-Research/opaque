"""Identity MF strategy — uncorrelated noise baseline (DP-SGD via MF API).

Use ``mf_gaussian_noise(template, identity_strategy(), ...)`` for standard DP-SGD
with independent noise at each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opaque.api.dpftrl.noise._plan import MfExecutionPlan, identity_execution_plan
from opaque.api.dpftrl.noise._strategy_codec import register_strategy
from opaque.api.dpftrl.noise._streaming_matrix import StreamingMatrix, identity

if TYPE_CHECKING:
    import numpy as np


@register_strategy
@dataclass(frozen=True, slots=True)
class IdentityStrategy:
    """Identity (uncorrelated) MF strategy — independent noise per step.

    Encoder ``C = I``: sensitivity ≡ 1 for any horizon; streaming matrix
    is the identity; gram matrix isn't read (the BnB Identity path uses
    a dedicated native primitive that exploits ``G = num_epochs · I_b``).
    """

    def execution_plan(self, *, n_steps: int, **_) -> MfExecutionPlan:
        return identity_execution_plan(n_steps)

    def coefficients(self, *, n_steps: int, **_) -> np.ndarray:
        return self.execution_plan(n_steps=n_steps).coefficients()

    def gram_matrix(self, **_) -> tuple[float, ...]:
        # The BnB Identity path never materialises the gram: it is exactly
        # ``num_epochs · I_b``, so the dominating pair collapses onto random
        # allocation and goes through the analytic transform instead.
        raise NotImplementedError(
            "IdentityStrategy does not materialize a gram matrix; BnB Identity "
            "uses ``random_allocation_gaussian_pld`` directly."
        )

    def streaming_matrix(self, **_) -> StreamingMatrix:
        return identity()

    def sensitivity(self, **_) -> float:
        return 1.0


def identity_strategy() -> IdentityStrategy:
    """Create an identity (DP-SGD) noise strategy."""
    return IdentityStrategy()


__all__ = ["IdentityStrategy", "identity_strategy"]
