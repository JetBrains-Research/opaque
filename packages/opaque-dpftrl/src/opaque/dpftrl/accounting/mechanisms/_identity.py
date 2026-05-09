"""MF identity baseline — uncorrelated (DP-SGD-style) noise under the MF training API.

Pairs with :func:`opaque.dpftrl.noise.identity_strategy` (encoder :math:`C^{-1}=I`).
Accounting is the **full run** under Poisson subsampling with per-step inclusion
rate ``sample_rate`` and ``num_steps`` iterations (same contract as
``examples/train_dp_ftrl.py`` with :class:`opaque.dpsgd.sampling.PoissonSampler`).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.accounting import _native

from opaque.accounting._base import DpProcess, Pld
from opaque.accounting.discretization import get_discretization


@dataclass(frozen=True, slots=True)
class IdentityMf(DpProcess):
    """Privacy cost of MF identity training — Poisson-subsampled Gaussian × steps.

    This is **not** matrix-factorisation correlated noise accounting; it matches
    uncorrelated Gaussian noise at each stochastic step.  Native PLD primitives
    are used only as an implementation detail — FTRL does **not** expose a
    generic Poisson amplification API.
    """

    noise_multiplier: float
    sample_rate: float
    num_steps: int

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        if self.num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        sr = float(self.sample_rate)
        if not (0.0 < sr <= 1.0):
            raise ValueError(f"sample_rate must be in (0, 1], got {self.sample_rate}")

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        native_cfg = config.to_native()

        if self.noise_multiplier == 0:
            return _native.non_private_pld(native_cfg)

        one_step = _native.poisson_gaussian_pld(
            float(self.noise_multiplier), sr, native_cfg
        )
        return one_step.self_compose(self.num_steps)


def mf_identity(
    noise_multiplier: float,
    *,
    sample_rate: float,
    num_steps: int,
) -> IdentityMf:
    """Whole-run accountant for MF :func:`~opaque.dpftrl.noise.identity_strategy`.

    Use the same ``sample_rate`` and step count as the training accountant for
    fair calibration (e.g. ``batch_size / |D|`` and ``total_steps`` in
    ``train_dp_ftrl.py``).

    Args:
        noise_multiplier: Gaussian noise multiplier (σ / Δ with Δ=1 on the summed
            gradient query scale used by calibration).
        sample_rate: Per-example inclusion probability per step ``(0, 1]``.
        num_steps: Number of composed stochastic steps ``>= 1``.

    Returns:
        An :class:`IdentityMf` process (total ε from ``epsilon_at``, not per-step).

    Example::

        proc = mf_identity(
            1.1, sample_rate=0.01, num_steps=1000
        )
        eps = proc.epsilon_at(1e-5)
    """
    nm = float(noise_multiplier)
    steps = int(num_steps)
    if steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {steps}")
    sr = float(sample_rate)
    if not (0.0 < sr <= 1.0):
        raise ValueError(f"sample_rate must be in (0, 1], got {sr}")
    return IdentityMf(noise_multiplier=nm, sample_rate=sr, num_steps=steps)


__all__ = ["IdentityMf", "mf_identity"]
