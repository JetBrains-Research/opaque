"""BandMF with cyclic Poisson amplification — amplified MF accounting.

When BandMF uses cyclic Poisson subsampling with band width b, the n
training rounds divide into k = ceil(n/b) groups. Each group is an
independent Poisson-subsampled Gaussian mechanism. The total privacy
is the k-fold composition of per-group PLDs.

References:
    - BandMF amplification: Choquette-Choo et al. (2023) https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .. import opaque_accounting as _native

from opaque_accounting.base import (
    DpProcess,
    Pld,
)
from opaque_accounting.discretization import DiscretizationConfig


@dataclass(frozen=True, slots=True)
class BandMfAmplified(DpProcess):
    """BandMF with cyclic Poisson amplification.

    Composes num_groups independent Poisson-subsampled Gaussian mechanisms.
    Each group uses effective noise multiplier noise_multiplier / sensitivity
    and the given sample_rate.
    """

    noise_multiplier: float
    sensitivity: float
    sample_rate: float
    num_groups: int

    @functools.lru_cache(maxsize=8)
    def pmf(self, config: DiscretizationConfig) -> Pld:
        effective_nm = self.noise_multiplier / self.sensitivity
        per_group_pld = _native.poisson_gaussian_pld(
            effective_nm, self.sample_rate, config.to_native()
        )
        return per_group_pld.self_compose(self.num_groups)


def band_mf_amplified(
    noise_multiplier: float,
    sensitivity: float,
    sample_rate: float,
    num_groups: int,
) -> BandMfAmplified:
    """BandMF with cyclic Poisson amplification.

    Composes ``num_groups`` independent Poisson-subsampled Gaussian mechanisms,
    where each group uses effective noise multiplier ``noise_multiplier / sensitivity``
    and sampling probability ``sample_rate``.

    For BandMF with band width b and n total rounds:
    - ``num_groups = ceil(n / b)``
    - ``sample_rate = b * batch_size / dataset_size``
    - ``sensitivity`` from ``banded_sensitivity()`` or
      ``toeplitz_minsep_sensitivity_squared().sqrt()``

    Args:
        noise_multiplier: Raw noise standard deviation sigma.
        sensitivity: L2 sensitivity of the encoder matrix under the
            participation pattern.
        sample_rate: Poisson sampling probability per group
            (typically b * batch_size / dataset_size).
        num_groups: Number of independent groups (typically ceil(n / b)).

    Returns:
        A :class:`BandMfAmplified` process.

    Example::

        import opaque.accounting as acc

        # BandMF with 5 bands over 1000 rounds
        proc = acc.band_mf_amplified(
            noise_multiplier=1.0,
            sensitivity=2.5,
            sample_rate=0.01,
            num_groups=200,  # ceil(1000 / 5)
        )
        eps = proc.epsilon_at(1e-5)
    """
    if noise_multiplier <= 0:
        raise ValueError(f"noise_multiplier must be positive, got {noise_multiplier}")
    if sensitivity <= 0:
        raise ValueError(f"sensitivity must be positive, got {sensitivity}")
    if not 0 < sample_rate <= 1:
        raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if num_groups < 1:
        raise ValueError(f"num_groups must be >= 1, got {num_groups}")
    return BandMfAmplified(noise_multiplier, sensitivity, sample_rate, num_groups)
