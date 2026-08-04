"""Random-allocation amplification over a declared DP-SGD horizon.

Each epoch, every example independently picks one of ``num_bins`` bins and
the epoch yields those bins as batches, redrawing the assignment every
epoch.  The privacy loss of one such epoch is the 1-out-of-``num_bins``
random-allocation PLD of Feldman & Shenfeld (2026), computed exactly by the
``random_allocation_gaussian_pld`` native primitive.

The process exposes exact prefix accounting: complete epochs compose
independently, while a partial final epoch uses the exact released-prefix PLD.

Pairs with :class:`opaque.dpsgd.sampling.RandomAllocationSampler` and with
nothing else.  ``PoissonSampler`` is not an allocation, and DP-FTRL's
``BallsInBinsSampler`` holds the assignment fixed across epochs — a
different (weaker) scheme that has its own accountant,
``dpftrl_acc.balls_in_bins``.

References:
    - Feldman & Shenfeld (2026), "Efficient privacy loss accounting for
      subsampling and random allocation"
    - Chua et al. (2025), "Balls-and-Bins Sampling for DP-SGD"
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import Pld

#: Mechanism types accepted by :func:`random_allocation`.
_Inner = Gaussian | AdaClip | NonPrivate


@dataclass(frozen=True, slots=True)
class RandomAllocation(DpHorizonProcess):
    """Redrawn 1-out-of-``num_bins`` allocation over ``n_steps`` releases.

    ``pld_at(K)`` is exact at every prefix, including a partial final epoch.
    Use :func:`opaque.accounting.per_step` in step-wise training loops.

    The assignment is independently redrawn at every epoch boundary.
    """

    inner: _Inner
    num_bins: int
    n_steps: int

    def __post_init__(self):
        # Validate here rather than only in the factory: deserialization
        # calls ``cls(**kwargs)`` directly, bypassing it.
        if int(self.num_bins) < 2:
            raise ValueError(
                f"RandomAllocation: num_bins must be >= 2, got {self.num_bins}"
            )
        if int(self.n_steps) < 1:
            raise ValueError(
                f"RandomAllocation: n_steps must be >= 1, got {self.n_steps}"
            )

    @property
    def steps_per_epoch(self) -> int:
        """Training steps covered by one atom — equal to ``num_bins``.

        Present so ``training = epoch * (n_steps // epoch.steps_per_epoch)``
        reads correctly without the caller rederiving the relationship.
        """
        return self.num_bins

    @property
    def num_epochs(self) -> int:
        """Number of allocation epochs touched by the declared horizon."""
        return -(-self.n_steps // self.num_bins)

    def _effective_noise_multiplier(self) -> float | None:
        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return None
            case Gaussian(noise_multiplier=nm):
                return nm
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return None
            case AdaClip(inner=Gaussian()) as ac:
                return ac.effective_noise_multiplier
            case _:
                raise TypeError(
                    "RandomAllocation requires a Gaussian, AdaClip(Gaussian), or "
                    "NonPrivate inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )

    @functools.lru_cache(maxsize=16)
    def pld_at(
        self,
        n_steps: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
    ) -> Pld:
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
        )
        native_cfg = config.to_native()

        if n_steps < 1 or n_steps > self.n_steps:
            raise ValueError(f"n_steps ({n_steps}) must be in [1, {self.n_steps}]")
        noise_multiplier = self._effective_noise_multiplier()
        if noise_multiplier is None:
            return _native.non_private_pld(native_cfg)

        full_epochs, remainder = divmod(n_steps, self.num_bins)
        result = None
        if full_epochs:
            epoch = _native.random_allocation_gaussian_pld(
                noise_multiplier,
                self.num_bins,
                1,
                native_cfg,
            )
            result = epoch if full_epochs == 1 else epoch.self_compose(full_epochs)
        if remainder:
            prefix = _native.random_allocation_gaussian_prefix_pld(
                noise_multiplier,
                self.num_bins,
                remainder,
                native_cfg,
            )
            result = prefix if result is None else result.compose(prefix)
        assert result is not None
        return result


def random_allocation(
    inner: _Inner,
    *,
    num_bins: int,
    n_steps: int,
) -> RandomAllocation:
    """Create a redrawn random-allocation horizon process.

    Every epoch redraws the assignment independently. Prefix accounting is
    exact for every number of released steps.

    Amplifies strictly more than :func:`poisson` at the matched rate
    ``1 / num_bins``: an example participates exactly once per epoch rather
    than a Binomial number of times, so the worst case the accountant has to
    cover is smaller.

    Args:
        inner: The base mechanism — :func:`gaussian`, :func:`adaclip`, or
            :func:`opaque.accounting.nonprivate`.
        num_bins: Bins per epoch (``b >= 2``).  Must match the sampler's
            ``num_bins``; typically ``dataset_size // batch_size``.
        n_steps: Total optimizer-step horizon.

    Returns:
        A whole-horizon :class:`RandomAllocation` process.

    Example::

        num_bins = dataset_size // batch_size
        process = dpsgd_acc.random_allocation(
            dpsgd_acc.gaussian(1.1),
            num_bins=num_bins,
            n_steps=1000,
        )
        eps = process.epsilon_at(1e-5)
    """
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case _:
            raise TypeError(
                "random_allocation() requires a Gaussian, AdaClip, or NonPrivate "
                f"inner mechanism, got {type(inner).__name__}. "
                "Example: dpsgd_acc.random_allocation(dpsgd_acc.gaussian(nm), "
                "num_bins=100, n_steps=1000)"
            )
    # ``num_bins`` bounds are validated in ``RandomAllocation.__post_init__``
    # so direct construction and deserialization stay safe.
    return RandomAllocation(
        inner=inner,
        num_bins=int(num_bins),
        n_steps=int(n_steps),
    )
