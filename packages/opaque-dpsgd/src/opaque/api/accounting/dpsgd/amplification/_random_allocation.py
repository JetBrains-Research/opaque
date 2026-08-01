"""Random-allocation amplification — a whole-EPOCH DP-SGD atom.

Each epoch, every example independently picks one of ``num_bins`` bins and
the epoch yields those bins as batches, redrawing the assignment every
epoch.  The privacy loss of one such epoch is the 1-out-of-``num_bins``
random-allocation PLD of Feldman & Shenfeld (2026), computed exactly by the
``random_allocation_gaussian_pld`` native primitive.

The per-epoch redraw makes epochs independent, so the cost of a full run is
this process composed ``* num_epochs``.  That is the one thing to get right
here: :class:`~opaque.api.accounting.dpsgd.amplification._poisson.Poisson`
next door is a per-*step* atom, and composing this one ``* n_steps`` would
over-count by a factor of ``num_bins``.

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

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian

#: Mechanism types accepted by :func:`random_allocation`.
_Inner = Gaussian | AdaClip | NonPrivate


@dataclass(frozen=True, slots=True)
class RandomAllocation(DpProcess):
    """1-out-of-``num_bins`` random allocation — a whole-**EPOCH** atom.

    Unlike :class:`Poisson`, which covers a single step, this covers all
    ``num_bins`` steps of one epoch.  Compose with ``* num_epochs``::

        epoch = dpsgd_acc.random_allocation(dpsgd_acc.gaussian(1.1), num_bins=100)
        training = epoch * 8          # 8 epochs = 800 steps
        eps = training.epsilon_at(1e-5)

    Composing ``* n_steps`` instead would charge ``num_bins`` times too
    much.  :attr:`steps_per_epoch` is the conversion factor.
    """

    inner: _Inner
    num_bins: int

    def __post_init__(self):
        # Validate here rather than only in the factory: deserialization
        # calls ``cls(**kwargs)`` directly, bypassing it.
        if int(self.num_bins) < 2:
            raise ValueError(
                f"RandomAllocation: num_bins must be >= 2, got {self.num_bins}"
            )

    @property
    def steps_per_epoch(self) -> int:
        """Training steps covered by one atom — equal to ``num_bins``.

        Present so ``training = epoch * (n_steps // epoch.steps_per_epoch)``
        reads correctly without the caller rederiving the relationship.
        """
        return self.num_bins

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )
        native_cfg = config.to_native()

        # ``k = 1``: every example lands in exactly one of the ``num_bins``
        # bins per epoch.  ``upper=True``: the lower variant exists only so
        # tests can bracket the discretisation error.
        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return _native.non_private_pld(native_cfg)
            case Gaussian(noise_multiplier=nm):
                return _native.random_allocation_gaussian_pld(
                    nm, self.num_bins, 1, True, native_cfg
                )
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return _native.non_private_pld(native_cfg)
            case AdaClip(inner=Gaussian()) as ac:
                return _native.random_allocation_gaussian_pld(
                    ac.effective_noise_multiplier, self.num_bins, 1, True, native_cfg
                )
            case _:
                raise TypeError(
                    "RandomAllocation requires a Gaussian, AdaClip(Gaussian), or "
                    "NonPrivate inner mechanism, got "
                    f"{type(self.inner).__name__}."
                )


def random_allocation(inner: _Inner, *, num_bins: int) -> RandomAllocation:
    """Random-allocation amplified mechanism (per-**epoch** DP-SGD factory).

    Returns a single-**epoch** process covering ``num_bins`` steps — compose
    externally with ``* num_epochs``, **not** ``* n_steps``.

    Amplifies strictly more than :func:`poisson` at the matched rate
    ``1 / num_bins``: an example participates exactly once per epoch rather
    than a Binomial number of times, so the worst case the accountant has to
    cover is smaller.

    Args:
        inner: The base mechanism — :func:`gaussian`, :func:`adaclip`, or
            :func:`opaque.accounting.nonprivate`.
        num_bins: Bins per epoch (``b >= 2``).  Must match the sampler's
            ``num_bins``; typically ``dataset_size // batch_size``.

    Returns:
        A :class:`RandomAllocation` process (one epoch).

    Example::

        num_bins = dataset_size // batch_size
        epoch = dpsgd_acc.random_allocation(
            dpsgd_acc.gaussian(1.1), num_bins=num_bins,
        )
        eps = (epoch * num_epochs).epsilon_at(1e-5)
    """
    match inner:
        case Gaussian() | AdaClip() | NonPrivate():
            pass
        case _:
            raise TypeError(
                "random_allocation() requires a Gaussian, AdaClip, or NonPrivate "
                f"inner mechanism, got {type(inner).__name__}. "
                "Example: dpsgd_acc.random_allocation(dpsgd_acc.gaussian(nm), "
                "num_bins=100)"
            )
    # ``num_bins`` bounds are validated in ``RandomAllocation.__post_init__``
    # so direct construction and deserialization stay safe.
    return RandomAllocation(inner=inner, num_bins=int(num_bins))
