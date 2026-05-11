"""MF Gaussian mechanism — one mechanism class for every MF strategy.

The privacy of a matrix-factorization Gaussian release reduces to a single
Gaussian mechanism with effective noise multiplier σ/S, regardless of which
encoder C the training side used.  The accounting mechanism is therefore a
thin wrapper over a noise multiplier and the strategy (which carries the
matrix-factorization shape: sensitivity, Gram matrix, coefficients, etc.).

The accounting amplifications (Poisson, BMinSep, BallsInBins) dispatch on
``type(mechanism.strategy)`` to select the right native PLD primitive.

Built via the :func:`mf_gaussian` factory in this module:

    proc = mf_gaussian(noise_multiplier, strategy)

where ``strategy`` is one of the dataclasses from :mod:`opaque.dpftrl.noise`
(``BltStrategy``, ``BsrStrategy``, ``BisrStrategy``, ``LambdaCgdStrategy``,
``BandMfStrategy``, ``IdentityStrategy``).

Serialization: a custom serializer pair is registered here that emits
``{"type": "MfGaussian", "noise_multiplier": ..., "strategy":
{"type": "<StrategyName>", ...}}``.  The strategy sub-dict is produced
and consumed by :mod:`opaque.api.dpftrl.noise._strategy_codec`, which
owns the strategy class registry.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opaque.api.accounting.core import _native

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core.discretization import get_discretization

if TYPE_CHECKING:
    from opaque.api.dpftrl.noise.types import MfStrategy


@dataclass(frozen=True, slots=True)
class MfGaussian(DpProcess):
    """MF Gaussian mechanism — ``noise_multiplier`` + ``strategy``.

    ``strategy`` is one of the dataclasses from :mod:`opaque.dpftrl.noise`.
    Its ``sensitivity`` (and, when relevant, ``gram_matrix`` /
    ``coefficients``) is read by the surrounding amplification at PLD time.
    """

    noise_multiplier: float
    strategy: "MfStrategy"

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )
        if self.noise_multiplier == 0:
            return _native.non_private_pld(config.to_native())
        return _native.mf_gaussian_pld(
            self.noise_multiplier,
            self.strategy.sensitivity,
            config.to_native(),
        )


def mf_gaussian(noise_multiplier: float, strategy: "MfStrategy") -> MfGaussian:
    """MF Gaussian mechanism — noise multiplier + strategy.

    Standalone, this models a single Gaussian release with effective noise
    multiplier ``noise_multiplier / strategy.sensitivity``.  Wrap in an
    amplification factory (``poisson``, ``b_min_sep``, ``balls_in_bins``)
    for the per-amplification PLD.

    Args:
        noise_multiplier: Raw noise standard deviation σ.
        strategy: One of the strategy dataclasses from
            :mod:`opaque.dpftrl.noise` — ``BltStrategy``, ``BsrStrategy``,
            ``BisrStrategy``, ``LambdaCgdStrategy``, ``BandMfStrategy``, or
            ``IdentityStrategy``.

    Returns:
        An :class:`MfGaussian` process.

    Example::

        s = blt_strategy(n_steps=100, min_sep=25, max_participations=4)
        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s), num_bins=25, n_steps=100,
        )
    """
    nm = float(noise_multiplier)
    if nm < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    return MfGaussian(noise_multiplier=nm, strategy=strategy)


# --- Custom serialization ---------------------------------------------------
#
# MfGaussian holds a ``strategy`` field whose value is one of the
# strategy dataclasses from opaque.dpftrl.noise.  The generic DpProcess
# codec only knows how to emit primitives, containers, and nested
# DpProcess values; it would silently drop the strategy.  Register a
# custom serializer pair that delegates strategy (de)serialization to
# the strategy codec, which owns the strategy class name registry.


def _serialize_mf_gaussian(p: MfGaussian) -> dict[str, Any]:
    from opaque.api.dpftrl.noise._strategy_codec import serialize_strategy

    return {
        "type": "MfGaussian",
        "noise_multiplier": p.noise_multiplier,
        "strategy": serialize_strategy(p.strategy),
    }


def _load_mf_gaussian(_template: Any, sd: dict[str, Any]) -> MfGaussian:
    from opaque.api.dpftrl.noise._strategy_codec import deserialize_strategy

    sd = dict(sd)
    sd.pop("type", None)
    nm = sd["noise_multiplier"]
    strategy = deserialize_strategy(dict(sd["strategy"]))
    return MfGaussian(noise_multiplier=nm, strategy=strategy)


def _register_mf_gaussian_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(
        MfGaussian, _serialize_mf_gaussian, _load_mf_gaussian
    )


_register_mf_gaussian_serializer()
