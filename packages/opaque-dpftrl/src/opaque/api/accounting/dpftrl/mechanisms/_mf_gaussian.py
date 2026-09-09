"""Gaussian accounting for matrix-factorization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache
from opaque.api.accounting.core.discretization import get_discretization
from opaque.api.dpftrl.noise._schedule_fingerprint import strategy_cache_key
from opaque.exceptions import CheckpointError, ConfigurationError, InputTypeError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opaque.api.dpftrl.noise.types import MfStrategy


def _validate_positive_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputTypeError(*(f"{name} must be an int, got {type(value).__name__}",))
    if value < 1:
        raise ConfigurationError(*(f"{name} must be >= 1, got {value}",))


@dataclass(frozen=True, slots=True)
class MfGaussian(DpProcess):
    """A matrix-factorization Gaussian process.

    The stored participation context defines bare accounting. Amplifiers
    derive their context from the outer process.
    """

    noise_multiplier: float
    strategy: MfStrategy
    n_steps: int
    min_sep: int = 1
    max_participations: int | None = None

    def __post_init__(self) -> None:
        if self.noise_multiplier < 0:
            raise ConfigurationError(
                *(
                    f"noise_multiplier must be non-negative, got {self.noise_multiplier}",
                )
            )
        _validate_positive_int("n_steps", self.n_steps)
        _validate_positive_int("min_sep", self.min_sep)
        if self.max_participations is not None:
            _validate_positive_int("max_participations", self.max_participations)

    @property
    def _effective_max_participations(self) -> int:
        return (
            self.max_participations
            if self.max_participations is not None
            else self.n_steps
        )

    def _pld_cache_key(self) -> tuple[object, ...]:
        return (
            "MfGaussian",
            self.noise_multiplier,
            self.n_steps,
            self.min_sep,
            self.max_participations,
            strategy_cache_key(self.strategy, self.n_steps),
        )

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            seed=seed,
            mc_resolution=mc_resolution,
            mc_failure_probability=mc_failure_probability,
        )
        if self.noise_multiplier == 0:
            return _native.non_private_pld(config.to_native())
        sens = self.strategy.sensitivity(
            n_steps=self.n_steps,
            min_sep=self.min_sep,
            max_participations=self._effective_max_participations,
        )
        return _native.mf_gaussian_pld(
            self.noise_multiplier,
            sens,
            config.to_native(),
        )


def mf_gaussian(
    noise_multiplier: float,
    strategy: MfStrategy,
    *,
    n_steps: int,
    min_sep: int = 1,
    max_participations: int | None = None,
) -> MfGaussian:
    """Build a matrix-factorization Gaussian process.

    Args:
        noise_multiplier: Raw noise standard deviation σ (>= 0).
        strategy: Matrix-factorization strategy recipe.
        n_steps: Bare accounting horizon. Amplifiers use their outer horizon.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations (``None`` means
            ``n_steps``).

    Returns:
        An :class:`MfGaussian` process.
    """
    return MfGaussian(
        noise_multiplier=float(noise_multiplier),
        strategy=strategy,
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
    )


# The strategy codec owns the polymorphic strategy payload.


def _serialize_mf_gaussian(p: MfGaussian) -> dict[str, Any]:
    from opaque.api.dpftrl.noise._strategy_codec import serialize_strategy

    return {
        "type": "MfGaussian",
        "noise_multiplier": p.noise_multiplier,
        "strategy": serialize_strategy(p.strategy),
        "n_steps": p.n_steps,
        "min_sep": p.min_sep,
        "max_participations": p.max_participations,
    }


def _load_mf_gaussian(_template: Any, sd: Mapping[str, Any]) -> MfGaussian:
    from opaque.api.dpftrl.noise._strategy_codec import deserialize_strategy

    state = dict(sd)
    state.pop("type", None)
    try:
        noise_multiplier = state.pop("noise_multiplier")
        strategy_value = state.pop("strategy")
        n_steps = state.pop("n_steps")
    except KeyError as error:
        raise CheckpointError(
            *(f"missing required field {error.args[0]!r} for MfGaussian",)
        ) from None

    min_sep = state.pop("min_sep", 1)
    max_participations = state.pop("max_participations", None)
    if state:
        raise CheckpointError(*(f"unexpected keys for MfGaussian: {sorted(state)!r}",))

    return MfGaussian(
        noise_multiplier=noise_multiplier,
        strategy=deserialize_strategy(dict(strategy_value)),
        n_steps=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
    )


def _register_mf_gaussian_serializer() -> None:
    from opaque.serialization import register_serializer

    register_serializer(MfGaussian, _serialize_mf_gaussian, _load_mf_gaussian)


_register_mf_gaussian_serializer()
