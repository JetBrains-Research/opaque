"""Shared cache decorators for PLD-producing process methods."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .discretization import (
    DiscretizationConfig,
    _use_discretization,
    get_discretization,
)

if TYPE_CHECKING:
    from ._base import Pld


def _resolve_config(
    *,
    discretization: float | None,
    log_x_mass_truncation_bound: float | None,
    max_grid_size: int | None,
    max_conv_grid: int | None,
    num_mc_samples: int | None,
    seed: int | None,
) -> DiscretizationConfig:
    return get_discretization(
        discretization=discretization,
        log_x_mass_truncation_bound=log_x_mass_truncation_bound,
        max_grid_size=max_grid_size,
        max_conv_grid=max_conv_grid,
        num_mc_samples=num_mc_samples,
        seed=seed,
    )


def pld_cache(*, maxsize: int):
    """Cache a ``DpProcess.pld`` method by resolved configuration and mechanism."""

    def decorator(method):
        @functools.lru_cache(maxsize=maxsize)
        def cached(process, fingerprint, config: DiscretizationConfig) -> Pld:
            with _use_discretization(config):
                return method(process)

        @functools.wraps(method)
        def wrapper(
            self,
            *,
            discretization: float | None = None,
            log_x_mass_truncation_bound: float | None = None,
            max_grid_size: int | None = None,
            max_conv_grid: int | None = None,
            num_mc_samples: int | None = None,
            seed: int | None = None,
        ) -> Pld:
            config = _resolve_config(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                num_mc_samples=num_mc_samples,
                seed=seed,
            )
            return cached(self, self._pld_cache_fingerprint(), config)

        wrapper.cache_clear = cached.cache_clear
        wrapper.cache_info = cached.cache_info
        return wrapper

    return decorator


def horizon_pld_cache(*, maxsize: int):
    """Cache a horizon ``pld_at`` method by configuration and prefix mechanism."""

    def decorator(method):
        @functools.lru_cache(maxsize=maxsize)
        def cached(
            process, fingerprint, config: DiscretizationConfig, n_steps: int
        ) -> Pld:
            with _use_discretization(config):
                return method(process, n_steps)

        @functools.wraps(method)
        def wrapper(
            self,
            n_steps: int,
            *,
            discretization: float | None = None,
            log_x_mass_truncation_bound: float | None = None,
            max_grid_size: int | None = None,
            max_conv_grid: int | None = None,
            num_mc_samples: int | None = None,
            seed: int | None = None,
        ) -> Pld:
            if n_steps <= 0 or n_steps > self.n_steps:
                return method(
                    self,
                    n_steps,
                    discretization=discretization,
                    log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                    max_grid_size=max_grid_size,
                    max_conv_grid=max_conv_grid,
                    num_mc_samples=num_mc_samples,
                    seed=seed,
                )
            config = _resolve_config(
                discretization=discretization,
                log_x_mass_truncation_bound=log_x_mass_truncation_bound,
                max_grid_size=max_grid_size,
                max_conv_grid=max_conv_grid,
                num_mc_samples=num_mc_samples,
                seed=seed,
            )
            return cached(
                self, self._pld_cache_fingerprint(n_steps=n_steps), config, n_steps
            )

        wrapper.cache_clear = cached.cache_clear
        wrapper.cache_info = cached.cache_info
        return wrapper

    return decorator
