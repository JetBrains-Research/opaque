"""Test-only Torch/autograd references for pre-neutral-planner MF planning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional

from opaque.api.dpftrl.noise._blt_math import Parameterization, get_init_blt
from opaque.api.dpftrl.noise._sensitivity import minsep_true_max_participations
from opaque.api.dpftrl.noise._toeplitz import optimal_max_error_strategy_coefs

_DTYPE = torch.float64
_EPS = 1e-9


def _band_mf_objective(
    coefficients: torch.Tensor,
    *,
    n: int,
    workload: torch.Tensor,
) -> torch.Tensor:
    inverse_coefficients = []
    for index in range(n):
        width = min(index, len(coefficients) - 1)
        previous = (
            torch.stack(inverse_coefficients[-width:]).flip(0)
            if width
            else coefficients.new_empty(0)
        )
        numerator = workload[index] - torch.dot(coefficients[1 : width + 1], previous)
        inverse_coefficients.append(numerator / coefficients[0])
    inverse = torch.stack(inverse_coefficients)
    error = torch.cumsum(inverse.square(), dim=0).mean()
    return error * coefficients.square().sum()


def legacy_band_mf_coefficients(
    *, n: int, bands: int, momentum: float = 1.0, max_optimizer_steps: int = 250
) -> np.ndarray:
    """Optimise the former BandMF objective with Torch L-BFGS/autograd."""
    coefficients = torch.nn.Parameter(
        torch.tensor(optimal_max_error_strategy_coefs(bands), dtype=_DTYPE)
    )
    workload = torch.tensor(
        np.power(momentum, np.arange(n), dtype=np.float64), dtype=_DTYPE
    )
    optimizer = torch.optim.LBFGS(
        [coefficients],
        max_iter=max_optimizer_steps,
        tolerance_change=1e-15,
        tolerance_grad=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        objective = _band_mf_objective(coefficients, n=n, workload=workload)
        objective.backward()
        return objective

    optimizer.step(closure)
    result = coefficients.detach().numpy()
    return result / np.linalg.norm(result)


def _rational_weights(theta: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
    difference = theta[:, None] - theta[None, :]
    difference = difference + torch.eye(len(theta), dtype=_DTYPE)
    return (theta[:, None] - theta_hat[None, :]).prod(dim=1) / difference.prod(dim=1)


def _blt_coefficients(
    theta: torch.Tensor, theta_hat: torch.Tensor, n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    def coefficients(decay: torch.Tensor, other_decay: torch.Tensor) -> torch.Tensor:
        weights = _rational_weights(decay, other_decay)
        powers = torch.arange(n - 1, dtype=_DTYPE)[:, None]
        tail = (decay[None, :].pow(powers) * weights).sum(dim=1)
        return torch.cat((torch.ones(1, dtype=_DTYPE), tail))

    return coefficients(theta, theta_hat), coefficients(theta_hat, theta)


def _minsep_sensitivity_squared(
    coefficients: torch.Tensor, *, min_sep: int, max_participations: int
) -> torch.Tensor:
    n = len(coefficients)
    padding = (-n) % min_sep
    padded = functional.pad(coefficients, (0, padding))
    vector = padded.reshape(-1, min_sep).cumsum(dim=0).reshape(-1)
    active = min_sep * max_participations
    if active < len(vector):
        vector = torch.cat((vector[:active], vector[active:] - vector[:-active]))
    return vector[:n].square().sum()


def _blt_objective(
    params: torch.Tensor,
    *,
    n: int,
    min_sep: int,
    max_participations: int,
    workload: torch.Tensor,
) -> torch.Tensor:
    theta, theta_hat = params.chunk(2)
    coefficients, inverse_coefficients = _blt_coefficients(theta, theta_hat, n)
    noising = functional.conv1d(
        workload[None, None], inverse_coefficients.flip(0)[None, None], padding=n - 1
    )[0, 0, :n]
    error = torch.cumsum(noising.square(), dim=0).max()
    return error * _minsep_sensitivity_squared(
        coefficients, min_sep=min_sep, max_participations=max_participations
    )


def _feasible_blt_params(params: torch.Tensor) -> bool:
    theta, theta_hat = params.chunk(2)
    weights = _rational_weights(theta, theta_hat)
    inverse_weights = _rational_weights(theta_hat, theta)
    if not bool(torch.all(weights > 0) and torch.all(inverse_weights < 0)):
        return False
    if not bool(weights.sum() < 1 and (weights / theta).sum() < 1):
        return False
    if len(theta) > 1:
        differences = torch.abs(theta[:, None] - theta[None, :])
        differences = differences + torch.eye(len(theta), dtype=_DTYPE)
        if not bool(torch.min(differences) > 1e-12):
            return False
    return True


def _legacy_blt_loss_for_buffers(
    *,
    n: int,
    min_sep: int,
    max_participations: int,
    workload: torch.Tensor,
    num_buffers: int,
    max_optimizer_steps: int,
) -> float:
    if num_buffers == 0:
        coefficients = torch.zeros(n, dtype=_DTYPE)
        coefficients[0] = 1.0
        error = torch.cumsum(workload.square(), dim=0).max()
        sensitivity = _minsep_sensitivity_squared(
            coefficients,
            min_sep=min_sep,
            max_participations=max_participations,
        )
        return float(error * sensitivity)

    init_blt = get_init_blt(num_buffers=num_buffers)
    initial = Parameterization.buf_decay_pair().params_from_blt(init_blt)
    params = torch.nn.Parameter(torch.tensor(initial, dtype=_DTYPE))
    optimizer = torch.optim.LBFGS(
        [params],
        max_iter=max_optimizer_steps,
        tolerance_change=1e-15,
        tolerance_grad=1e-10,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        with torch.no_grad():
            params.clamp_(_EPS, 1.0 - _EPS)
        if not _feasible_blt_params(params):
            return params.sum() * 0 + 1e100
        objective = _blt_objective(
            params,
            n=n,
            min_sep=min_sep,
            max_participations=max_participations,
            workload=workload,
        )
        if not torch.isfinite(objective):
            raise RuntimeError("legacy BLT objective became non-finite")
        objective.backward()
        return objective

    try:
        optimizer.step(closure)
        return float(
            _blt_objective(
                params.detach(),
                n=n,
                min_sep=min_sep,
                max_participations=max_participations,
                workload=workload,
            )
        )
    except RuntimeError:
        return float("inf")


def legacy_blt_loss(
    *,
    n: int,
    min_sep: int,
    max_participations: int | None,
    momentum: float,
    max_buffers: int = 3,
    max_optimizer_steps: int = 600,
) -> float:
    """Return the legacy Torch planner's selected unpenalized BLT objective."""
    participations = minsep_true_max_participations(
        n=n, min_sep=min_sep, max_participations=max_participations
    )
    workload = torch.tensor(
        np.power(momentum, np.arange(n), dtype=np.float64), dtype=_DTYPE
    )
    previous_loss = _legacy_blt_loss_for_buffers(
        n=n,
        min_sep=min_sep,
        max_participations=participations,
        workload=workload,
        num_buffers=0,
        max_optimizer_steps=max_optimizer_steps,
    )
    for num_buffers in range(1, max_buffers + 1):
        current_loss = _legacy_blt_loss_for_buffers(
            n=n,
            min_sep=min_sep,
            max_participations=participations,
            workload=workload,
            num_buffers=num_buffers,
            max_optimizer_steps=max_optimizer_steps,
        )
        if 1.01 * current_loss < previous_loss:
            previous_loss = current_loss
        else:
            break
    return previous_loss
