"""Optimization wrapper for matrix factorization strategy optimization.

Uses scipy.optimize.minimize (L-BFGS-B) for optimizing strategy matrices,
which requires float64 precision for numerical stability.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar

import torch

ParamT = TypeVar("ParamT")
CallbackFnType: TypeAlias = Callable[["CallbackArgs"], None | bool]


@dataclasses.dataclass
class CallbackArgs:
    """Information passed to the callback on each optimization step.

    Attributes:
        step: Current optimization step number.
        loss: Loss value at current step.
        grad: Gradient at current step (if available).
        params: Current parameters.
        state: Current optimizer state.
    """

    step: int
    loss: torch.Tensor
    grad: Any | None
    params: Any
    state: Any


def optimize(
    loss_fn: Callable,
    params: torch.Tensor,
    *,
    max_optimizer_steps: int = 250,
    grad: bool = False,
    callback: CallbackFnType = lambda _: None,
) -> torch.Tensor:
    """Optimize a differentiable loss function using L-BFGS.

    This wrapper uses scipy's L-BFGS-B optimizer for robustness. Parameters
    are internally cast to float64 for numerical stability.

    Args:
        loss_fn: Loss function to minimize. If ``grad=True``, should return
            (loss, gradient) tuple.
        params: Initial parameters as a 1D tensor.
        max_optimizer_steps: Maximum number of L-BFGS iterations.
        grad: If True, loss_fn returns (loss, grad) tuple.
        callback: Optional callback called each iteration.

    Returns:
        Optimized parameters tensor.
    """
    from scipy.optimize import minimize

    original_dtype = params.dtype
    params_np = params.detach().double().numpy().copy()

    step_counter = [0]

    def scipy_loss(x):
        x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=not grad)

        if grad:
            loss_val, grad_val = loss_fn(x_tensor)
            loss_np = float(loss_val.detach())
            if isinstance(grad_val, torch.Tensor):
                grad_np = grad_val.detach().numpy().copy()
            else:
                grad_np = grad_val
        else:
            loss_val = loss_fn(x_tensor)
            loss_val.backward()
            loss_np = float(loss_val.detach())
            grad_np = x_tensor.grad.numpy().copy()

        # Call user callback
        cb_result = callback(
            CallbackArgs(
                step=step_counter[0],
                loss=torch.tensor(loss_np),
                grad=torch.tensor(grad_np) if grad_np is not None else None,
                params=x_tensor.detach(),
                state=None,
            )
        )
        step_counter[0] += 1

        if cb_result:
            # Signal early termination (scipy doesn't support this directly,
            # but we can set a flag)
            raise _EarlyStopException(x)

        return loss_np, grad_np.astype("float64")

    try:
        result = minimize(
            scipy_loss,
            params_np,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_optimizer_steps, "ftol": 1e-15, "gtol": 1e-10},
        )
        optimal_params = torch.tensor(result.x, dtype=original_dtype)
    except _EarlyStopException as e:
        optimal_params = torch.tensor(e.params, dtype=original_dtype)

    return optimal_params


class _EarlyStopException(Exception):
    """Internal exception for early stopping in scipy optimization."""

    def __init__(self, params):
        self.params = params
        super().__init__("Early stop")
