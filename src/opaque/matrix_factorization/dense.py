"""Optimization and error functions for dense (explicitly represented) strategies.

See ``sensitivity.py`` for sensitivity calculations.

References:
    - Denisov et al., 2022: https://arxiv.org/abs/2202.08312
    - Choquette-Choo et al., 2022: https://arxiv.org/abs/2211.06530
"""

from __future__ import annotations

import numpy as np
import torch

from . import checks, optimization, sensitivity


def per_query_error(
    *,
    strategy_matrix: torch.Tensor | None = None,
    noising_matrix: torch.Tensor | None = None,
    workload_matrix: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Expected per-query squared error for a general matrix mechanism.

    Exactly one of ``strategy_matrix`` and ``noising_matrix`` must be provided.

    Args:
        strategy_matrix: The square strategy matrix C.
        noising_matrix: The (possibly non-square) noising matrix C^{-1}.
        workload_matrix: The workload matrix. Defaults to tri(n) (prefix sum).
        skip_checks: If True, skip input verification.

    Returns:
        Per-query expected squared error, a tensor of length n.
    """
    if not skip_checks:
        if (strategy_matrix is None) == (noising_matrix is None):
            raise ValueError(
                "Specify exactly one of strategy_matrix or noising_matrix."
            )

    if strategy_matrix is not None:
        C = strategy_matrix
        if not skip_checks:
            checks.check_square(C, "strategy_matrix")
        n = C.shape[1]
        A = (
            workload_matrix
            if workload_matrix is not None
            else torch.tril(torch.ones(n, n, dtype=C.dtype))
        )
        # Solve B @ C = A for B  =>  B = A @ C^{-T} = solve(C.T, A.T).T
        B = torch.linalg.solve(C.T, A.T).T
        if not skip_checks:
            checks.check(A=A, B=B, C=C)
    else:
        assert noising_matrix is not None
        C_inv = noising_matrix
        n = C_inv.shape[0]
        A = (
            workload_matrix
            if workload_matrix is not None
            else torch.tril(torch.ones(n, n, dtype=C_inv.dtype))
        )
        B = A @ C_inv
        if not skip_checks:
            checks.check(A=A, B=B)

    return torch.sum(B * B, dim=1)


def max_error(
    *,
    strategy_matrix: torch.Tensor | None = None,
    noising_matrix: torch.Tensor | None = None,
    workload_matrix: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Max-over-iterations squared error for a general matrix mechanism."""
    return per_query_error(
        strategy_matrix=strategy_matrix,
        noising_matrix=noising_matrix,
        workload_matrix=workload_matrix,
        skip_checks=skip_checks,
    ).max()


def mean_error(
    *,
    strategy_matrix: torch.Tensor | None = None,
    noising_matrix: torch.Tensor | None = None,
    workload_matrix: torch.Tensor | None = None,
    skip_checks: bool = False,
) -> torch.Tensor:
    """Mean-over-iterations squared error for a general matrix mechanism."""
    return per_query_error(
        strategy_matrix=strategy_matrix,
        noising_matrix=noising_matrix,
        workload_matrix=workload_matrix,
        skip_checks=skip_checks,
    ).mean()


def get_orthogonal_mask(n: int, epochs: int = 1) -> torch.Tensor:
    """Compute a mask that imposes orthogonality constraints.

    Specific to the (k, b)-fixed-epoch-order participation schema.

    Args:
        n: Size of the mask.
        epochs: Number of epochs.

    Returns:
        An n x n 0/1 mask tensor.
    """
    mask = np.ones((n, n))
    b = n // epochs
    for i in range(b):
        mask[i::b, i::b] = np.eye(epochs)
    return torch.tensor(mask, dtype=torch.float64)


def strategy_from_X(X: torch.Tensor) -> torch.Tensor:
    """Return a lower-triangular strategy matrix C from its Gram matrix X.

    Uses Cholesky decomposition with a reversal trick to get the correct
    lower-triangular form.

    Args:
        X: A positive symmetric semi-definite matrix (X = C.T @ C).

    Returns:
        Lower-triangular matrix C satisfying X = C.T @ C.
    """
    # Reverse, Cholesky, reverse back
    X_rev = X.flip(0, 1)
    L = torch.linalg.cholesky(X_rev)
    return L.T.flip(0, 1)


def optimize(
    n: int,
    *,
    epochs: int = 1,
    bands: int | None = None,
    equal_norm: bool = False,
    A: torch.Tensor | None = None,
    max_optimizer_steps: int = 10000,
    callback: optimization.CallbackFnType = lambda _: True
    if _.grad is not None and float(torch.abs(_.grad).max()) <= 1e-3
    else None,
) -> torch.Tensor:
    """Optimize a strategy matrix C for mean loss and a participation pattern.

    Supports:
    - Single participation (default)
    - Multi-participation with fixed-epoch order (epochs=k)
    - Multi-participation with min-separation (bands=min_sep, equal_norm=True)

    Args:
        n: Number of iterations.
        epochs: Number of epochs.
        bands: Number of bands in the strategy.
        equal_norm: If True, each column of C should have equal norm.
        A: Workload matrix (defaults to prefix sum).
        max_optimizer_steps: Maximum L-BFGS steps.
        callback: Optional callback for monitoring/early stopping.

    Returns:
        The optimized strategy matrix C.
    """
    if A is None:
        A = torch.tril(torch.ones(n, n, dtype=torch.float64))

    mask = get_orthogonal_mask(n, epochs)
    if bands is not None:
        mask = mask * sensitivity.banded_symmetric_mask(n, bands).double()

    def loss_and_projected_grad(X_flat):
        X = X_flat.reshape(n, n)
        # Ensure symmetry for the loss computation
        X = (X + X.T) / 2

        # Compute loss: tr[A^T A X^{-1}] / n
        try:
            H = torch.linalg.solve(X, A.T)
        except torch.linalg.LinAlgError:
            return torch.tensor(float("inf"), dtype=torch.float64), torch.zeros_like(
                X_flat
            )

        loss = torch.trace(H @ A) / n
        # Gradient: -X^{-1} A^T A X^{-1} / n  =  -H @ H.T / n
        dX = -H @ H.T / n

        if equal_norm:
            diag = torch.zeros(n, dtype=torch.float64)
        else:
            dX_diag = torch.diag(dX)
            dsum = dX_diag.reshape(epochs, -1).sum(dim=0) / epochs
            diag = dX_diag - torch.kron(torch.ones(epochs, dtype=torch.float64), dsum)

        dX = dX.clone()
        dX[range(n), range(n)] = diag
        dX = dX * mask

        return loss, dX.reshape(-1)

    X_init = (torch.eye(n, dtype=torch.float64) / epochs).reshape(-1)

    X_flat = optimization.optimize(
        loss_and_projected_grad,
        X_init,
        max_optimizer_steps=max_optimizer_steps,
        grad=True,
        callback=callback,
    )

    X = X_flat.reshape(n, n).double()
    X = (X + X.T) / 2  # Ensure symmetry
    return strategy_from_X(X)
