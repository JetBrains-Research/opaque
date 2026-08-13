"""Focused FP32 parity for public fixed/AUTO-S clipping on Torch and MLX."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from tests.integration.backend._harness import run_clipping
from tests.integration.backend._providers import provider_case

from opaque import pytree
from opaque.api.engine.backend import use_backend

mx = pytest.importorskip("mlx.core")

_BOUND = 0.75
_GAMMA = 0.05
_RTOL = 2e-5
_ATOL = 2e-5


TORCH = provider_case("torch")
MLX = provider_case("mlx")


def _torch_logistic_loss(params, x, y):
    logits = x @ params["w"] + params["b"]
    return torch.nn.functional.softplus(logits) - y * logits, {"logits": logits}


def _mlx_logistic_loss(params, x, y):
    logits = x @ params["w"] + params["b"]
    return mx.logaddexp(0.0, logits) - y * logits, {"logits": logits}


def _make_torch_inputs() -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    return (
        {
            "w": torch.tensor([0.25, -0.5], dtype=torch.float32),
            "b": torch.tensor(0.1, dtype=torch.float32),
        },
        torch.tensor(
            [[1.0, 2.0], [-2.0, 1.0], [0.5, -1.0], [3.0, -0.25]],
            dtype=torch.float32,
        ),
        torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32),
    )


def _make_mlx_inputs() -> tuple[dict[str, Any], Any, Any]:
    params, x, y = _make_torch_inputs()
    return (
        {name: mx.array(value.numpy()) for name, value in params.items()},
        mx.array(x.numpy()),
        mx.array(y.numpy()),
    )


def _assert_tree_close(
    torch_tree: Any,
    mlx_tree: Any,
) -> None:
    with use_backend(TORCH.backend):
        torch_paths, torch_leaves, _ = pytree.tree_flatten_with_paths(torch_tree)
    with use_backend(MLX.backend):
        mlx_paths, mlx_leaves, _ = pytree.tree_flatten_with_paths(mlx_tree)
    assert mlx_paths == torch_paths
    for torch_leaf, mlx_leaf in zip(torch_leaves, mlx_leaves, strict=True):
        np.testing.assert_allclose(
            TORCH.to_numpy(torch_leaf),
            MLX.to_numpy(mlx_leaf),
            rtol=_RTOL,
            atol=_ATOL,
        )


def _assert_finite_and_bounded(norms: Any, case: Any) -> None:
    host_norms = case.to_numpy(norms)
    assert np.isfinite(host_norms).all()
    assert (host_norms <= _BOUND + _ATOL).all()


@pytest.mark.parametrize("kind", ["fixed", "auto"])
def test_neutral_clipping_matches_torch_in_fp32(kind: str) -> None:
    torch_params, torch_x, torch_y = _make_torch_inputs()
    mlx_params, mlx_x, mlx_y = _make_mlx_inputs()

    torch_run = run_clipping(
        TORCH,
        _torch_logistic_loss,
        torch_params,
        torch_x,
        torch_y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        has_aux=True,
        return_aux=True,
    )
    mlx_run = run_clipping(
        MLX,
        _mlx_logistic_loss,
        mlx_params,
        mlx_x,
        mlx_y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        has_aux=True,
        return_aux=True,
    )

    assert TORCH.to_numpy(torch_run.aux.loss_values).shape == (4,)
    assert MLX.to_numpy(mlx_run.aux.loss_values).shape == (4,)
    assert TORCH.to_numpy(torch_run.aux.loss_aux["logits"]).shape == (4,)
    assert MLX.to_numpy(mlx_run.aux.loss_aux["logits"]).shape == (4,)
    _assert_finite_and_bounded(torch_run.aux.clipped_grad_norms, TORCH)
    _assert_finite_and_bounded(mlx_run.aux.clipped_grad_norms, MLX)
    _assert_tree_close(torch_run.grads.pytree, mlx_run.grads.pytree)


@pytest.mark.parametrize("kind", ["fixed", "auto"])
def test_public_clipping_sanitizes_nonfinite_per_example_gradients(kind: str) -> None:
    torch_params, _, _ = _make_torch_inputs()
    mlx_params, _, _ = _make_mlx_inputs()
    nonfinite_x = np.array(
        [[1.0, -1.0], [np.nan, 0.5], [np.inf, -2.0]], dtype=np.float32
    )
    targets = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def torch_loss(params, x, y):
        prediction = x @ params["w"] + params["b"]
        return (prediction - y) ** 2, {"prediction": prediction}

    def mlx_loss(params, x, y):
        prediction = x @ params["w"] + params["b"]
        return (prediction - y) ** 2, {"prediction": prediction}

    torch_run = run_clipping(
        TORCH,
        torch_loss,
        torch_params,
        torch.tensor(nonfinite_x),
        torch.tensor(targets),
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        has_aux=True,
        return_aux=True,
    )
    mlx_run = run_clipping(
        MLX,
        mlx_loss,
        mlx_params,
        mx.array(nonfinite_x),
        mx.array(targets),
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
        has_aux=True,
        return_aux=True,
    )

    _assert_finite_and_bounded(torch_run.aux.clipped_grad_norms, TORCH)
    _assert_finite_and_bounded(mlx_run.aux.clipped_grad_norms, MLX)
    _assert_tree_close(torch_run.grads.pytree, mlx_run.grads.pytree)
