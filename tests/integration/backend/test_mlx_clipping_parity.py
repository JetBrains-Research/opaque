"""FP32 parity of the neutral fixed/AUTO-S clipping path on Torch and MLX."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from tests.integration.backend._harness import HostBridge, run_clipping

from opaque import pytree
from opaque.api.engine.backend import use_backend
from opaque.api.engine.clipping import clipped_grad
from opaque.mlx import mlx_backend
from opaque.torch import torch_backend

mx = pytest.importorskip("mlx.core")

_BOUND = 0.75
_GAMMA = 0.05
_RTOL = 2e-5
_ATOL = 2e-5


def _torch_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _mlx_to_numpy(value: Any) -> np.ndarray:
    mx.eval(value)
    return np.asarray(value)


TORCH = HostBridge(to_numpy=_torch_to_numpy)
MLX = HostBridge(to_numpy=_mlx_to_numpy)


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
    *,
    comparison_backend: Any | None = None,
) -> None:
    with use_backend(torch_backend()):
        torch_paths, torch_leaves, _ = pytree.tree_flatten_with_paths(torch_tree)
    with use_backend(comparison_backend or mlx_backend()):
        mlx_paths, mlx_leaves, _ = pytree.tree_flatten_with_paths(mlx_tree)
    assert mlx_paths == torch_paths
    for torch_leaf, mlx_leaf in zip(torch_leaves, mlx_leaves, strict=True):
        np.testing.assert_allclose(
            TORCH.to_numpy(torch_leaf),
            MLX.to_numpy(mlx_leaf),
            rtol=_RTOL,
            atol=_ATOL,
        )


def _assert_finite_and_bounded(norms: Any, bridge: HostBridge) -> None:
    host_norms = bridge.to_numpy(norms)
    assert np.isfinite(host_norms).all()
    assert (host_norms <= _BOUND + _ATOL).all()


@pytest.mark.parametrize("kind", ["fixed", "auto"])
def test_neutral_clipping_matches_torch_in_fp32(kind: str) -> None:
    torch_params, torch_x, torch_y = _make_torch_inputs()
    mlx_params, mlx_x, mlx_y = _make_mlx_inputs()

    torch_run = run_clipping(
        torch_backend(),
        _torch_logistic_loss,
        torch_params,
        torch_x,
        torch_y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )
    mlx_run = run_clipping(
        mlx_backend(),
        _mlx_logistic_loss,
        mlx_params,
        mlx_x,
        mlx_y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )

    assert TORCH.to_numpy(torch_run.values).shape == (4,)
    assert MLX.to_numpy(mlx_run.values).shape == (4,)
    assert TORCH.to_numpy(torch_run.value_aux["logits"]).shape == (4,)
    assert MLX.to_numpy(mlx_run.value_aux["logits"]).shape == (4,)
    _assert_finite_and_bounded(torch_run.clipped_norms, TORCH)
    _assert_finite_and_bounded(mlx_run.clipped_norms, MLX)
    _assert_tree_close(torch_run.summed_grads, mlx_run.summed_grads)


def test_fixed_neutral_path_agrees_with_torch_clipped_grad_oracle() -> None:
    params, batch_x, batch_y = _make_torch_inputs()
    neutral_run = run_clipping(
        torch_backend(),
        _torch_logistic_loss,
        params,
        batch_x,
        batch_y,
        kind="fixed",
        bound=_BOUND,
    )
    oracle, state = clipped_grad(
        _torch_logistic_loss,
        has_aux=True,
        clipping_norm=_BOUND,
        batch_argnums=(1, 2),
    )
    oracle_grads, _ = oracle(params, batch_x, batch_y, state=state)

    _assert_tree_close(
        neutral_run.summed_grads,
        oracle_grads.pytree,
        comparison_backend=torch_backend(),
    )


@pytest.mark.parametrize("kind", ["fixed", "auto"])
def test_neutral_clipping_sanitizes_nonfinite_per_example_gradients(kind: str) -> None:
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
        torch_backend(),
        torch_loss,
        torch_params,
        torch.tensor(nonfinite_x),
        torch.tensor(targets),
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )
    mlx_run = run_clipping(
        mlx_backend(),
        mlx_loss,
        mlx_params,
        mx.array(nonfinite_x),
        mx.array(targets),
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )

    _assert_finite_and_bounded(torch_run.clipped_norms, TORCH)
    _assert_finite_and_bounded(mlx_run.clipped_norms, MLX)
    _assert_tree_close(torch_run.clipped_grads, mlx_run.clipped_grads)
