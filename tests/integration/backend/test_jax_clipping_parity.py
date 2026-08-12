"""FP32 parity of the neutral fixed/AUTO-S clipping path on Torch and JAX."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from tests.integration.backend._harness import HostBridge, run_clipping

from opaque import pytree
from opaque.api.engine.backend import use_backend
from opaque.api.engine.clipping import clipped_grad
from opaque.jax import jax_backend
from opaque.torch import torch_backend

jnp = pytest.importorskip("jax.numpy")

_BOUND = 0.75
_GAMMA = 0.05
_RTOL = 2e-5
_ATOL = 2e-5


def _torch_to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _jax_to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


TORCH = HostBridge(to_numpy=_torch_to_numpy)
JAX = HostBridge(to_numpy=_jax_to_numpy)


def _torch_logistic_loss(params: Any, x: torch.Tensor, y: torch.Tensor) -> Any:
    logits = x @ params["w"] + params["b"]
    return torch.nn.functional.softplus(logits) - y * logits, {"logits": logits}


def _jax_logistic_loss(params: Any, x: Any, y: Any) -> Any:
    logits = x @ params["w"] + params["b"]
    return jnp.logaddexp(0.0, logits) - y * logits, {"logits": logits}


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


def _make_jax_inputs() -> tuple[dict[str, Any], Any, Any]:
    params, x, y = _make_torch_inputs()
    return (
        {name: jnp.asarray(value.numpy()) for name, value in params.items()},
        jnp.asarray(x.numpy()),
        jnp.asarray(y.numpy()),
    )


def _assert_tree_close(
    torch_tree: Any,
    jax_tree: Any,
    *,
    comparison_backend: Any | None = None,
) -> None:
    with use_backend(torch_backend()):
        torch_paths, torch_leaves, _ = pytree.tree_flatten_with_paths(torch_tree)
    with use_backend(comparison_backend or jax_backend()):
        jax_paths, jax_leaves, _ = pytree.tree_flatten_with_paths(jax_tree)

    assert jax_paths == torch_paths
    for torch_leaf, jax_leaf in zip(torch_leaves, jax_leaves, strict=True):
        np.testing.assert_allclose(
            TORCH.to_numpy(torch_leaf),
            JAX.to_numpy(jax_leaf),
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
    jax_params, jax_x, jax_y = _make_jax_inputs()

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
    jax_run = run_clipping(
        jax_backend(),
        _jax_logistic_loss,
        jax_params,
        jax_x,
        jax_y,
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )

    assert TORCH.to_numpy(torch_run.values).shape == (4,)
    assert JAX.to_numpy(jax_run.values).shape == (4,)
    assert TORCH.to_numpy(torch_run.value_aux["logits"]).shape == (4,)
    assert JAX.to_numpy(jax_run.value_aux["logits"]).shape == (4,)
    _assert_finite_and_bounded(torch_run.clipped_norms, TORCH)
    _assert_finite_and_bounded(jax_run.clipped_norms, JAX)
    _assert_tree_close(torch_run.summed_grads, jax_run.summed_grads)


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
    jax_params, _, _ = _make_jax_inputs()
    nonfinite_x = np.array(
        [[1.0, -1.0], [np.nan, 0.5], [np.inf, -2.0]], dtype=np.float32
    )
    targets = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def torch_loss(params: Any, x: torch.Tensor, y: torch.Tensor) -> Any:
        prediction = x @ params["w"] + params["b"]
        return (prediction - y) ** 2, {"prediction": prediction}

    def jax_loss(params: Any, x: Any, y: Any) -> Any:
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
    jax_run = run_clipping(
        jax_backend(),
        jax_loss,
        jax_params,
        jnp.asarray(nonfinite_x),
        jnp.asarray(targets),
        kind=kind,
        bound=_BOUND,
        gamma=_GAMMA,
    )

    _assert_finite_and_bounded(torch_run.clipped_norms, TORCH)
    _assert_finite_and_bounded(jax_run.clipped_norms, JAX)
    _assert_tree_close(torch_run.clipped_grads, jax_run.clipped_grads)
