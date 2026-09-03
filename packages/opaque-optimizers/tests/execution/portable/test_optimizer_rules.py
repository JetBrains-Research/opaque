"""Portable numerical coverage for backend-neutral optimizer rules."""

from __future__ import annotations

from opaque.optimizers import adamw, apply_updates, sgd


def test_sgd_applies_l2_decay_and_preserves_momentum_state(backend_case) -> None:
    params = {
        "weight": backend_case.array([1.0, -2.0], dtype=backend_case.dtype("float32"))
    }
    gradients = {
        "weight": backend_case.array([0.5, -0.25], dtype=backend_case.dtype("float32"))
    }
    step, state = sgd(params, lr=0.1, momentum=0.5, weight_decay=0.1)

    updates, state = step(gradients, state, params=params)

    backend_case.assert_allclose(updates["weight"], [-0.06, 0.045])
    backend_case.assert_allclose(
        apply_updates(params, updates)["weight"], [0.94, -1.955]
    )
    assert state.step == 1
    backend_case.assert_allclose(state.momentum["weight"], [0.6, -0.45])


def test_adamw_decouples_weight_decay_on_portable_arrays(backend_case) -> None:
    params = {
        "weight": backend_case.array([1.0, -2.0], dtype=backend_case.dtype("float32"))
    }
    zero_gradients = {
        "weight": backend_case.array([0.0, 0.0], dtype=backend_case.dtype("float32"))
    }
    step, state = adamw(params, lr=0.1, weight_decay=0.1)

    updates, state = step(zero_gradients, state, params=params)

    backend_case.assert_allclose(updates["weight"], [-0.01, 0.02])
    backend_case.assert_allclose(
        apply_updates(params, updates)["weight"], [0.99, -1.98]
    )
    assert state.step == 1
