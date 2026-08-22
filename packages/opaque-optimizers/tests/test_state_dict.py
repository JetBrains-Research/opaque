"""Tests for :mod:`opaque.serialization` on optimizer state.

Round-trip coverage for every optimizer + the schedule-free wrapper.
The contract: after serialise → fresh init → deserialise, the next
``step()`` call must produce *bit-identical* updates to the ones the
retained state would have produced.

:func:`_assert_round_trip` enforces two properties for every test: the
comparison is exact (``rtol=atol=0``), and the restored state demonstrably
influences the update, checked against a freshly-initialised one.  The second
depends on :func:`_grad_sequence` actually evolving state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from opaque.optimizers import (
    adafactor,
    adagrad,
    adamw,
    ademamix,
    apply_updates,
    lion,
    radam,
    rmsprop,
    schedule_free,
)

if TYPE_CHECKING:
    from collections.abc import Callable
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
    SecondMomentNoiseOutput,
    noised,
)


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _grad_sequence(params, steps: int = 6, *, seed: int = 20260819):
    """A deterministic, non-constant gradient sequence that evolves state.

    Magnitude must vary: under a constant gradient a bias-corrected optimizer
    is step-invariant (Adam has ``m̂ = g``, ``v̂ = g²`` exactly), so a fresh
    state gives the same update as an evolved one.  Direction alone is not
    enough either — Lion's ``sign(β₁m+(1−β₁)g)`` is identical from either
    state when the last gradient dominates the momentum, so magnitudes ramp
    *down* 4x → 1x.  Margins run 0.73 (Lion) to 1.1e+02 (RAdam); see
    ``_MIN_STATE_SENSITIVITY``.

    The explicit generator keeps the sequence off global RNG state.
    """
    gen = torch.Generator().manual_seed(seed)
    span = max(steps - 1, 1)
    return [
        {
            # Drawn on CPU (where ``gen`` lives) and moved onto the leaf, so
            # this keeps the device-following behavior of ``randn_like``.
            k: torch.randn(v.shape, generator=gen, dtype=v.dtype).to(v.device)
            * (1.0 + 3.0 * (span - i) / span)
            for k, v in params.items()
        }
        for i in range(steps)
    ]


@pytest.fixture
def grad_seq(params):
    return _grad_sequence(params)


# Minimum relative gap between an evolved-state update and a fresh-state one
# for a round-trip comparison to carry information.  Measured margins across
# the fixtures in this module run from 0.73 (Lion) to 1.1e+02 (RAdam), leaving
# ≥70x headroom, while a fixture that stops evolving state fails loudly.
_MIN_STATE_SENSITIVITY = 1e-2


def _rel_gap(a, b) -> float:
    """Relative L2 distance between two flat update pytrees."""
    num = sum(((a[k] - b[k]) ** 2).sum() for k in a).sqrt()
    den = sum((a[k] ** 2).sum() for k in a).sqrt()
    return (num / den).item()


def _round_trip(
    factory: Callable[..., tuple[Callable[..., tuple[Any, Any]], Any]],
    params,
    seq,
    *,
    wrap=None,
    **factory_kwargs,
):
    """Evolve state over ``seq[:-1]``, serialise, restore on a fresh init.

    Returns the next update from the evolved, restored, and freshly
    initialised state.  :func:`_assert_round_trip` compares the first two and
    uses the third to confirm the comparison is informative.
    """
    wrap = wrap or (lambda g: g)
    step, state = factory(params, **factory_kwargs)
    for step_grads in seq[:-1]:
        _, state = step(wrap(step_grads), state, params=params)

    # Fresh template — same shape, zeroed leaves.
    _step2, template = factory(params, **factory_kwargs)
    restored = from_state_dict(template, state_dict(state))

    final = wrap(seq[-1])
    u_evolved, _ = step(final, state, params=params)
    u_restored, _ = step(final, restored, params=params)
    _step3, fresh = factory(params, **factory_kwargs)
    u_fresh, _ = step(final, fresh, params=params)
    return u_evolved, u_restored, u_fresh


def _assert_round_trip(u_evolved, u_restored, u_fresh):
    """Assert an exact restore, and that the comparison could have failed.

    ``from_state_dict`` puts the saved values back into the template, so the
    follow-up update runs the same ops on the same bits: ``rtol=atol=0`` is
    the contract.  At ``lr=1e-3`` the updates are ~1e-3, where the default
    float32 ``atol=1e-5`` alone would absorb a genuine state difference.
    """
    for k in u_evolved:
        torch.testing.assert_close(u_restored[k], u_evolved[k], rtol=0, atol=0)

    gap = _rel_gap(u_evolved, u_fresh)
    assert gap > _MIN_STATE_SENSITIVITY, (
        f"round-trip assertion is vacuous: a freshly-initialised state produces "
        f"the same update as the evolved one (relative gap {gap:.2e} <= "
        f"{_MIN_STATE_SENSITIVITY:.0e}), so this test cannot detect a broken "
        f"restore.  The gradient sequence is not evolving optimizer state."
    )


def _noised(sigma):
    """``wrap`` for :func:`_round_trip` that attaches DP noise metadata."""
    return lambda g: noised(g, max_norm=1.0, noise_stddev=sigma)


def _second_moment(sigma=0.1):
    """``wrap`` that also supplies a privatised ``g²`` stream."""

    def wrap(g):
        sq = {k: v.pow(2) + 0.01 for k, v in g.items()}
        return SecondMomentNoiseOutput(
            noised(g, max_norm=1.0, noise_stddev=sigma),
            noised(sq, max_norm=1.0, noise_stddev=sigma),
        )

    return wrap


class TestAdamW:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(adamw, params, grad_seq, lr=1e-3, weight_decay=0.01)
        )

    def test_round_trip_bc(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(
                adamw,
                params,
                grad_seq,
                wrap=_noised(0.5),
                lr=1e-3,
                noise_bias_correction=True,
            )
        )

    def test_round_trip_second_moment(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(adamw, params, grad_seq, wrap=_second_moment(), lr=1e-3)
        )

    def test_round_trip_l2_wd(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(
                adamw,
                params,
                grad_seq,
                lr=1e-3,
                weight_decay=0.5,
                decoupled_weight_decay=False,
            )
        )

    def test_round_trip_with_rms_clip(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(adamw, params, grad_seq, lr=1e-3, update_rms_clip=0.5)
        )

    def test_step_and_phi_preserved(self, params, grads):
        step, state = adamw(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(7):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        _s2, template = adamw(params, lr=1e-3, noise_bias_correction=True)
        restored = from_state_dict(template, sd)
        assert restored.step == 7
        assert isinstance(state.phi, dict)
        assert all(v != pytest.approx(0.0) for v in state.phi.values())
        assert restored.phi == pytest.approx(state.phi)

    def test_per_group_phi_round_trip_nested(self):
        """Path-keyed φ survives state_dict when BC is enabled from init."""
        from opaque.types import PerGroup

        nested_params = {
            "layer1": {
                "weight": torch.randn(4, 3),
                "bias": torch.randn(4),
            },
            "layer2": {"weight": torch.randn(2, 4)},
        }
        nested_grads = {
            "layer1": {
                "weight": torch.randn_like(nested_params["layer1"]["weight"]),
                "bias": torch.randn_like(nested_params["layer1"]["bias"]),
            },
            "layer2": {"weight": torch.randn_like(nested_params["layer2"]["weight"])},
        }
        pg = PerGroup(
            groups={
                ("layer1", "weight"): "g_a",
                ("layer1", "bias"): "g_a",
                ("layer2", "weight"): "g_b",
            },
            values={"g_a": 0.2, "g_b": 0.7},
        )
        step, state = adamw(nested_params, lr=1e-3, noise_bias_correction=True)
        assert isinstance(state.phi, dict)
        assert set(state.phi) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        for _ in range(3):
            _, state = step(
                noised(nested_grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=nested_params,
            )
        assert state.phi[("layer1", "weight")] != pytest.approx(0.0)
        sd = state_dict(state)
        _s2, template = adamw(nested_params, lr=1e-3, noise_bias_correction=True)
        restored = from_state_dict(template, sd)
        assert restored.phi == state.phi
        u_orig, _ = step(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=nested_params,
        )
        u_rest, _ = step(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            restored,
            params=nested_params,
        )
        torch.testing.assert_close(
            u_orig["layer1"]["weight"], u_rest["layer1"]["weight"]
        )
        torch.testing.assert_close(
            u_orig["layer2"]["weight"], u_rest["layer2"]["weight"]
        )

    def test_torch_save_load_round_trip(self, params, grad_seq, tmp_path):
        step, state = adamw(params, lr=1e-3, weight_decay=0.01)
        for step_grads in grad_seq[:-1]:
            _, state = step(step_grads, state, params=params)
        sd = state_dict(state)
        path = tmp_path / "opt.pt"
        torch.save(sd, path)
        sd_loaded = torch.load(path, weights_only=False)
        assert set(sd_loaded.keys()) == set(sd.keys())
        _s2, template = adamw(params, lr=1e-3, weight_decay=0.01)
        restored = from_state_dict(template, sd_loaded)

        final = grad_seq[-1]
        u_orig, _ = step(final, state, params=params)
        u_rest, _ = step(final, restored, params=params)
        _s3, fresh = adamw(params, lr=1e-3, weight_decay=0.01)
        u_fresh, _ = step(final, fresh, params=params)
        _assert_round_trip(u_orig, u_rest, u_fresh)


class TestLion:
    def test_round_trip(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(lion, params, grad_seq, lr=1e-4, weight_decay=0.0)
        )

    def test_step_preserved(self, params, grads):
        step, state = lion(params, lr=1e-4)
        for _ in range(4):
            _, state = step(grads, state, params=params)
        sd = state_dict(state)
        _s2, template = lion(params, lr=1e-4)
        restored = from_state_dict(template, sd)
        assert restored.step == 4


class TestAdEMAMix:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(ademamix, params, grad_seq, lr=1e-3))

    def test_round_trip_bc(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(
                ademamix,
                params,
                grad_seq,
                wrap=_noised(0.4),
                lr=1e-3,
                noise_bias_correction=True,
            )
        )

    def test_phi_preserved(self, params, grads):
        step, state = ademamix(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(7):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=0.4),
                state,
                params=params,
            )
        sd = state_dict(state)
        _s2, template = ademamix(params, lr=1e-3, noise_bias_correction=True)
        restored = from_state_dict(template, sd)
        assert restored.step == 7
        assert isinstance(state.phi, dict)
        assert all(v != pytest.approx(0.0) for v in state.phi.values())
        assert restored.phi == pytest.approx(state.phi)


class TestAdafactor:
    @pytest.fixture
    def matrix_params(self):
        torch.manual_seed(0)
        return {"fc.weight": torch.randn(8, 4), "bias": torch.randn(4)}

    @pytest.fixture
    def matrix_grads(self, matrix_params):
        torch.manual_seed(1)
        return {k: torch.randn_like(v) for k, v in matrix_params.items()}

    @pytest.fixture
    def matrix_grad_seq(self, matrix_params):
        return _grad_sequence(matrix_params, 4)

    def test_round_trip(self, matrix_params, matrix_grad_seq):
        _assert_round_trip(
            *_round_trip(adafactor, matrix_params, matrix_grad_seq, lr=1e-3, beta1=0.9)
        )

    def test_factored_v_serialised(self, matrix_params, matrix_grads):
        """v_row / v_col tensors round-trip; the optree treespec is
        skipped from the saved dict (re-derived from the template)."""
        step, state = adafactor(matrix_params, lr=1e-3)
        _, state = step(matrix_grads, state, params=matrix_params)
        sd = state_dict(state)
        # treespec is opaque; should not appear in the saved dict.
        assert not any("treespec" in k for k in sd)
        # v_flat tensors should be there.
        assert any("v_flat" in k for k in sd)

    def test_per_group_phi_flat_round_trip(self, matrix_params, matrix_grads):
        """Adafactor phi_flat + paths round-trip under PerGroup BC."""
        from opaque.types import PerGroup

        pg = PerGroup(
            groups={
                ("fc.weight",): "attn",
                ("bias",): "mlp",
            },
            values={"attn": 0.2, "mlp": 0.8},
        )
        step, state = adafactor(
            matrix_params, lr=1e-3, beta1=0.9, noise_bias_correction=True
        )
        for _ in range(3):
            _, state = step(
                noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=matrix_params,
            )
        sd = state_dict(state)
        _s2, template = adafactor(
            matrix_params, lr=1e-3, beta1=0.9, noise_bias_correction=True
        )
        restored = from_state_dict(template, sd)
        assert restored.phi_flat == state.phi_flat
        assert restored.paths == state.paths
        u_orig, _ = step(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=matrix_params,
        )
        u_rest, _ = step(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            restored,
            params=matrix_params,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])


class TestRAdam:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(radam, params, grad_seq, lr=1e-3))

    def test_round_trip_bc(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(
                radam,
                params,
                grad_seq,
                wrap=_noised(0.3),
                lr=1e-3,
                noise_bias_correction=True,
            )
        )

    def test_phi_preserved(self, params, grads):
        step, state = radam(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(7):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        _s2, template = radam(params, lr=1e-3, noise_bias_correction=True)
        restored = from_state_dict(template, sd)
        assert restored.step == 7
        assert isinstance(state.phi, dict)
        assert all(v != pytest.approx(0.0) for v in state.phi.values())
        assert restored.phi == pytest.approx(state.phi)


class TestRMSprop:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(rmsprop, params, grad_seq, lr=1e-2))

    def test_round_trip_bc(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(
                rmsprop,
                params,
                grad_seq,
                wrap=_noised(0.3),
                lr=1e-2,
                noise_bias_correction=True,
            )
        )

    def test_phi_preserved(self, params, grads):
        step, state = rmsprop(params, lr=1e-2, noise_bias_correction=True)
        for _ in range(5):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        _s2, template = rmsprop(params, lr=1e-2, noise_bias_correction=True)
        restored = from_state_dict(template, sd)
        assert restored.step == 5
        assert isinstance(state.phi, dict)
        assert all(v != pytest.approx(0.0) for v in state.phi.values())
        assert restored.phi == pytest.approx(state.phi)


class TestAdagrad:
    def test_round_trip_vanilla(self, params, grad_seq):
        _assert_round_trip(*_round_trip(adagrad, params, grad_seq, lr=1e-2))

    def test_round_trip_bc(self, params, grad_seq):
        _assert_round_trip(
            *_round_trip(
                adagrad,
                params,
                grad_seq,
                wrap=_noised(0.3),
                lr=1e-2,
                noise_bias_correction=True,
            )
        )

    def test_phi_acc_preserved(self, params, grads):
        step, state = adagrad(params, lr=1e-2, noise_bias_correction=True)
        for _ in range(5):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        _s2, template = adagrad(params, lr=1e-2, noise_bias_correction=True)
        restored = from_state_dict(template, sd)
        assert restored.step == 5
        assert isinstance(state.phi_acc, dict)
        assert all(v != pytest.approx(0.0) for v in state.phi_acc.values())
        assert restored.phi_acc == pytest.approx(state.phi_acc)


class TestScheduleFree:
    def test_round_trip_over_adamw(self, params, grad_seq):
        step, state = schedule_free(params, adamw, lr=1e-3)
        p = params
        for step_grads in grad_seq[:-1]:
            delta, state = step(step_grads, state, params=p)
            p = apply_updates(p, delta)
        sd = state_dict(state)
        _s2, template = schedule_free(p, adamw, lr=1e-3)
        restored = from_state_dict(template, sd)
        # x and z must come back bit-for-bit, not merely close.
        for k in state.x:
            torch.testing.assert_close(restored.x[k], state.x[k], rtol=0, atol=0)
            torch.testing.assert_close(restored.z[k], state.z[k], rtol=0, atol=0)
        assert restored.step == state.step
        assert restored.beta == state.beta

        final = grad_seq[-1]
        u_evolved, _ = step(final, state, params=p)
        u_restored, _ = step(final, restored, params=p)
        _s3, fresh = schedule_free(p, adamw, lr=1e-3)
        u_fresh, _ = step(final, fresh, params=p)
        _assert_round_trip(u_evolved, u_restored, u_fresh)


class TestRobustness:
    def test_missing_path_keeps_template(self, params, grads):
        """Forward-compat: a saved dict missing a path keeps the
        template's value at that path."""
        step, state = adamw(params, lr=1e-3)
        _, state = step(grads, state, params=params)
        sd = state_dict(state)
        # Drop the step entry from the saved dict.
        sd_partial = {
            k: v for k, v in sd.items() if k != "step" and not k.endswith(".step")
        }
        _s2, template = adamw(params, lr=1e-3)
        restored = from_state_dict(template, sd_partial)
        # ``step`` falls back to template's 0.
        assert restored.step == 0

    def test_tensor_dtype_device_preserved(self, params, grads):
        """Saved tensors load back at the template's dtype/device."""
        step, state = adamw(params, lr=1e-3)
        _, state = step(grads, state, params=params)
        sd = state_dict(state)
        # Mutate the saved tensors to bf16 to simulate a saved
        # checkpoint at a different precision than the template.
        sd_bf16 = {
            k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
            for k, v in sd.items()
        }
        _s2, template = adamw(params, lr=1e-3)
        restored = from_state_dict(template, sd_bf16)
        # Restored tensors should match the template's dtype.
        assert restored.mu["weight"].dtype == template.mu["weight"].dtype

    def test_wrong_type_raises(self, params, grads):
        """A path that should hold a tensor but holds a non-tensor in
        the dict raises ``TypeError`` rather than silently corrupting."""
        step, state = adamw(params, lr=1e-3)
        _, state = step(grads, state, params=params)
        sd = state_dict(state)
        # Find a tensor key and replace its value with a string.
        tensor_key = next(k for k, v in sd.items() if isinstance(v, torch.Tensor))
        sd[tensor_key] = "not a tensor"
        _s2, template = adamw(params, lr=1e-3)
        with pytest.raises(TypeError, match=r"torch.Tensor"):
            from_state_dict(template, sd)
