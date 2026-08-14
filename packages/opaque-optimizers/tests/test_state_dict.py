"""Tests for :mod:`opaque.serialization` on optimizer state.

Round-trip coverage for every optimizer + the schedule-free wrapper.
The contract: after serialise → fresh init → deserialise, the next
``step()`` call must produce the same updates as if we had kept
the original state.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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


def _round_trip(
    factory: Callable[..., tuple[Callable[..., tuple[Any, Any]], Any]],
    params,
    grads,
    steps: int = 5,
    **factory_kwargs,
):
    """Train ``steps`` steps, serialise, restore on a fresh init,
    assert the next update is identical."""
    step, state = factory(params, **factory_kwargs)
    for _ in range(steps):
        _, state = step(grads, state, params=params)
    sd = state_dict(state)

    # Fresh template — same shape, zeroed leaves.
    _step2, template = factory(params, **factory_kwargs)
    restored = from_state_dict(template, sd)

    # Both states should produce the same next update.
    u_orig, _ = step(grads, state, params=params)
    u_rest, _ = step(grads, restored, params=params)
    return u_orig, u_rest


class TestAdamW:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(
            adamw, params, grads, lr=1e-3, weight_decay=0.01
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        u_orig, u_rest = _round_trip(
            adamw,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            lr=1e-3,
            noise_bias_correction=True,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_second_moment(self, params, grads):
        sq = {k: v.pow(2) + 0.01 for k, v in grads.items()}
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        u_orig, u_rest = _round_trip(adamw, params, output, lr=1e-3)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_l2_wd(self, params, grads):
        u_orig, u_rest = _round_trip(
            adamw,
            params,
            grads,
            lr=1e-3,
            weight_decay=0.5,
            decoupled_weight_decay=False,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_with_rms_clip(self, params, grads):
        u_orig, u_rest = _round_trip(
            adamw, params, grads, lr=1e-3, update_rms_clip=0.5
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

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

    def test_torch_save_load_round_trip(self, params, grads, tmp_path):
        step, state = adamw(params, lr=1e-3, weight_decay=0.01)
        for _ in range(3):
            _, state = step(grads, state, params=params)
        sd = state_dict(state)
        path = tmp_path / "opt.pt"
        torch.save(sd, path)
        sd_loaded = torch.load(path, weights_only=False)
        assert set(sd_loaded.keys()) == set(sd.keys())
        _s2, template = adamw(params, lr=1e-3, weight_decay=0.01)
        restored = from_state_dict(template, sd_loaded)
        u_orig, _ = step(grads, state, params=params)
        u_rest, _ = step(grads, restored, params=params)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])


class TestLion:
    def test_round_trip(self, params, grads):
        u_orig, u_rest = _round_trip(
            lion, params, grads, lr=1e-4, weight_decay=0.0
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_step_preserved(self, params, grads):
        step, state = lion(params, lr=1e-4)
        for _ in range(4):
            _, state = step(grads, state, params=params)
        sd = state_dict(state)
        _s2, template = lion(params, lr=1e-4)
        restored = from_state_dict(template, sd)
        assert restored.step == 4


class TestAdEMAMix:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(ademamix, params, grads, lr=1e-3)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        u_orig, u_rest = _round_trip(
            ademamix,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.4),
            lr=1e-3,
            noise_bias_correction=True,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

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

    def test_round_trip(self, matrix_params, matrix_grads):
        u_orig, u_rest = _round_trip(
            adafactor, matrix_params, matrix_grads, steps=3, lr=1e-3, beta1=0.9
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

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
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(radam, params, grads, lr=1e-3)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        u_orig, u_rest = _round_trip(
            radam,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.3),
            lr=1e-3,
            noise_bias_correction=True,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

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
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(rmsprop, params, grads, lr=1e-2)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        u_orig, u_rest = _round_trip(
            rmsprop,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.3),
            lr=1e-2,
            noise_bias_correction=True,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

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
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(adagrad, params, grads, lr=1e-2)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        u_orig, u_rest = _round_trip(
            adagrad,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.3),
            lr=1e-2,
            noise_bias_correction=True,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

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
    def test_round_trip_over_adamw(self, params, grads):
        step, state = schedule_free(params, adamw, lr=1e-3)
        p = params
        for _ in range(4):
            delta, state = step(grads, state, params=p)
            p = apply_updates(p, delta)
        sd = state_dict(state)
        _s2, template = schedule_free(p, adamw, lr=1e-3)
        restored = from_state_dict(template, sd)
        for k in state.x:
            torch.testing.assert_close(restored.x[k], state.x[k])
            torch.testing.assert_close(restored.z[k], state.z[k])
        assert restored.step == state.step
        assert restored.beta == state.beta
        ep = restored.x
        for k in ep:
            torch.testing.assert_close(ep[k], state.x[k])


class TestRobustness:
    def test_missing_path_keeps_template(self, params, grads):
        """Forward-compat: a saved dict missing a path keeps the
        template's value at that path."""
        step, state = adamw(params, lr=1e-3)
        _, state = step(grads, state, params=params)
        sd = state_dict(state)
        # Drop the step entry from the saved dict.
        sd_partial = {k: v for k, v in sd.items() if k != "step" and not k.endswith(".step")}
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
