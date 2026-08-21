"""Tests for :mod:`opaque.serialization` on optimizer chain state.

Round-trip coverage for every optimizer + the schedule-free wrapper.
The contract: after serialise → fresh init → deserialise, the next
``update()`` call must produce the same updates as if we had kept
the original state.
"""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.optimizers import (
    adafactor,
    adagrad,
    adamw,
    ademamix,
    get_eval_params,
    get_train_params,
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


def _round_trip(opt, params, grads, steps: int = 5, **update_kwargs):
    """Train ``steps`` steps, serialise, restore on a fresh init,
    assert the next update is identical."""
    state = opt.init(params)
    for _ in range(steps):
        _, state = opt.update(grads, state, params=params, **update_kwargs)
    sd = state_dict(state)

    # Fresh template — same shape, zeroed leaves.
    template = opt.init(params)
    restored = from_state_dict(template, sd)

    # Both states should produce the same next update.
    u_orig, _ = opt.update(grads, state, params=params, **update_kwargs)
    u_rest, _ = opt.update(grads, restored, params=params, **update_kwargs)
    return u_orig, u_rest


class TestAdamW:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(adamw(lr=1e-3, weight_decay=0.01), params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        u_orig, u_rest = _round_trip(
            opt,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.5),
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_second_moment(self, params, grads):
        sq = {k: v.pow(2) + 0.01 for k, v in grads.items()}
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        opt = adamw(lr=1e-3)
        u_orig, u_rest = _round_trip(opt, params, output)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_l2_wd(self, params, grads):
        opt = adamw(lr=1e-3, weight_decay=0.5, decoupled_weight_decay=False)
        u_orig, u_rest = _round_trip(opt, params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_with_rms_clip(self, params, grads):
        opt = adamw(lr=1e-3, update_rms_clip=0.5)
        u_orig, u_rest = _round_trip(opt, params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_step_and_phi_preserved(self, params, grads):
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(7):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        # Adam state is at chain index 0 (decoupled WD).
        assert restored[0].step == 7
        assert isinstance(state[0].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi.values())
        assert restored[0].phi == pytest.approx(state[0].phi)

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
        opt = adamw(lr=1e-3, noise_bias_correction=True)
        state = opt.init(nested_params)
        assert isinstance(state[0].phi, dict)
        assert set(state[0].phi) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        for _ in range(3):
            _, state = opt.update(
                noised(nested_grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=nested_params,
            )
        assert state[0].phi[("layer1", "weight")] != pytest.approx(0.0)
        sd = state_dict(state)
        restored = from_state_dict(opt.init(nested_params), sd)
        assert restored[0].phi == state[0].phi
        u_orig, _ = opt.update(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=nested_params,
        )
        u_rest, _ = opt.update(
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
        opt = adamw(lr=1e-3, weight_decay=0.01)
        state = opt.init(params)
        for _ in range(3):
            _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        path = tmp_path / "opt.pt"
        torch.save(sd, path)
        sd_loaded = torch.load(path, weights_only=False)
        assert set(sd_loaded.keys()) == set(sd.keys())
        template = opt.init(params)
        restored = from_state_dict(template, sd_loaded)
        u_orig, _ = opt.update(grads, state, params=params)
        u_rest, _ = opt.update(grads, restored, params=params)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])


class TestLion:
    def test_round_trip(self, params, grads):
        u_orig, u_rest = _round_trip(lion(lr=1e-4, weight_decay=0.0), params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_step_preserved(self, params, grads):
        opt = lion(lr=1e-4)
        state = opt.init(params)
        for _ in range(4):
            _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        restored = from_state_dict(opt.init(params), sd)
        assert restored[0].step == 4


class TestAdEMAMix:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(ademamix(lr=1e-3), params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        opt = ademamix(lr=1e-3, noise_bias_correction=True)
        u_orig, u_rest = _round_trip(
            opt,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.4),
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_phi_preserved(self, params, grads):
        opt = ademamix(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(7):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.4),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        assert restored[0].step == 7
        assert isinstance(state[0].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi.values())
        assert restored[0].phi == pytest.approx(state[0].phi)


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
        opt = adafactor(lr=1e-3, beta1=0.9)
        u_orig, u_rest = _round_trip(opt, matrix_params, matrix_grads, steps=3)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_factored_v_serialised(self, matrix_params, matrix_grads):
        """v_row / v_col tensors round-trip; the optree treespec is
        skipped from the saved dict (re-derived from the template)."""
        opt = adafactor(lr=1e-3)
        state = opt.init(matrix_params)
        _, state = opt.update(matrix_grads, state, params=matrix_params)
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
        opt = adafactor(lr=1e-3, beta1=0.9, noise_bias_correction=True)
        state = opt.init(matrix_params)
        for _ in range(3):
            _, state = opt.update(
                noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
                state,
                params=matrix_params,
            )
        sd = state_dict(state)
        restored = from_state_dict(opt.init(matrix_params), sd)
        assert restored[0].phi_flat == state[0].phi_flat
        assert restored[0].paths == state[0].paths
        u_orig, _ = opt.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=matrix_params,
        )
        u_rest, _ = opt.update(
            noised(matrix_grads, max_norm=1.0, noise_stddev=pg),
            restored,
            params=matrix_params,
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])


class TestRAdam:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(radam(lr=1e-3), params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        opt = radam(lr=1e-3, noise_bias_correction=True)
        u_orig, u_rest = _round_trip(
            opt,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.3),
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_phi_preserved(self, params, grads):
        opt = radam(lr=1e-3, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(7):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        # RAdam default uses L2 WD (decoupled_weight_decay=False), so the
        # chain is (wd, moment, clip, neg_lr) — moment state is at index 1.
        assert restored[1].step == 7
        assert isinstance(state[1].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[1].phi.values())
        assert restored[1].phi == pytest.approx(state[1].phi)


class TestRMSprop:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(rmsprop(lr=1e-2), params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        opt = rmsprop(lr=1e-2, noise_bias_correction=True)
        u_orig, u_rest = _round_trip(
            opt,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.3),
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_phi_preserved(self, params, grads):
        opt = rmsprop(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(5):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        assert restored[0].step == 5
        assert isinstance(state[0].phi, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi.values())
        assert restored[0].phi == pytest.approx(state[0].phi)


class TestAdagrad:
    def test_round_trip_vanilla(self, params, grads):
        u_orig, u_rest = _round_trip(adagrad(lr=1e-2), params, grads)
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_round_trip_bc(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        u_orig, u_rest = _round_trip(
            opt,
            params,
            noised(grads, max_norm=1.0, noise_stddev=0.3),
        )
        for k in u_orig:
            torch.testing.assert_close(u_orig[k], u_rest[k])

    def test_phi_acc_preserved(self, params, grads):
        opt = adagrad(lr=1e-2, noise_bias_correction=True)
        state = opt.init(params)
        for _ in range(5):
            _, state = opt.update(
                noised(grads, max_norm=1.0, noise_stddev=0.3),
                state,
                params=params,
            )
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        assert restored[0].step == 5
        assert isinstance(state[0].phi_acc, dict)
        assert all(v != pytest.approx(0.0) for v in state[0].phi_acc.values())
        assert restored[0].phi_acc == pytest.approx(state[0].phi_acc)


class TestScheduleFree:
    def test_round_trip_over_adamw(self, params, grads):
        opt = schedule_free(adamw(lr=1e-3))
        state = opt.init(params)
        for _ in range(4):
            delta, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, delta)
        sd = state_dict(state)
        template = opt.init(params)
        restored = from_state_dict(template, sd)
        # x and z should match exactly.
        for k in state.x:
            torch.testing.assert_close(restored.x[k], state.x[k])
            torch.testing.assert_close(restored.z[k], state.z[k])
        assert restored.step == state.step
        assert restored.beta == state.beta
        for k in state.x:
            torch.testing.assert_close(
                get_eval_params(restored)[k], get_eval_params(state)[k]
            )
            torch.testing.assert_close(
                get_train_params(restored)[k], get_train_params(state)[k]
            )


class TestRobustness:
    def test_missing_path_keeps_template(self, params, grads):
        """Forward-compat: a saved dict missing a path keeps the
        template's value at that path."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        # Drop the step entry from the saved dict.
        sd_partial = {k: v for k, v in sd.items() if not k.endswith(".step")}
        template = opt.init(params)
        restored = from_state_dict(template, sd_partial)
        # ``step`` falls back to template's 0.
        assert restored[0].step == 0

    def test_tensor_dtype_device_preserved(self, params, grads):
        """Saved tensors load back at the template's dtype/device."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        # Mutate the saved tensors to bf16 to simulate a saved
        # checkpoint at a different precision than the template.
        sd_bf16 = {
            k: (v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v)
            for k, v in sd.items()
        }
        template = opt.init(params)
        restored = from_state_dict(template, sd_bf16)
        # Restored tensors should match the template's dtype.
        assert restored[0].mu["weight"].dtype == template[0].mu["weight"].dtype

    def test_wrong_type_raises(self, params, grads):
        """A path that should hold a tensor but holds a non-tensor in
        the dict raises ``TypeError`` rather than silently corrupting."""
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params)
        sd = state_dict(state)
        # Find a tensor key and replace its value with a string.
        tensor_key = next(k for k, v in sd.items() if isinstance(v, torch.Tensor))
        sd[tensor_key] = "not a tensor"
        template = opt.init(params)
        with pytest.raises(TypeError, match=r"torch.Tensor"):
            from_state_dict(template, sd)
