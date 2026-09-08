"""Tests for DP-lambda-CGD noise generation via PRNG replay."""

from collections import defaultdict
from dataclasses import replace

import pytest
import torch

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.dpftrl.noise import _lambda_cgd as lambda_cgd_module
from opaque.api.dpftrl.noise._lambda_cgd import (
    LAMBDA_CGD_STREAM_FOLD,
    LambdaCgdStrategy,
    lambda_cgd_strategy,
)
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.exceptions import CheckpointError
from opaque.random import fold_in, generator_from_key, key
from opaque.random.types import RngKey
from opaque.serialization import from_state_dict, state_dict
from opaque.types import NoisedPytree, PerGroup, clipped


def _make_noise(
    template,
    n_steps=100,
    lambda_=0.9,
    normalized=True,
    seed=42,
    noise_multiplier=1.0,
    compute_dtype=torch.float32,
    rng_key=None,
):
    """Helper: create lambda-CGD noise via the strategy + mf_gaussian_noise API.

    Uses ``noise_multiplier=1.0`` so realized stddev equals each call's
    ``ClippedPytree.max_norm``; tests pass ``max_norm=1.0`` to recover the
    historical ``stddev=1.0`` semantics.
    """
    strategy = lambda_cgd_strategy(lambda_=lambda_, normalized=normalized)
    return mf_gaussian_noise(
        template,
        strategy,
        n_steps=n_steps,
        min_sep=1,
        max_participations=1,
        noise_multiplier=noise_multiplier,
        key=key(seed) if rng_key is None else rng_key,
        compute_dtype=compute_dtype,
    )


def _call(noise_fn, grad_pytree, state, *, max_norm=1.0):
    """Wrap ``grad_pytree`` as clipped, run noise, return (noisy_pytree, state)."""
    noisy_out, new_state = noise_fn(clipped(grad_pytree, max_norm=max_norm), state)
    assert isinstance(noisy_out, NoisedPytree)
    return noisy_out.pytree, new_state


class TestLambdaCgdNoise:
    def _make_template(self):
        return {"w": torch.zeros(10)}

    def test_basic_noise_generation(self):
        """Noise function returns correctly shaped output."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        noised, new_state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert noised["w"].shape == (10,)
        assert new_state._step_counter == 1

    def test_deterministic_with_same_key(self):
        """Same key produces identical noise sequences."""
        template = self._make_template()
        results = []
        for _ in range(2):
            noise_fn, state = _make_noise(template)
            noised, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            noisy2, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
            results.append(torch.cat([noised["w"], noisy2["w"]]))
        torch.testing.assert_close(results[0], results[1])

    def test_different_keys_give_different_noise(self):
        """Different keys produce different sequences."""
        template = self._make_template()
        noise_fn1, state1 = _make_noise(template, seed=1)
        noise_fn2, state2 = _make_noise(template, seed=2)
        noisy1, _ = _call(noise_fn1, {"w": torch.zeros(10)}, state1)
        noisy2, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        assert not torch.allclose(noisy1["w"], noisy2["w"])

    def test_lambda_zero_is_independent(self):
        """lambda=0 should produce independent noise at each step (DP-SGD)."""
        template = self._make_template()
        noise_fn, state = _make_noise(template, lambda_=0.0)

        noisy0, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        noisy1, state = _call(noise_fn, {"w": torch.zeros(10)}, state)

        assert noisy0["w"].std().item() > 0.1
        assert noisy1["w"].std().item() > 0.1

    def test_first_step_same_as_lambda_zero(self):
        """First step is z_0 regardless of lambda (no previous noise) -- unnormalized."""
        template = self._make_template()

        noise_fn_corr, state_corr = _make_noise(template, lambda_=0.9, normalized=False)
        noise_fn_ind, state_ind = _make_noise(template, lambda_=0.0, normalized=False)

        noisy_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        torch.testing.assert_close(noisy_corr["w"], noisy_ind["w"])

    def test_correlation_changes_second_step(self):
        """Second step with lambda>0 should differ from lambda=0."""
        template = self._make_template()

        noise_fn_corr, state_corr = _make_noise(template, lambda_=0.9)
        noise_fn_ind, state_ind = _make_noise(template, lambda_=0.0)

        _, state_corr = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        _, state_ind = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        noisy_corr, _ = _call(noise_fn_corr, {"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = _call(noise_fn_ind, {"w": torch.zeros(10)}, state_ind)

        assert not torch.allclose(noisy_corr["w"], noisy_ind["w"])

    def test_multi_param(self):
        """Works with multiple parameter tensors."""
        template = {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)}
        noise_fn, state = _make_noise(template)
        noised, _new_state = _call(
            noise_fn,
            {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)},
            state,
        )
        assert noised["w1"].shape == (5,)
        assert noised["w2"].shape == (3, 4)

    def test_noise_adds_to_grads(self):
        """Noise is added to the gradient, not overwriting it."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        grad = {"w": torch.ones(10) * 5.0}
        noised, _ = _call(noise_fn, grad, state)
        noise_fn2, state2 = _make_noise(template)
        noise_only, _ = _call(noise_fn2, {"w": torch.zeros(10)}, state2)
        torch.testing.assert_close(noised["w"] - 5.0, noise_only["w"])

    def test_step_counter_increments(self):
        """Step counter increments with each call."""
        template = self._make_template()
        noise_fn, state = _make_noise(template)
        assert state._step_counter == 0
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 1
        _, state = _call(noise_fn, {"w": torch.zeros(10)}, state)
        assert state._step_counter == 2

    def test_rejects_invalid_lambda(self):
        for value in (-0.1, 1.0, float("nan"), float("inf")):
            with pytest.raises(
                ValueError, match=r"lambda_ must be finite and in \[0, 1\)"
            ):
                lambda_cgd_strategy(lambda_=value)

    def test_namespaced_replay_matches_manual_draws(self):
        template = {"w": torch.zeros(20)}
        base = key(42)
        noise_fn, state = _make_noise(template, lambda_=0.5, normalized=False)

        step0, state = _call(noise_fn, template, state)
        step1, _ = _call(noise_fn, template, state)

        def draw(step):
            generator = generator_from_key(fold_in(base, LAMBDA_CGD_STREAM_FOLD, step))
            return torch.randn((20,), generator=generator)

        z0 = draw(0)
        z1 = draw(1)
        torch.testing.assert_close(step0["w"], z0, atol=0, rtol=0)
        torch.testing.assert_close(step1["w"], z1 - 0.5 * z0, atol=0, rtol=0)

        legacy_generator = generator_from_key(fold_in(base, 0))
        legacy_z0 = torch.randn((20,), generator=legacy_generator)
        assert not torch.equal(step0["w"], legacy_z0)

    def test_lambda_zero_still_uses_namespace(self):
        template = {"w": torch.zeros(20)}
        noise_fn, state = _make_noise(template, lambda_=0.0, normalized=False)
        actual, _ = _call(noise_fn, template, state)

        rooted_generator = generator_from_key(
            fold_in(key(42), LAMBDA_CGD_STREAM_FOLD, 0)
        )
        expected = torch.randn((20,), generator=rooted_generator)
        torch.testing.assert_close(actual["w"], expected, atol=0, rtol=0)

    def test_state_round_trip_preserves_replay_contract(self):
        template = {"w": torch.zeros(20)}
        noise_fn, state = _make_noise(template, lambda_=0.5, normalized=False)
        _, state = _call(noise_fn, template, state)
        saved = state_dict(state)

        resumed_fn, fresh = _make_noise(template, lambda_=0.5, normalized=False)
        restored = from_state_dict(fresh, saved)
        expected, _ = _call(noise_fn, template, state)
        actual, _ = _call(resumed_fn, template, restored)

        assert saved["_inner_state.execution_identity.replay_version"] == 1
        assert (
            saved["_inner_state.execution_identity.stream_root"]
            == LAMBDA_CGD_STREAM_FOLD
        )
        assert saved["_inner_state.step"] == 1
        assert len(saved["_inner_state.base_stddev_digest"]) == 64
        torch.testing.assert_close(actual["w"], expected["w"], atol=0, rtol=0)

    def test_equivalent_defaultdict_layout_resumes_exactly(self):
        source_template = defaultdict(lambda: None, w=torch.zeros(20))
        source_fn, source = _make_noise(source_template)
        _, source = _call(source_fn, source_template, source)

        target_template = defaultdict(lambda: None, w=torch.zeros(20))
        target_fn, target = _make_noise(target_template)
        restored = from_state_dict(target, state_dict(source))
        expected, _ = _call(source_fn, source_template, source)
        actual, _ = _call(target_fn, target_template, restored)

        torch.testing.assert_close(actual["w"], expected["w"], atol=0, rtol=0)

    def test_restore_uses_saved_rng_implementation(self):
        template = {"w": torch.zeros(20)}
        noise_fn, state = _make_noise(template, lambda_=0.5, normalized=False)
        _, state = _call(noise_fn, template, state)
        resumed_fn, fresh = _make_noise(
            template,
            lambda_=0.5,
            normalized=False,
            rng_key=RngKey(seed=999, impl="poison-template"),
        )

        restored = from_state_dict(fresh, state_dict(state))
        expected, _ = _call(noise_fn, template, state)
        actual, resumed = _call(resumed_fn, template, restored)

        assert restored._rng_key.impl == "opaque_threefry_like"
        assert (
            resumed._inner_state.execution_identity.rng_seed == restored._rng_key.seed
        )
        assert (
            resumed._inner_state.execution_identity.rng_impl == restored._rng_key.impl
        )
        torch.testing.assert_close(actual["w"], expected["w"], atol=0, rtol=0)

    def test_rejects_inconsistent_saved_rng_seed_before_draw(self, monkeypatch):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        _, state = _call(noise_fn, template, state)
        saved = state_dict(state)
        saved["_rng_key.seed"] += 1
        _, fresh = _make_noise(template)
        restored = from_state_dict(fresh, saved)
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="incompatible RNG seed"):
            _call(noise_fn, template, restored)

    def test_rejects_inconsistent_saved_rng_implementation_before_draw(
        self, monkeypatch
    ):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        _, state = _call(noise_fn, template, state)
        saved = state_dict(state)
        saved["_rng_key.impl"] = "other"
        _, fresh = _make_noise(template)
        restored = from_state_dict(fresh, saved)
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="incompatible RNG implementation"):
            _call(noise_fn, template, restored)

    @pytest.mark.parametrize(
        ("lambda_", "completed_steps"),
        [
            (0.0, 0),
            (0.5, 0),
            (0.5, 4),
        ],
        ids=("lambda-zero", "fresh", "completed-horizon"),
    )
    def test_rejects_legacy_unrooted_state(self, lambda_, completed_steps):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template, lambda_=lambda_, n_steps=4)
        for _ in range(completed_steps):
            _, state = _call(noise_fn, template, state)
        saved = state_dict(state)
        for field in tuple(saved):
            if field.startswith("_inner_state."):
                del saved[field]
        saved["_inner_state"] = None
        saved["_inner_state_fields"] = '["_inner_state"]'
        _, fresh = _make_noise(template)

        with pytest.raises(CheckpointError, match="incompatible RNG topology"):
            from_state_dict(fresh, saved)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("replay_version", 2),
            ("replay_version", 1.0),
            ("replay_version", True),
            ("stream_root", "opaque.dpftrl.other"),
        ],
    )
    def test_rejects_different_replay_topology(self, field, value):
        template = {"w": torch.zeros(2)}
        _, state = _make_noise(template)
        saved = state_dict(state)
        saved[f"_inner_state.execution_identity.{field}"] = value
        _, fresh = _make_noise(template)

        with pytest.raises(CheckpointError, match="execution identity"):
            from_state_dict(fresh, saved)

    @pytest.mark.parametrize("mutation", ["missing", "extra"])
    def test_rejects_malformed_replay_contract(self, mutation):
        template = {"w": torch.zeros(2)}
        _, state = _make_noise(template)
        saved = state_dict(state)
        fields = sorted(key for key in saved if key.startswith("_inner_state."))
        if mutation == "missing":
            field = "_inner_state.execution_identity.stream_root"
            del saved[field]
            fields.remove(field)
        else:
            saved["_inner_state.unexpected"] = "value"
            fields.append("_inner_state.unexpected")
        saved["_inner_state_fields"] = '["' + '","'.join(sorted(fields)) + '"]'
        _, fresh = _make_noise(template)

        with pytest.raises(CheckpointError):
            from_state_dict(fresh, saved)

    def test_rejects_legacy_runtime_state_before_draw(self, monkeypatch):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        legacy_state = replace(state, _inner_state=None)
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="incompatible RNG topology"):
            _call(noise_fn, template, legacy_state)

    @pytest.mark.parametrize(
        "target_kwargs",
        [
            {"lambda_": 0.6},
            {"normalized": True},
            {"n_steps": 101},
            {"compute_dtype": torch.float64},
        ],
    )
    def test_rejects_checkpoint_from_different_execution(self, target_kwargs):
        template = {"w": torch.zeros(2)}
        source_fn, source = _make_noise(
            template,
            lambda_=0.5,
            normalized=False,
            n_steps=100,
        )
        _, source = _call(source_fn, template, source)
        kwargs = {"lambda_": 0.5, "normalized": False, "n_steps": 100}
        kwargs.update(target_kwargs)
        _, target = _make_noise(template, **kwargs)

        with pytest.raises(CheckpointError, match="execution identity"):
            from_state_dict(target, state_dict(source))

    @pytest.mark.parametrize(
        "target_template",
        [
            {"w": torch.zeros(3)},
            {"w": torch.zeros(2, dtype=torch.float64)},
            [torch.zeros(2)],
        ],
    )
    def test_rejects_checkpoint_for_different_gradient_layout(self, target_template):
        source_template = {"w": torch.zeros(2)}
        source_fn, source = _make_noise(source_template)
        _, source = _call(source_fn, source_template, source)
        _, target = _make_noise(target_template)

        with pytest.raises(CheckpointError, match="execution identity"):
            from_state_dict(target, state_dict(source))

    @pytest.mark.parametrize(
        "current",
        [
            {"w": torch.zeros(3)},
            {"w": torch.zeros(2, dtype=torch.float64)},
            [torch.zeros(2)],
        ],
    )
    def test_rejects_runtime_gradient_layout_before_draw(self, current, monkeypatch):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        _, state = _call(noise_fn, template, state)
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(TypeError, match="construction template"):
            _call(noise_fn, current, state)

    def test_rejects_noise_multiplier_drift_before_draw(self, monkeypatch):
        template = {"w": torch.zeros(2)}
        source_fn, source = _make_noise(template, noise_multiplier=1.0)
        _, source = _call(source_fn, template, source)
        target_fn, target = _make_noise(template, noise_multiplier=2.0)
        restored = from_state_dict(target, state_dict(source))
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="base noise scale changed"):
            _call(target_fn, template, restored)

    def test_rejects_per_group_scale_drift_before_draw(self, monkeypatch):
        template = {"a": torch.zeros(2), "b": torch.zeros(2)}
        bound = PerGroup(
            groups={"a": "left", "b": "right"},
            values={"left": 1.0, "right": 2.0},
        )
        source_fn, source = _make_noise(template, noise_multiplier=1.0)
        _, source = _call(source_fn, template, source, max_norm=bound)
        target_fn, target = _make_noise(template, noise_multiplier=1.5)
        restored = from_state_dict(target, state_dict(source))
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="base noise scale changed"):
            _call(target_fn, template, restored, max_norm=bound)

    @pytest.mark.parametrize("digest", [None, "not-a-digest"])
    def test_rejects_malformed_scale_latch(self, digest):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        _, state = _call(noise_fn, template, state)
        saved = state_dict(state)
        saved["_inner_state.base_stddev_digest"] = digest
        _, fresh = _make_noise(template)

        with pytest.raises(CheckpointError, match="noise-scale latch"):
            from_state_dict(fresh, saved)

    def test_rejects_missing_sensitivity_latch_before_draw(self, monkeypatch):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        _, state = _call(noise_fn, template, state)
        malformed = replace(
            state,
            _first_max_norm=None,
            _first_max_norm_sync_fingerprint=None,
        )
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="inconsistent MF noise state"):
            _call(noise_fn, template, malformed)

    def test_rejects_inner_outer_step_mismatch_before_draw(self, monkeypatch):
        template = {"w": torch.zeros(2)}
        noise_fn, state = _make_noise(template)
        inner = replace(state._inner_state, step=1, base_stddev_digest="0" * 64)
        malformed = replace(state, _inner_state=inner)
        monkeypatch.setattr(
            lambda_cgd_module,
            "generator_from_key",
            lambda _key: pytest.fail("generator must not be constructed"),
        )

        with pytest.raises(CheckpointError, match="inner and outer replay steps"):
            _call(noise_fn, template, malformed)


_PARTICIPATION = {"n_steps": 100, "min_sep": 25, "max_participations": 4}


class TestLambdaCgdStrategy:
    def test_returns_correct_type(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert isinstance(s, LambdaCgdStrategy)

    def test_sensitivity_positive(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.sensitivity(**_PARTICIPATION) > 0

    def test_gram_matrix_present(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        gram = s.gram_matrix(**_PARTICIPATION)
        assert gram is not None
        assert len(gram) == 25 * 25

    def test_lr_schedule_is_rejected_with_recalibration_guidance(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            lambda_cgd_strategy(lambda_=0.4, lr_schedule=lambda _step: 1.0)

    def test_legacy_none_schedule_state_loads(self):
        strategy = from_state_dict(
            lambda_cgd_strategy(lambda_=0.4),
            {
                "type": "LambdaCgdStrategy",
                "lambda_": 0.4,
                "normalized": False,
                "lr_schedule": None,
            },
        )

        assert strategy == lambda_cgd_strategy(lambda_=0.4, normalized=False)
        assert state_dict(strategy)["lr_schedule"] is None

    def test_legacy_non_none_schedule_state_is_rejected(self):
        with pytest.raises(
            ValueError, match=r"does not support lr_schedule.*recalibrate"
        ):
            from_state_dict(
                lambda_cgd_strategy(lambda_=0.4),
                {
                    "type": "LambdaCgdStrategy",
                    "lambda_": 0.4,
                    "normalized": False,
                    "lr_schedule": {
                        "__opaque_recipe__": "ConstantSchedule",
                        "value": 0.1,
                    },
                },
            )

    def test_gram_uses_only_unweighted_native_path(self, monkeypatch):
        expected = (2.0, 0.5, 0.5, 1.0)

        class CapturingNative:
            def lambda_cgd_gram_matrix(self, *_args):
                return expected

            def lambda_cgd_gram_matrix_lr(self, *_args):
                raise AssertionError("weighted Gram path must not be called")

        monkeypatch.setattr(lambda_cgd_module, "_native", CapturingNative)
        lambda_cgd_module._lambda_cgd_gram_matrix_cached.cache_clear()
        strategy = lambda_cgd_strategy(lambda_=0.4)

        assert strategy.gram_matrix(
            n_steps=4, min_sep=2, max_participations=2
        ) == pytest.approx(expected)
        lambda_cgd_module._lambda_cgd_gram_matrix_cached.cache_clear()

    def test_normalized_single_participation_sensitivity_one(self):
        """Normalized + single participation -> sensitivity = 1.0."""
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.sensitivity(
            n_steps=100, min_sep=1, max_participations=1
        ) == pytest.approx(1.0, abs=1e-6)

    def test_momentum_not_accepted(self):
        """lambda_cgd_strategy does not accept momentum (use bisr_strategy for that)."""
        with pytest.raises(TypeError):
            lambda_cgd_strategy(lambda_=0.5, momentum=0.95)

    def test_unnormalized(self):
        s = lambda_cgd_strategy(lambda_=0.9, normalized=False)
        assert s.sensitivity(**_PARTICIPATION) > 0

    def test_internal_fields(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        assert s.lambda_ == pytest.approx(0.9)
        assert s.normalized is True


class TestLambdaCgdPld:
    delta = 1e-5

    def test_lambda_cgd_pld(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        eps = ftrl_acc.mf_gaussian(1.0, s, **_PARTICIPATION).epsilon_at(self.delta)
        assert eps > 0

    def test_lambda_cgd_bnb(self):
        s = lambda_cgd_strategy(lambda_=0.9)
        eps = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, s),
            num_bins=25,
            n_steps=100,
        ).epsilon_at(
            1e-2,
            mc_resolution=5e-3,
            mc_failure_probability=1e-2,
        )
        assert eps > 0
