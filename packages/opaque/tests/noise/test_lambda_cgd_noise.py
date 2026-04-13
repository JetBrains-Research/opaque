"""Tests for DP-λCGD noise generation via PRNG replay."""

import torch
import torch.nn as nn

from opaque.noise.lambda_cgd_noise import lambda_cgd_noise
from opaque.random import key


class TestLambdaCgdNoise:
    def _make_template(self):
        return {"w": torch.zeros(10)}

    def test_basic_noise_generation(self):
        """Noise function returns correctly shaped output."""
        template = self._make_template()
        noise_fn, state = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9
        )
        noisy, new_state = noise_fn({"w": torch.zeros(10)}, state)
        assert noisy["w"].shape == (10,)
        assert new_state._step_counter == 1

    def test_deterministic_with_same_key(self):
        """Same key produces identical noise sequences."""
        template = self._make_template()
        results = []
        for _ in range(2):
            noise_fn, state = lambda_cgd_noise(
                template, 100, stddev=1.0, key=key(42), lambda_=0.9
            )
            noisy, state = noise_fn({"w": torch.zeros(10)}, state)
            noisy2, state = noise_fn({"w": torch.zeros(10)}, state)
            results.append(torch.cat([noisy["w"], noisy2["w"]]))
        torch.testing.assert_close(results[0], results[1])

    def test_different_keys_give_different_noise(self):
        """Different keys produce different sequences."""
        template = self._make_template()
        noise_fn1, state1 = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(1), lambda_=0.9
        )
        noise_fn2, state2 = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(2), lambda_=0.9
        )
        noisy1, _ = noise_fn1({"w": torch.zeros(10)}, state1)
        noisy2, _ = noise_fn2({"w": torch.zeros(10)}, state2)
        assert not torch.allclose(noisy1["w"], noisy2["w"])

    def test_lambda_zero_is_independent(self):
        """λ=0 should produce independent noise at each step (DP-SGD)."""
        template = self._make_template()
        noise_fn, state = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.0
        )

        # First step: z_0
        noisy0, state = noise_fn({"w": torch.zeros(10)}, state)
        # Second step: z_1 (no subtraction of previous)
        noisy1, state = noise_fn({"w": torch.zeros(10)}, state)

        # Both should have similar variance (stddev=1.0)
        assert noisy0["w"].std().item() > 0.1
        assert noisy1["w"].std().item() > 0.1

    def test_first_step_same_as_lambda_zero(self):
        """First step is z_0 regardless of λ (no previous noise to subtract) — unnormalized."""
        template = self._make_template()

        noise_fn_corr, state_corr = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9, normalized=False
        )
        noise_fn_ind, state_ind = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.0, normalized=False
        )

        noisy_corr, _ = noise_fn_corr({"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = noise_fn_ind({"w": torch.zeros(10)}, state_ind)

        torch.testing.assert_close(noisy_corr["w"], noisy_ind["w"])

    def test_correlation_changes_second_step(self):
        """Second step with λ>0 should differ from λ=0."""
        template = self._make_template()

        noise_fn_corr, state_corr = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9
        )
        noise_fn_ind, state_ind = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.0
        )

        _, state_corr = noise_fn_corr({"w": torch.zeros(10)}, state_corr)
        _, state_ind = noise_fn_ind({"w": torch.zeros(10)}, state_ind)

        noisy_corr, _ = noise_fn_corr({"w": torch.zeros(10)}, state_corr)
        noisy_ind, _ = noise_fn_ind({"w": torch.zeros(10)}, state_ind)

        assert not torch.allclose(noisy_corr["w"], noisy_ind["w"])

    def test_multi_param(self):
        """Works with multiple parameter tensors."""
        template = {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)}
        noise_fn, state = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9
        )
        noisy, new_state = noise_fn(
            {"w1": torch.zeros(5), "w2": torch.zeros(3, 4)}, state
        )
        assert noisy["w1"].shape == (5,)
        assert noisy["w2"].shape == (3, 4)

    def test_noise_adds_to_grads(self):
        """Noise is added to the gradient, not overwriting it."""
        template = self._make_template()
        noise_fn, state = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9
        )
        grad = {"w": torch.ones(10) * 5.0}
        noisy, _ = noise_fn(grad, state)
        # The result should be 5.0 + noise, so different from just noise
        noise_fn2, state2 = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9
        )
        noise_only, _ = noise_fn2({"w": torch.zeros(10)}, state2)
        torch.testing.assert_close(noisy["w"] - 5.0, noise_only["w"])

    def test_step_counter_increments(self):
        """Step counter increments with each call."""
        template = self._make_template()
        noise_fn, state = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.9
        )
        assert state._step_counter == 0
        _, state = noise_fn({"w": torch.zeros(10)}, state)
        assert state._step_counter == 1
        _, state = noise_fn({"w": torch.zeros(10)}, state)
        assert state._step_counter == 2

    def test_rejects_invalid_lambda(self):
        with torch.no_grad():
            template = self._make_template()
            import pytest

            with pytest.raises(ValueError):
                lambda_cgd_noise(template, 100, stddev=1.0, key=key(42), lambda_=-0.1)
            with pytest.raises(ValueError):
                lambda_cgd_noise(template, 100, stddev=1.0, key=key(42), lambda_=1.0)

    def test_prng_replay_correctness(self):
        """Verify the PRNG replay: step 1's z_prev should equal step 0's z_current."""
        template = {"w": torch.zeros(20)}

        # Run with λ=0, normalized=False to get individual z_0 and z_1
        noise_fn_ind, state_ind = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.0, normalized=False
        )
        z0, state_ind = noise_fn_ind({"w": torch.zeros(20)}, state_ind)
        z1, _ = noise_fn_ind({"w": torch.zeros(20)}, state_ind)

        # Run with λ>0, normalized=False: step 1 should be z_1 - λ*z_0
        noise_fn_corr, state_corr = lambda_cgd_noise(
            template, 100, stddev=1.0, key=key(42), lambda_=0.5, normalized=False
        )
        _, state_corr = noise_fn_corr({"w": torch.zeros(20)}, state_corr)
        step1_corr, _ = noise_fn_corr({"w": torch.zeros(20)}, state_corr)

        expected = z1["w"] - 0.5 * z0["w"]
        torch.testing.assert_close(step1_corr["w"], expected, atol=1e-6, rtol=1e-6)


class TestLambdaCgdTraining:
    """Integration test: λCGD noise in a training loop."""

    def _make_template(self, model):
        return {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}

    def test_trains_simple_regression(self):
        """DP-λCGD noise can train a simple model (loss decreases)."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = lambda_cgd_noise(
            self._make_template(model),
            50,
            stddev=0.1,
            key=key(42),
            lambda_=0.9,
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        params = list(model.parameters())
        losses = []
        for _ in range(50):
            optimizer.zero_grad()
            pred = model(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()

            grads = {i: p.grad.clone() for i, p in enumerate(params)}
            noisy_grads, state = noise_fn(grads, state)
            for i, p in enumerate(params):
                p.grad = noisy_grads[i].to(p.dtype)

            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0]


class TestLambdaCgdColumnNormalization:
    """Tests for column-normalized DP-λCGD noise (Appendix A)."""

    def _make_template(self):
        return {"w": torch.zeros(20)}

    def test_normalized_scales_first_step(self):
        """Step 0 with normalized=True: d_0 * z_0 ≠ z_0 for λ>0."""
        template = self._make_template()
        noise_fn_norm, state_norm = lambda_cgd_noise(
            template,
            100,
            stddev=1.0,
            key=key(42),
            lambda_=0.9,
            normalized=True,
        )
        noise_fn_raw, state_raw = lambda_cgd_noise(
            template,
            100,
            stddev=1.0,
            key=key(42),
            lambda_=0.9,
            normalized=False,
        )
        out_norm, _ = noise_fn_norm({"w": torch.zeros(20)}, state_norm)
        out_raw, _ = noise_fn_raw({"w": torch.zeros(20)}, state_raw)
        # d_0 > 1 for λ=0.9, so normalized output should be larger
        assert out_norm["w"].norm().item() > out_raw["w"].norm().item()

    def test_normalized_matches_column_norm_formula(self):
        """Verify d_t scaling: normalized = d_t * unnormalized at each step."""
        from opaque.noise.lambda_cgd_noise import _column_norm

        template = self._make_template()
        n_steps = 50
        lam = 0.7

        noise_fn_norm, state_norm = lambda_cgd_noise(
            template,
            n_steps,
            stddev=1.0,
            key=key(42),
            lambda_=lam,
            normalized=True,
        )
        noise_fn_raw, state_raw = lambda_cgd_noise(
            template,
            n_steps,
            stddev=1.0,
            key=key(42),
            lambda_=lam,
            normalized=False,
        )

        for step in range(5):
            out_norm, state_norm = noise_fn_norm({"w": torch.zeros(20)}, state_norm)
            out_raw, state_raw = noise_fn_raw({"w": torch.zeros(20)}, state_raw)
            d_t = _column_norm(lam, n_steps, step)
            torch.testing.assert_close(
                out_norm["w"],
                out_raw["w"] * d_t,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_column_norm_decreasing(self):
        """Column norms decrease from first to last step."""
        from opaque.noise.lambda_cgd_noise import _column_norm

        n_steps = 100
        lam = 0.9
        d_prev = _column_norm(lam, n_steps, 0)
        for t in range(1, n_steps):
            d_t = _column_norm(lam, n_steps, t)
            assert d_t <= d_prev + 1e-12, f"step {t}: d_t={d_t} > d_prev={d_prev}"
            d_prev = d_t
        # Last column norm should be 1.0
        assert abs(_column_norm(lam, n_steps, n_steps - 1) - 1.0) < 1e-10

    def test_column_norm_lambda_zero(self):
        """λ=0: all column norms are 1.0 (identity matrix)."""
        from opaque.noise.lambda_cgd_noise import _column_norm

        for t in range(10):
            assert _column_norm(0.0, 100, t) == 1.0

    def test_normalized_deterministic(self):
        """Normalized noise is reproducible with same key."""
        template = self._make_template()
        results = []
        for _ in range(2):
            noise_fn, state = lambda_cgd_noise(
                template,
                100,
                stddev=1.0,
                key=key(42),
                lambda_=0.9,
                normalized=True,
            )
            out, _ = noise_fn({"w": torch.zeros(20)}, state)
            results.append(out["w"].clone())
        torch.testing.assert_close(results[0], results[1])

    def test_rejects_invalid_n_steps(self):
        """n_steps < 1 should raise ValueError."""
        import pytest

        template = self._make_template()
        with pytest.raises(ValueError):
            lambda_cgd_noise(template, 0, stddev=1.0, key=key(42), lambda_=0.9)
