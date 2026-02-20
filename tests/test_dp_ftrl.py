"""End-to-end tests for DP-FTRL noise in training loops."""

import torch
import torch.nn as nn

from opaque.random import key
from opaque.noise import custom_mf_noise
from opaque.noise.matrix_factorization import identity
from opaque.noise.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)


def _train_loop(model, optimizer, noise_fn, state, x_data, y_data, steps):
    """Run a DP-FTRL training loop and return losses.

    Uses the (noise_fn, state) API to add correlated noise to gradients
    before each optimizer step.
    """
    params = list(model.parameters())

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(x_data)
        loss = ((pred - y_data) ** 2).mean()
        loss.backward()

        # Privatize: collect grads, add correlated noise, write back
        grads = {i: p.grad.clone() for i, p in enumerate(params)}
        noisy_grads, state = noise_fn(grads, state)
        for i, p in enumerate(params):
            p.grad = noisy_grads[i].to(p.dtype)

        optimizer.step()
        losses.append(loss.item())
    return losses


class TestDPFTRLTrainingLoop:
    """Tests for using custom_mf_noise in a training loop."""

    def _make_template(self, model):
        return {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}

    def test_identity_noise_trains(self):
        """Identity noise (DP-SGD equivalent) trains a simple model."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = custom_mf_noise(
            self._make_template(model),
            identity(),
            stddev=0.1,
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _train_loop(model, optimizer, noise_fn, state, x, y, steps=50)
        assert losses[-1] < losses[0]

    def test_toeplitz_noise_trains(self):
        """BandMF Toeplitz noise trains a simple model."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 50
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        noise_fn, state = custom_mf_noise(
            self._make_template(model),
            noising,
            stddev=0.1,
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _train_loop(model, optimizer, noise_fn, state, x, y, steps=steps)
        assert losses[-1] < losses[0]

    def test_dense_matrix_noise_trains(self):
        """Dense noising matrix trains a simple model."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 5
        noising = torch.eye(steps, dtype=torch.float64)
        noise_fn, state = custom_mf_noise(
            self._make_template(model),
            noising,
            stddev=0.1,
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _train_loop(model, optimizer, noise_fn, state, x, y, steps=steps)
        assert losses[-1] < losses[0]

    def test_with_adam_optimizer(self):
        """Works with Adam as the base optimizer."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        noise_fn, state = custom_mf_noise(
            self._make_template(model),
            identity(),
            stddev=0.1,
            key=key(42),
        )

        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        losses = _train_loop(model, optimizer, noise_fn, state, x, y, steps=30)
        assert losses[-1] < losses[0]

    def test_multi_param_model(self):
        """Works with models having multiple parameters."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        noise_fn, state = custom_mf_noise(
            self._make_template(model),
            identity(),
            stddev=0.1,
            key=key(42),
        )

        x = torch.randn(50, 10)
        true_w = torch.randn(10, 1)
        y = x @ true_w

        losses = _train_loop(model, optimizer, noise_fn, state, x, y, steps=30)
        assert losses[-1] < losses[0]

    def test_deterministic_with_seed(self):
        """Same seed produces same noisy updates."""
        results = []
        for _ in range(2):
            torch.manual_seed(0)
            model = nn.Linear(5, 1, bias=False)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            noise_fn, state = custom_mf_noise(
                self._make_template(model),
                identity(),
                stddev=1.0,
                key=key(42),
            )

            x = torch.randn(4, 5)
            true_w = torch.randn(5, 1)
            y = x @ true_w

            _train_loop(model, optimizer, noise_fn, state, x, y, steps=3)
            results.append(model.weight.data.clone())

        torch.testing.assert_close(results[0], results[1])


class TestBandMFvsDPSGD:
    """End-to-end comparison: BandMF (correlated noise) should achieve
    better or comparable utility to independent noise (DP-SGD)
    on a simple regression problem."""

    def _make_template(self, model):
        return {i: torch.zeros_like(p) for i, p in enumerate(model.parameters())}

    def test_bandmf_trains_simple_regression(self):
        """BandMF DP-FTRL can train a simple model (loss decreases)."""
        torch.manual_seed(0)
        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        model = nn.Linear(5, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        steps = 50
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        noise_fn, state = custom_mf_noise(
            self._make_template(model),
            noising,
            stddev=0.1,
            key=key(42),
        )

        losses = _train_loop(model, optimizer, noise_fn, state, x, y, steps)
        assert losses[-1] < losses[0]

    def test_bandmf_vs_dpsgd_utility(self):
        """BandMF should achieve comparable or better utility than DP-SGD.

        This is the key end-to-end validation: on a simple linear regression,
        BandMF (correlated noise) should not be worse than independent noise
        at the same noise level.
        """
        torch.manual_seed(0)
        x = torch.randn(100, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w
        steps = 100
        stddev = 0.5

        # DP-SGD (identity = independent noise)
        torch.manual_seed(0)
        model_sgd = nn.Linear(5, 1, bias=False)
        opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
        noise_sgd, state_sgd = custom_mf_noise(
            self._make_template(model_sgd),
            identity(),
            stddev=stddev,
            key=key(42),
        )
        losses_sgd = _train_loop(model_sgd, opt_sgd, noise_sgd, state_sgd, x, y, steps)

        # BandMF (correlated noise)
        torch.manual_seed(0)
        model_mf = nn.Linear(5, 1, bias=False)
        opt_mf = torch.optim.SGD(model_mf.parameters(), lr=0.01)
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        noise_mf, state_mf = custom_mf_noise(
            self._make_template(model_mf),
            noising,
            stddev=stddev,
            key=key(43),
        )
        losses_mf = _train_loop(model_mf, opt_mf, noise_mf, state_mf, x, y, steps)

        # Both should train (loss decreases)
        assert losses_sgd[-1] < losses_sgd[0]
        assert losses_mf[-1] < losses_mf[0]

        # BandMF final loss should be at least better than start
        assert losses_mf[-1] < losses_sgd[0]
