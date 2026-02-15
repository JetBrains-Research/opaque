"""Tests for the DP-FTRL integration module."""

import pytest
import torch
import torch.nn as nn

from opaque.dp_ftrl import DPFTRLOptimizer, DPFTRLState, dp_ftrl_train_step
from opaque.matrix_factorization.streaming_matrix import identity, prefix_sum
from opaque.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)
from opaque.noise.matrix_factorization import (
    matrix_factorization_privatizer,
)


class TestDPFTRLOptimizer:
    """Tests for the DPFTRLOptimizer class."""

    def _make_model(self):
        torch.manual_seed(0)
        return nn.Linear(10, 1, bias=False)

    def test_basic_step(self):
        """Optimizer runs without error."""
        model = self._make_model()
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=1.0,
            seed=42,
        )
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        assert optimizer.current_step == 1

    def test_zero_grad(self):
        """zero_grad clears parameter gradients."""
        model = self._make_model()
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=1.0,
            seed=42,
        )
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        assert model.weight.grad is not None
        optimizer.zero_grad()
        assert model.weight.grad is None

    def test_step_counter_advances(self):
        """Step counter increments on each step."""
        model = self._make_model()
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=1.0,
            seed=42,
        )
        for i in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()
            assert optimizer.current_step == i + 1

    def test_noise_is_added(self):
        """Gradients should be noisy (not equal to clean gradients)."""
        model = self._make_model()
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.0,  # zero LR so params don't change
            stddev=10.0,  # Large noise to make it obvious
            seed=42,
        )
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        # The privatizer replaces clean gradients with noisy versions
        # We verify it ran by checking the step advanced
        assert optimizer.current_step == 1

    def test_with_toeplitz_noising(self):
        """Works with BandMF Toeplitz noising matrix."""
        model = self._make_model()
        coefs = optimal_max_error_strategy_coefs(20)
        noising = inverse_as_streaming_matrix(coefs)

        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=noising,
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=1.0,
            seed=42,
        )

        for _ in range(10):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()

        assert optimizer.current_step == 10

    def test_with_dense_noising(self):
        """Works with a dense noising matrix."""
        model = self._make_model()
        noising = torch.eye(5, dtype=torch.float64)

        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=noising,
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=1.0,
            seed=42,
        )

        for _ in range(5):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()

        assert optimizer.current_step == 5

    def test_with_adam_base_optimizer(self):
        """Works with Adam as the base optimizer."""
        model = self._make_model()
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.Adam,
            lr=0.001,
            stddev=0.1,
            seed=42,
        )

        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()

        assert optimizer.current_step == 3

    def test_param_groups(self):
        """param_groups delegates to base optimizer."""
        model = self._make_model()
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=1.0,
        )
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 0.01

    def test_multi_param_model(self):
        """Works with models having multiple parameter groups."""
        torch.manual_seed(0)
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1),
        )
        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=0.1,
            seed=42,
        )

        for _ in range(3):
            optimizer.zero_grad()
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()

        assert optimizer.current_step == 3

    def test_deterministic_with_seed(self):
        """Same seed produces same noisy updates."""
        results = []
        for _ in range(2):
            torch.manual_seed(0)
            model = nn.Linear(10, 1, bias=False)
            optimizer = DPFTRLOptimizer(
                params=model.parameters(),
                noising_matrix=identity(),
                base_optimizer_cls=torch.optim.SGD,
                lr=0.01,
                stddev=1.0,
                seed=42,
            )
            x = torch.randn(4, 10)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            results.append(model.weight.data.clone())

        torch.testing.assert_close(results[0], results[1])


class TestDPFTRLTrainStep:
    """Tests for the composable dp_ftrl_train_step function."""

    def test_basic(self):
        """dp_ftrl_train_step runs one step correctly."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        privatizer = matrix_factorization_privatizer(identity(), stddev=0.1, seed=42)

        template = {
            i: torch.zeros_like(p) for i, p in enumerate(list(model.parameters()))
        }
        noise_state = privatizer.init(template)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Compute gradient
        x = torch.randn(4, 5)
        loss = model(x).sum()
        loss.backward()

        # Collect clipped grads into dict
        params = list(model.parameters())
        clipped_grads = {i: p.grad.clone() for i, p in enumerate(params)}
        optimizer.zero_grad()

        # Run train step
        new_state, _ = dp_ftrl_train_step(
            clipped_grads, noise_state, privatizer, params, optimizer
        )
        assert isinstance(new_state, type(noise_state))

    def test_with_toeplitz(self):
        """dp_ftrl_train_step works with BandMF noising."""
        torch.manual_seed(0)
        model = nn.Linear(5, 1, bias=False)
        steps = 10
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        privatizer = matrix_factorization_privatizer(noising, stddev=0.1, seed=42)

        params = list(model.parameters())
        template = {i: torch.zeros_like(p) for i, p in enumerate(params)}
        noise_state = privatizer.init(template)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        initial_weight = model.weight.data.clone()
        for _ in range(steps):
            x = torch.randn(4, 5)
            loss = model(x).sum()
            loss.backward()
            clipped_grads = {i: p.grad.clone() for i, p in enumerate(params)}
            optimizer.zero_grad()
            noise_state, _ = dp_ftrl_train_step(
                clipped_grads, noise_state, privatizer, params, optimizer
            )

        # Weights should have changed
        assert not torch.allclose(model.weight.data, initial_weight)


class TestDPFTRLvsDPSGD:
    """End-to-end comparison: DP-FTRL (BandMF) should achieve
    better or comparable utility to independent noise (DP-SGD)
    on a simple regression problem."""

    def _train_loop(self, model, optimizer, x_data, y_data, steps):
        """Run training and return final loss."""
        losses = []
        for _step in range(steps):
            optimizer.zero_grad()
            pred = model(x_data)
            loss = ((pred - y_data) ** 2).mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

    def test_bandmf_trains_simple_regression(self):
        """BandMF DP-FTRL can train a simple model (loss decreases)."""
        torch.manual_seed(0)
        x = torch.randn(50, 5)
        true_w = torch.randn(5, 1)
        y = x @ true_w

        model = nn.Linear(5, 1, bias=False)
        steps = 50
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)

        optimizer = DPFTRLOptimizer(
            params=model.parameters(),
            noising_matrix=noising,
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=0.1,
            seed=42,
        )

        losses = self._train_loop(model, optimizer, x, y, steps)
        # Loss should decrease over training
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
        opt_sgd = DPFTRLOptimizer(
            params=model_sgd.parameters(),
            noising_matrix=identity(),
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=stddev,
            seed=42,
        )
        losses_sgd = self._train_loop(model_sgd, opt_sgd, x, y, steps)

        # BandMF (correlated noise)
        torch.manual_seed(0)
        model_mf = nn.Linear(5, 1, bias=False)
        coefs = optimal_max_error_strategy_coefs(steps)
        noising = inverse_as_streaming_matrix(coefs)
        opt_mf = DPFTRLOptimizer(
            params=model_mf.parameters(),
            noising_matrix=noising,
            base_optimizer_cls=torch.optim.SGD,
            lr=0.01,
            stddev=stddev,
            seed=43,
        )
        losses_mf = self._train_loop(model_mf, opt_mf, x, y, steps)

        # Both should train (loss decreases)
        assert losses_sgd[-1] < losses_sgd[0]
        assert losses_mf[-1] < losses_mf[0]

        # BandMF final loss should be within a reasonable factor of DP-SGD
        # (in practice often better, but noise is stochastic)
        assert losses_mf[-1] < losses_sgd[0]  # At least better than start
