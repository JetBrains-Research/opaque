"""Tests for auditing.gradient_scores."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import opaque.auditing as auditing
from opaque.auditing import gradient_scores, one_run
from opaque.random import key


@pytest.fixture
def linear_setup():
    """Create a simple linear model, dataset, and loss function."""
    torch.manual_seed(42)
    input_dim = 10
    n_samples = 200

    X = torch.randn(n_samples, input_dim)
    true_w = torch.randn(input_dim)
    y = X @ true_w

    dataset = TensorDataset(X, y)
    params = true_w.clone()  # Perfect model

    def loss_fn(params, x, y):
        return F.mse_loss(x @ params, y, reduction="sum")

    return params, dataset, loss_fn


class TestGradientScores:
    """Tests for auditing.gradient_scores."""

    def test_basic_scoring(self, linear_setup):
        """Test that gradient_scores returns correct shape."""
        params, dataset, loss_fn = linear_setup
        loader = DataLoader(dataset, batch_size=64)
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        assert scores.shape == (200,)

    def test_scoring_with_subset(self, linear_setup):
        """Test scoring a subset via Subset + DataLoader."""
        params, dataset, loss_fn = linear_setup
        indices = [0, 10, 20, 30, 40]
        loader = DataLoader(Subset(dataset, indices), batch_size=32)
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        assert scores.shape == (5,)

    def test_scores_are_non_positive(self, linear_setup):
        """Test that scores are <= 0 (negative squared norms)."""
        params, dataset, loss_fn = linear_setup
        loader = DataLoader(dataset, batch_size=64)
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        assert np.all(scores <= 0)

    def test_perfect_model_has_near_zero_scores(self, linear_setup):
        """Perfect model: gradient norms near 0, so scores near 0."""
        params, dataset, loss_fn = linear_setup
        loader = DataLoader(dataset, batch_size=64)
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        # Perfect model has zero loss => zero gradient => scores near 0
        assert np.all(scores > -1e-3)

    def test_random_model_has_larger_norms(self, linear_setup):
        """Random model has larger gradient norms than perfect model."""
        params, dataset, loss_fn = linear_setup
        loader = DataLoader(dataset, batch_size=64)

        scores_perfect = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )

        random_params = torch.randn_like(params) * 10
        scores_random = gradient_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )

        # Random model has larger gradient norms => more negative scores
        assert np.mean(scores_perfect) > np.mean(scores_random)

    def test_batch_size_does_not_affect_result(self, linear_setup):
        """Test that different batch sizes give same scores."""
        params, dataset, loss_fn = linear_setup
        scores_32 = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=DataLoader(dataset, batch_size=32),
        )
        scores_128 = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=DataLoader(dataset, batch_size=128),
        )
        np.testing.assert_allclose(scores_32, scores_128, atol=1e-5)

    def test_single_batch_argnum(self):
        """Test with a single batch argument."""
        torch.manual_seed(42)
        X = torch.randn(30, 5)
        params = torch.randn(5)

        def loss_fn(params, x):
            return (x @ params).pow(2).sum()

        loader = DataLoader(TensorDataset(X), batch_size=16)
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        assert scores.shape == (30,)
        assert np.all(scores <= 0)


class TestGradientScoresValidation:
    """Tests for input validation."""

    def test_batch_argnums_includes_zero_raises(self):
        """batch_argnums=(0,) must raise since position 0 is params."""
        torch.manual_seed(42)
        X = torch.randn(10, 4)
        params = torch.randn(4)

        def loss_fn(x, params):
            return (x @ params).pow(2).sum()

        loader = DataLoader(TensorDataset(X), batch_size=10)

        with pytest.raises(ValueError, match="must not be in batch_argnums"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(0,),
                dataloader=loader,
            )

    def test_empty_batch_argnums_raises(self):
        """Empty batch_argnums must raise."""
        torch.manual_seed(42)
        params = torch.randn(4)
        loader = DataLoader(TensorDataset(torch.randn(10, 4)), batch_size=10)

        def loss_fn(params, x):
            return (x @ params).pow(2).sum()

        with pytest.raises(ValueError, match="non-empty"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(),
                dataloader=loader,
            )


class TestGradientScoresReference:
    """Tests for reference_scores subtraction."""

    def test_reference_scores_subtraction(self):
        """Test that reference_scores are subtracted from current scores."""
        torch.manual_seed(42)
        dim = 8
        n = 50

        tokens = torch.randn(n, dim)
        dataset = TensorDataset(tokens)
        params = torch.randn(dim)
        ref_params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        loader = DataLoader(dataset, batch_size=32)

        ref_scores = gradient_scores(
            loss_fn,
            ref_params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
            reference_scores=ref_scores,
        )

        # Manually compute expected: raw - ref
        raw_scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        expected = raw_scores - ref_scores
        np.testing.assert_allclose(scores, expected, atol=1e-5)

    def test_reference_scores_shape_mismatch(self):
        """Test that mismatched reference_scores raises ValueError."""
        torch.manual_seed(42)
        dim = 8
        tokens = torch.randn(10, dim)
        dataset = TensorDataset(tokens)
        params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        loader = DataLoader(dataset, batch_size=32)
        wrong_ref = np.zeros(5)

        with pytest.raises(ValueError, match="reference_scores shape"):
            gradient_scores(
                loss_fn,
                params,
                batch_argnums=(1,),
                dataloader=loader,
                reference_scores=wrong_ref,
            )


class TestGradientScoresPyTree:
    """Tests for PyTree (dict) parameter support."""

    def test_dict_params(self):
        """Test gradient_scores with dict parameters (common for models)."""
        torch.manual_seed(42)
        n = 20
        X = torch.randn(n, 4)
        params = {"w": torch.randn(4, 3), "b": torch.randn(3)}

        def loss_fn(params, x):
            return F.mse_loss(x @ params["w"] + params["b"], torch.zeros(3))

        loader = DataLoader(TensorDataset(X), batch_size=8)
        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        assert scores.shape == (n,)
        assert np.all(scores <= 0)


class TestEndToEnd:
    """Tests for end-to-end workflow: coin_flip -> gradient_scores -> one_run."""

    def test_full_workflow(self):
        """Test the full coin_flip -> gradient_scores -> one_run flow."""
        torch.manual_seed(42)
        input_dim = 10
        n_samples = 200

        X = torch.randn(n_samples, input_dim)
        true_w = torch.randn(input_dim)
        y = X @ true_w
        dataset = TensorDataset(X, y)

        # Train to convergence on a subset (members)
        params = true_w.clone()

        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        loader = DataLoader(
            Subset(dataset, cf.canary_indices.tolist()),
            batch_size=32,
        )

        def loss_fn(params, x, y):
            return F.mse_loss(x @ params, y, reduction="sum")

        scores = gradient_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        assert scores.shape == (len(cf.canary_indices),)

        estimate = one_run(scores, coin_flip=cf)
        # Perfect model should give a valid estimate
        assert estimate.eps_delta().epsilon_at(significance=0.05, delta=1e-5) >= 0.0
