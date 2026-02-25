"""Tests for auditing.score and auditing.evaluate."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import opaque.auditing as auditing
from opaque.auditing import AuditResult, score
from opaque.random import key


@pytest.fixture()
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


class TestScore:
    """Tests for auditing.score (replacement for score_by_loss)."""

    def test_basic_scoring(self, linear_setup):
        """Test that score returns correct shape."""
        params, dataset, loss_fn = linear_setup
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )
        assert scores.shape == (200,)

    def test_scoring_with_indices(self, linear_setup):
        """Test scoring a subset via indices."""
        params, dataset, loss_fn = linear_setup
        indices = np.array([0, 10, 20, 30, 40])
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
            indices=indices,
        )
        assert scores.shape == (5,)

    def test_scores_are_negative_loss(self, linear_setup):
        """Test that scores are negated losses (higher = better fit)."""
        params, dataset, loss_fn = linear_setup
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )
        # Perfect model: losses near 0, scores near 0 (but negative of near-0)
        # All scores should be <= 0 (negative loss)
        assert np.all(scores <= 1e-3)  # Near-zero losses

    def test_batch_size_does_not_affect_result(self, linear_setup):
        """Test that different batch sizes give same scores."""
        params, dataset, loss_fn = linear_setup
        scores_32 = score(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
            batch_size=32,
        )
        scores_128 = score(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
            batch_size=128,
        )
        np.testing.assert_allclose(scores_32, scores_128, atol=1e-5)

    def test_trained_vs_untrained(self, linear_setup):
        """Test that trained model gives higher scores than random."""
        params, dataset, loss_fn = linear_setup
        random_params = torch.randn_like(params)

        scores_trained = score(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )
        scores_random = score(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )

        # Trained model should have higher scores (lower loss)
        assert np.mean(scores_trained) > np.mean(scores_random)


class TestScoreSingleArg:
    """Tests for score with single batch arg (HF-like pattern)."""

    @pytest.fixture()
    def single_arg_setup(self):
        """Create a setup where loss_fn takes (params, tokens)."""
        torch.manual_seed(42)
        n_samples = 100
        dim = 8

        tokens = torch.randn(n_samples, dim)
        dataset = TensorDataset(tokens)
        params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        return params, dataset, loss_fn

    def test_single_batch_arg(self, single_arg_setup):
        """Test scoring with a single batched argument."""
        params, dataset, loss_fn = single_arg_setup
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
        )
        assert scores.shape == (100,)

    def test_with_batch_unpack(self, single_arg_setup):
        """Test scoring with custom batch_unpack function."""
        params, dataset, loss_fn = single_arg_setup
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
            batch_unpack=lambda b: (b[0],),
        )
        assert scores.shape == (100,)


class TestScoreDictBatch:
    """Tests for score with dict-style batches (HuggingFace pattern)."""

    @pytest.fixture()
    def dict_dataset_setup(self):
        """Create a setup with a dict-style dataset and collate_fn."""
        torch.manual_seed(42)
        n_samples = 50
        dim = 8

        tokens = torch.randn(n_samples, dim)

        class DictDataset:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {"input_ids": self.data[idx]}

        dataset = DictDataset(tokens)
        params = torch.randn(dim)

        def loss_fn(params, tokens):
            return F.mse_loss(tokens @ params, torch.zeros(1), reduction="sum")

        def collate_fn(batch):
            return {"input_ids": torch.stack([b["input_ids"] for b in batch])}

        return params, dataset, loss_fn, collate_fn

    def test_dict_batch_with_unpack(self, dict_dataset_setup):
        """Test dict-style batches with batch_unpack."""
        params, dataset, loss_fn, collate_fn = dict_dataset_setup
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
            collate_fn=collate_fn,
            batch_unpack=lambda b: (b["input_ids"],),
        )
        assert scores.shape == (50,)

    def test_dict_batch_auto_unpack(self, dict_dataset_setup):
        """Test dict-style batches without batch_unpack (auto-detect)."""
        params, dataset, loss_fn, collate_fn = dict_dataset_setup
        scores = score(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
            collate_fn=collate_fn,
        )
        assert scores.shape == (50,)


class TestEvaluate:
    """Tests for auditing.evaluate."""

    def test_evaluate_returns_audit_result(self, linear_setup):
        """Test that evaluate returns an AuditResult."""
        params, dataset, loss_fn = linear_setup
        exp = auditing.setup(dataset, num_canaries=50, key=key(42))

        # Train briefly (just use the true params as "trained")
        audit = auditing.evaluate(
            exp,
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )

        assert isinstance(audit, AuditResult)
        assert audit.n_in + audit.n_out == 50

    def test_evaluate_from_coin_flip(self, linear_setup):
        """Test that evaluate result defaults to one_run method."""
        params, dataset, loss_fn = linear_setup
        exp = auditing.setup(dataset, num_canaries=50, key=key(42))

        audit = auditing.evaluate(
            exp,
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )

        assert audit._from_coin_flip is True
        # epsilon_at should use one_run by default
        eps = audit.epsilon_at(delta=0.0)
        assert isinstance(eps, float)

    def test_end_to_end_workflow(self, linear_setup):
        """Test the full auditing.setup -> train -> auditing.evaluate flow."""
        _, dataset, loss_fn = linear_setup

        # Setup
        experiment = auditing.setup(dataset, num_canaries=50, key=key(42))
        train_data = experiment.subset(dataset)
        assert len(train_data) == 200 - len(experiment.out_indices)

        # "Train" on subset (just use fresh random params)
        torch.manual_seed(0)
        params = torch.randn(10)

        # Evaluate
        audit = auditing.evaluate(
            experiment,
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )

        # Should have valid metrics
        assert 0.0 <= audit.auc() <= 1.0
        assert audit.epsilon_at(delta=0.0) >= 0.0

        # Summary should work
        s = audit.summary()
        assert "one-run" in s
        assert "Audit Summary" in s
