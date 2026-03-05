"""Tests for auditing.loss_scores."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset

import opaque.auditing as auditing
from opaque.auditing import CoinFlip, OneRunEstimate, loss_scores, one_run
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


class TestLossScores:
    """Tests for auditing.loss_scores."""

    def test_basic_scoring(self, linear_setup):
        """Test that loss_scores returns correct shape."""
        params, dataset, loss_fn = linear_setup
        scores = loss_scores(
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
        scores = loss_scores(
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
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )
        # Perfect model: losses near 0, scores near 0 (but negative of near-0)
        assert np.all(scores <= 1e-3)

    def test_batch_size_does_not_affect_result(self, linear_setup):
        """Test that different batch sizes give same scores."""
        params, dataset, loss_fn = linear_setup
        scores_32 = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
            batch_size=32,
        )
        scores_128 = loss_scores(
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

        scores_trained = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )
        scores_random = loss_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            dataset=dataset,
        )

        assert np.mean(scores_trained) > np.mean(scores_random)


class TestLossScoresSingleArg:
    """Tests for loss_scores with single batch arg (HF-like pattern)."""

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
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
        )
        assert scores.shape == (100,)

    def test_with_batch_unpack(self, single_arg_setup):
        """Test scoring with custom batch_unpack function."""
        params, dataset, loss_fn = single_arg_setup
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
            batch_unpack=lambda b: (b[0],),
        )
        assert scores.shape == (100,)


class TestLossScoresDictBatch:
    """Tests for loss_scores with dict-style batches (HuggingFace pattern)."""

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
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
            collate_fn=collate_fn,
            batch_unpack=lambda b: (b["input_ids"],),
        )
        assert scores.shape == (50,)

    def test_dict_batch_without_unpack_raises(self, dict_dataset_setup):
        """Test that dict-style batches without batch_unpack raise TypeError."""
        params, dataset, loss_fn, collate_fn = dict_dataset_setup
        with pytest.raises(TypeError, match="batch_unpack"):
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1,),
                dataset=dataset,
                collate_fn=collate_fn,
            )


class TestEndToEnd:
    """Tests for end-to-end workflow: coin_flip -> loss_scores -> one_run."""

    def test_full_workflow(self, linear_setup):
        """Test the full coin_flip -> loss_scores -> one_run flow."""
        params, dataset, loss_fn = linear_setup

        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataset=dataset,
            indices=cf.canary_indices,
        )

        estimate = one_run(scores, coin_flip=cf)

        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in + estimate.n_out == 50
        assert 0.0 <= estimate.auc() <= 1.0
        assert estimate.epsilon_at(delta=0.0) >= 0.0

        s = estimate.summary()
        assert "one-run" in s
        assert "Audit Summary" in s

    def test_with_collate_fn(self):
        """Test workflow with collate_fn and batch_unpack."""
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

        cf = auditing.coin_flip(dataset, num_canaries=20, key=key(42))
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataset=dataset,
            indices=cf.canary_indices,
            collate_fn=collate_fn,
            batch_unpack=lambda b: (b["input_ids"],),
        )

        estimate = one_run(scores, coin_flip=cf)
        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in + estimate.n_out == 20
