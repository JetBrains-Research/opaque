"""Tests for auditing.loss_scores."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

import opaque.auditing as auditing
from opaque.auditing import loss_scores, one_run
from opaque.auditing.types import OneRunEstimate
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


class TestLossScores:
    """Tests for auditing.loss_scores."""

    @pytest.mark.parametrize(
        ("batch_argnums", "message"),
        [
            ((2, 1), r"must be sorted, got \(2, 1\)"),
            ((1, 1), r"must be unique, got \(1, 1\)"),
            ((-1,), r"must be non-negative, got \(-1,\)"),
            ((2,), r"out of range.*got \(2,\)"),
        ],
    )
    def test_invalid_batch_argnums_raise(self, batch_argnums, message):
        """Invalid batch positions are rejected before scoring."""
        params = torch.tensor(1.0)
        loader = DataLoader(TensorDataset(torch.tensor([1.0])), batch_size=1)

        def loss_fn(params, x):
            return (params - x).square()

        with pytest.raises(ValueError, match=message):
            loss_scores(
                loss_fn,
                params,
                batch_argnums=batch_argnums,
                dataloader=loader,
            )

    def test_basic_scoring(self, linear_setup):
        """Test that loss_scores returns correct shape."""
        params, dataset, loss_fn = linear_setup
        loader = DataLoader(dataset, batch_size=64)
        scores = loss_scores(
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
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        assert scores.shape == (5,)

    def test_scores_are_negative_loss(self, linear_setup):
        """Test that scores are negated losses (higher = better fit)."""
        params, dataset, loss_fn = linear_setup
        loader = DataLoader(dataset, batch_size=64)
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
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
            dataloader=DataLoader(dataset, batch_size=32),
        )
        scores_128 = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=DataLoader(dataset, batch_size=128),
        )
        np.testing.assert_allclose(scores_32, scores_128, atol=1e-5)

    def test_trained_vs_untrained(self, linear_setup):
        """Test that trained model gives higher scores than random."""
        params, dataset, loss_fn = linear_setup
        random_params = torch.randn_like(params)
        loader = DataLoader(dataset, batch_size=64)

        scores_trained = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )
        scores_random = loss_scores(
            loss_fn,
            random_params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )

        assert np.mean(scores_trained) > np.mean(scores_random)


class TestLossScoresSingleArg:
    """Tests for loss_scores with single batch arg (HF-like pattern)."""

    @pytest.fixture
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
        loader = DataLoader(dataset, batch_size=32)
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        assert scores.shape == (100,)


class TestLossScoresDictBatch:
    """Tests for loss_scores with dict-style batches (HuggingFace pattern)."""

    @pytest.fixture
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
            """Collate dict batches into tensor tuples (HF pattern)."""
            return (torch.stack([b["input_ids"] for b in batch]),)

        return params, dataset, loss_fn, collate_fn

    def test_dict_batch_with_collate(self, dict_dataset_setup):
        """Test dict-style batches with collate that returns tuples."""
        params, dataset, loss_fn, collate_fn = dict_dataset_setup
        loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        assert scores.shape == (50,)


class TestReferenceScores:
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

        ref_scores = loss_scores(
            loss_fn,
            ref_params,
            batch_argnums=(1,),
            dataloader=loader,
        )
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
            reference_scores=ref_scores,
        )

        # Manually compute expected: -loss(params) - (-loss(ref_params))
        raw_scores = loss_scores(
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
            loss_scores(
                loss_fn,
                params,
                batch_argnums=(1,),
                dataloader=loader,
                reference_scores=wrong_ref,
            )


class TestEndToEnd:
    """Tests for end-to-end workflow: coin_flip -> loss_scores -> one_run."""

    def test_full_workflow(self, linear_setup):
        """Test the full coin_flip -> loss_scores -> one_run flow."""
        params, dataset, loss_fn = linear_setup

        cf = auditing.coin_flip(dataset, num_canaries=50, key=key(42))
        loader = DataLoader(
            Subset(dataset, cf.canary_indices.tolist()),
            batch_size=32,
        )
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1, 2),
            dataloader=loader,
        )

        estimate = one_run(scores, coin_flip=cf)

        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in + estimate.n_out == 50
        assert 0.0 <= estimate.attack_auc() <= 1.0
        assert estimate.eps_delta().epsilon_at(delta=0.0) >= 0.0
        assert estimate.attack_beta_at(alpha=0.1) >= 0.0

    def test_with_collate_fn(self):
        """Test workflow with custom collate_fn on DataLoader."""
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
            return (torch.stack([b["input_ids"] for b in batch]),)

        cf = auditing.coin_flip(dataset, num_canaries=20, key=key(42))
        loader = DataLoader(
            Subset(dataset, cf.canary_indices.tolist()),
            batch_size=16,
            collate_fn=collate_fn,
        )
        scores = loss_scores(
            loss_fn,
            params,
            batch_argnums=(1,),
            dataloader=loader,
        )

        estimate = one_run(scores, coin_flip=cf)
        assert isinstance(estimate, OneRunEstimate)
        assert estimate.n_in + estimate.n_out == 20
