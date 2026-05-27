# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DP-FTRL helper module (``_dpftrl``).

Covers strategy construction, amplifier wiring, per-step composition,
and sampler dispatch — the pure-function surface that the trainer
calls into.  End-to-end DPTrainer + DP-FTRL is covered by
``tests/validation/test_dp_ftrl_trainer.py``.
"""

from __future__ import annotations

import pytest

import torch
from torch.utils.data import Dataset

from opaque.api.transformers.trainer import _dpftrl
from opaque.dpftrl import (
    BallsInBinsSampler,
    BMinSepSampler,
    CyclicPoissonSampler,
    SequentialBatchSampler,
)
from opaque.api.accounting.dpftrl._base import DpFtrlProcess
from opaque.dpftrl.noise.types import (
    BandMfStrategy,
    BisrStrategy,
    BltStrategy,
    BsrStrategy,
    IdentityStrategy,
    LambdaCgdStrategy,
)
from opaque.dpsgd.sampling import PoissonSampler
from opaque.random import key


class _ListDataset(Dataset):
    def __init__(self, n: int = 32) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int) -> int:
        return i


class TestBuildStrategyLrSchedule:
    """``lr_schedule`` auto-injection for BandMF / BLT only."""

    def test_band_mf_receives_lr_schedule(self):
        from opaque.scheduling import cosine_schedule

        sched = cosine_schedule(1e-3, 0.0, 1000)
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4}, lr_schedule=sched)
        assert isinstance(strategy, BandMfStrategy)
        assert strategy.lr_schedule is sched

    def test_blt_receives_lr_schedule(self):
        from opaque.scheduling import cosine_schedule

        sched = cosine_schedule(1e-3, 0.0, 1000)
        strategy = _dpftrl.build_strategy(
            "mf_blt", {"max_buffers": 4}, lr_schedule=sched
        )
        assert isinstance(strategy, BltStrategy)
        assert strategy.lr_schedule is sched

    def test_other_mechanisms_ignore_lr_schedule(self):
        # BiSR/BSR/lambda_cgd/identity don't accept an lr_schedule kwarg;
        # build_strategy must not pass one through (no TypeError from the
        # factory).
        from opaque.scheduling import cosine_schedule

        sched = cosine_schedule(1e-3, 0.0, 1000)
        for mech, kw in [
            ("mf_bisr", {"bandwidth": 4}),
            ("mf_bsr", {"bandwidth": 4, "alpha": 1.0, "beta": 0.9}),
            ("mf_lambda_cgd", {"lambda_": 0.5}),
            ("mf_identity", {}),
        ]:
            strategy = _dpftrl.build_strategy(mech, kw, lr_schedule=sched)
            assert not hasattr(strategy, "lr_schedule") or strategy.lr_schedule is None

    def test_user_kwarg_wins_over_auto_injection(self):
        from opaque.scheduling import constant_schedule, cosine_schedule

        live = cosine_schedule(1e-3, 0.0, 1000)
        user = constant_schedule(5e-4)
        strategy = _dpftrl.build_strategy(
            "mf_band",
            {"bands": 4, "lr_schedule": user},
            lr_schedule=live,
        )
        assert strategy.lr_schedule is user


class TestBuildStrategy:
    @pytest.mark.parametrize(
        "mechanism,kwargs,cls",
        [
            ("mf_band", {"bands": 4}, BandMfStrategy),
            ("mf_blt", {"max_buffers": 4}, BltStrategy),
            ("mf_bisr", {"bandwidth": 4}, BisrStrategy),
            ("mf_bsr", {"bandwidth": 4, "alpha": 1.0, "beta": 0.9}, BsrStrategy),
            ("mf_lambda_cgd", {"lambda_": 0.5}, LambdaCgdStrategy),
            ("mf_identity", {}, IdentityStrategy),
        ],
    )
    def test_builds_strategy(self, mechanism, kwargs, cls):
        strategy = _dpftrl.build_strategy(mechanism, kwargs)
        assert isinstance(strategy, cls)

    def test_identity_ignores_kwargs(self):
        strategy = _dpftrl.build_strategy("mf_identity", {"ignored": 1})
        assert isinstance(strategy, IdentityStrategy)


class TestBuildAmplifierFactory:
    def test_band_poisson(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="poisson",
            strategy=strategy,
            sample_rate=0.05,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        proc = amp(1.0)
        assert isinstance(proc, DpFtrlProcess)
        assert proc.n_steps == 100

    def test_band_b_min_sep(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="b_min_sep",
            strategy=strategy,
            sample_rate=0.05,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        proc = amp(1.0)
        assert proc.n_steps == 100
        assert proc.min_sep == 4

    def test_blt_balls_in_bins(self):
        strategy = _dpftrl.build_strategy("mf_blt", {"max_buffers": 4})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="balls_in_bins",
            strategy=strategy,
            sample_rate=0.1,
            n_steps=100,
            num_bins=10,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        proc = amp(1.0)
        assert proc.n_steps == 100

    def test_unknown_sampling_mode_raises(self):
        strategy = _dpftrl.build_strategy("mf_band", {"bands": 4})
        with pytest.raises(ValueError, match="sampling_mode"):
            _dpftrl.build_amplifier_factory(
                sampling_mode="cyclic_poisson",  # no accountant amplifier
                strategy=strategy,
                sample_rate=0.05,
                n_steps=100,
                num_bins=10,
                dataset_size=1000,
                truncated_batch_size=None,
            )


class TestPerStepWrapper:
    def test_wraps_to_per_step(self):
        strategy = _dpftrl.build_strategy("mf_identity", {})
        amp = _dpftrl.build_amplifier_factory(
            sampling_mode="poisson",
            strategy=strategy,
            sample_rate=0.01,
            n_steps=50,
            num_bins=5,
            dataset_size=1000,
            truncated_batch_size=None,
        )
        step_factory = _dpftrl.build_step_mechanism_factory(amp)
        from opaque.api.accounting.dpftrl.composition._per_step import PerStep

        assert isinstance(step_factory(1.0), PerStep)


class TestBuildSampler:
    def test_poisson(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="poisson",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mechanism_kwargs=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, PoissonSampler)

    def test_b_min_sep_sources_bands_from_mechanism_kwargs(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="b_min_sep",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mechanism_kwargs={"bands": 4},
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, BMinSepSampler)
        assert sampler.bands == 4

    def test_balls_in_bins(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="balls_in_bins",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mechanism_kwargs=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, BallsInBinsSampler)
        assert sampler.num_bins == 4

    def test_cyclic_poisson(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="cyclic_poisson",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mechanism_kwargs={"bands": 4},
            num_bins=4,
            expected_batch_size=4,
        )
        assert isinstance(sampler, CyclicPoissonSampler)

    def test_sequential(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="sequential",
            dataset=dataset,
            sample_rate=0.1,
            n_steps=8,
            key=key(0),
            sampling_kwargs=None,
            mechanism_kwargs=None,
            num_bins=4,
            expected_batch_size=8,
        )
        assert isinstance(sampler, SequentialBatchSampler)

    def test_unknown_sampling_mode_raises(self):
        dataset = _ListDataset(64)
        with pytest.raises(ValueError, match="Unknown sampling_mode"):
            _dpftrl.build_sampler(
                sampling_mode="nope",
                dataset=dataset,
                sample_rate=0.1,
                n_steps=8,
                key=key(0),
                sampling_kwargs=None,
                mechanism_kwargs=None,
                num_bins=4,
                expected_batch_size=4,
            )

    def test_poisson_honors_truncated_batch_size(self):
        dataset = _ListDataset(64)
        sampler = _dpftrl.build_sampler(
            sampling_mode="poisson",
            dataset=dataset,
            sample_rate=0.5,
            n_steps=4,
            key=key(0),
            sampling_kwargs={"truncated_batch_size": 8},
            mechanism_kwargs=None,
            num_bins=4,
            expected_batch_size=4,
        )
        assert sampler.truncated_batch_size == 8
