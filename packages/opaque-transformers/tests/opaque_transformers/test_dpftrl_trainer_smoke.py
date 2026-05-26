# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Fast end-to-end smoke tests for DPTrainer + DP-FTRL.

Uses a tiny embedded LM (no HF model load) so the full
mechanism-dispatch surface — strategy construction, amplifier wiring,
noise-function lifecycle, sampler dispatch, and checkpoint+resume —
can be exercised in seconds.

The slower GPT-2-based parity is covered in
``tests/validation/test_dp_ftrl_trainer.py``; this file pins the
in-package CI signal so regressions surface immediately.
"""

from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import Dataset

from opaque.api.transformers.trainer._dp_trainer import DPTrainer
from opaque.transformers import TrainingArguments


class _TinyLM(torch.nn.Module):
    """8-dim causal-LM toy model — fast to train end-to-end on CPU."""

    def __init__(self, vocab: int = 32, dim: int = 8) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, input_ids: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        hidden = self.embed(input_ids)
        logits = self.head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), input_ids.reshape(-1)
        )
        return {"loss": loss, "logits": logits}


class _TinyDS(Dataset):
    def __init__(self, n: int = 64, seq: int = 8, vocab: int = 32) -> None:
        torch.manual_seed(0)
        self._data = torch.randint(0, vocab, (n, seq))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self._data[i]}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.stack([b["input_ids"] for b in batch])}


# Mellum-shaped defaults are tuned for real training; with a 16-step
# smoke run the strategy hyperparameters must be small enough to
# satisfy per-strategy validity constraints (e.g. BandMF requires
# ``bands <= n_steps``).  These keep the test surface fast while staying
# in legitimate mechanism territory.
_MF_TEST_KWARGS: dict[str, dict[str, object]] = {
    "mf_band": {"bands": 4},
    "mf_blt": {"max_buffers": 4},
    "mf_bisr": {"bandwidth": 4},
    "mf_bsr": {"bandwidth": 4, "alpha": 1.0, "beta": 0.9},
    "mf_lambda_cgd": {"lambda_": 0.5},
    "mf_identity": {},
}


def _args(
    *,
    output_dir: str,
    mechanism: str,
    max_steps: int,
    save_steps: int | None = None,
) -> TrainingArguments:
    kwargs = (
        dict(_MF_TEST_KWARGS[mechanism]) if mechanism in _MF_TEST_KWARGS else {}
    )
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        max_steps=max_steps,
        save_steps=save_steps,
        save_strategy="steps" if save_steps else "no",
        privacy_noise_mechanism=mechanism,
        privacy_noise_mechanism_kwargs=kwargs,
        privacy_noise_multiplier=1.0,
        clipping_norm=1.0,
        learning_rate=1e-3,
        optim="sgd",
        report_to=[],
        eval_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
        seed=0,
    )


_MF_MECHANISMS = pytest.mark.parametrize(
    "mechanism",
    ["mf_identity", "mf_band", "mf_blt", "mf_bisr", "mf_bsr", "mf_lambda_cgd"],
)


class TestDpFtrlTrain:
    @_MF_MECHANISMS
    def test_trains_for_each_mechanism(self, tmp_path, mechanism):
        # mf_identity is the only mechanism whose tiny smoke-test budget
        # (max_steps=4) doesn't trip a per-strategy validity guard
        # (BandMF bands<=n_steps, BallsInBins n_steps%num_bins==0).
        # For the others we use 16 steps to satisfy num_bins=16 (from
        # dataset_size=64 / batch_size=4) and BandMF bands=4 <= 16.
        max_steps = 4 if mechanism == "mf_identity" else 16
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
        )
        torch.manual_seed(0)
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        # BallsInBinsSampler skips empty bins (random partition of small
        # datasets occasionally leaves bins empty), so the realised step
        # count can fall short of ``max_steps``.  All other mechanisms
        # hit ``max_steps`` exactly.
        assert 0 < out.global_step <= max_steps
        # Every MF mechanism reports the same privacy metric surface.
        assert "privacy_epsilon" in out.metrics
        assert out.metrics["privacy_noise_multiplier"] == pytest.approx(1.0)


class TestDpFtrlSamplerDispatch:
    @pytest.mark.parametrize(
        "mechanism,expected_sampler_module_name",
        [
            ("mf_identity", "opaque.api.dpsgd.sampling._poisson"),
            ("mf_band", "opaque.api.dpftrl.sampling._b_min_sep"),
            ("mf_blt", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_bisr", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_bsr", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_lambda_cgd", "opaque.api.dpftrl.sampling._balls_in_bins"),
        ],
    )
    def test_sampler_dispatched_via_auto(
        self, tmp_path, mechanism, expected_sampler_module_name
    ):
        # Argument validation already pins ``sampling_mode``; this test
        # confirms the trainer's ``get_train_dataloader`` builds the
        # right concrete sampler class.  Snapshot the sampler with an
        # ``on_step_begin`` callback (fires after the dataloader and
        # sampler are constructed).
        max_steps = 4 if mechanism == "mf_identity" else 16
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
        )
        torch.manual_seed(0)
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        captured: dict[str, type] = {}

        from transformers import TrainerCallback

        class _Capture(TrainerCallback):
            def on_step_begin(self, args_, state_, control_, **_kw):
                ctx = getattr(trainer, "_ctx", None)
                if ctx is not None and ctx.current_sampler is not None:
                    captured.setdefault("cls", type(ctx.current_sampler))

        trainer.add_callback(_Capture())
        trainer.train()
        assert "cls" in captured, "sampler was not captured at on_step_begin"
        assert captured["cls"].__module__ == expected_sampler_module_name


class TestDpFtrlCheckpointRoundTrip:
    @pytest.mark.parametrize(
        "mechanism,max_steps",
        [
            ("mf_identity", 8),
            ("mf_band", 16),
            ("mf_blt", 16),
            ("mf_bisr", 16),
            ("mf_bsr", 16),
            ("mf_lambda_cgd", 16),
        ],
    )
    def test_resume_from_midtrain_checkpoint(self, tmp_path, mechanism, max_steps):
        outdir = tmp_path / mechanism
        ds = _TinyDS()

        args1 = _args(
            output_dir=str(outdir),
            mechanism=mechanism,
            max_steps=max_steps,
            save_steps=4,
        )
        torch.manual_seed(0)
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=args1,
            train_dataset=ds,
            data_collator=_collate,
        )
        out1 = trainer1.train()
        # BallsInBinsSampler may skip empty bins under random partitions
        # of small datasets — the realised step count can fall short of
        # ``max_steps``.  Still assert it ran far enough for at least
        # one mid-train checkpoint to land.
        assert out1.global_step >= 8

        ckpts = sorted(
            [d for d in os.listdir(outdir) if d.startswith("checkpoint-")],
            key=lambda d: int(d.split("-")[1]),
        )
        assert len(ckpts) >= 2, f"expected mid-train + end ckpts, got {ckpts}"
        mid_ckpt = str(outdir / ckpts[-2])

        outdir2 = tmp_path / f"{mechanism}-resumed"
        args2 = _args(
            output_dir=str(outdir2),
            mechanism=mechanism,
            max_steps=max_steps,
            save_steps=4,
        )
        torch.manual_seed(0)
        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=args2,
            train_dataset=ds,
            data_collator=_collate,
        )
        out2 = trainer2.train(resume_from_checkpoint=mid_ckpt)
        # Resume converges to the same realised step count as the
        # original (BallsInBins skips empty bins; sampler determinism
        # is bit-exact across runs at the same data_seed).
        assert out2.global_step == out1.global_step
        # Same mechanism + total steps + multiplier ⇒ deterministic ε.
        assert out1.metrics["privacy_epsilon"] == pytest.approx(
            out2.metrics["privacy_epsilon"], rel=1e-3
        )


class TestGaussianPathUnchanged:
    """Sanity: DP-SGD path still works (no regression from MF wiring)."""

    def test_gaussian_train(self, tmp_path):
        args = _args(
            output_dir=str(tmp_path / "gaussian"),
            mechanism="gaussian",
            max_steps=4,
        )
        torch.manual_seed(0)
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert out.global_step == 4
        assert "privacy_epsilon" in out.metrics
