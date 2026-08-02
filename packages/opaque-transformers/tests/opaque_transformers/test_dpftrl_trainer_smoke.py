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

import math
from pathlib import Path

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
    noise_multiplier: float = 1.0,
    clipping_norm: float | str = 1.0,
    sampling_mode: str = "auto",
) -> TrainingArguments:
    kwargs = dict(_MF_TEST_KWARGS[mechanism]) if mechanism in _MF_TEST_KWARGS else {}
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        max_steps=max_steps,
        save_steps=save_steps,
        save_strategy="steps" if save_steps else "no",
        privacy_noise_mechanism=mechanism,
        privacy_noise_mechanism_kwargs=kwargs,
        sampling_mode=sampling_mode,
        privacy_noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
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
        assert out.global_step == max_steps
        # Every MF mechanism reports the same privacy metric surface.
        assert "privacy_epsilon" in out.metrics
        assert out.metrics["privacy_noise_multiplier"] == pytest.approx(1.0)


class TestDpFtrlSamplerDispatch:
    @pytest.mark.parametrize(
        ("mechanism", "sampling_mode", "expected_sampler_module_name"),
        [
            ("mf_identity", "auto", "opaque.api.dpsgd.sampling._poisson"),
            (
                "mf_identity",
                "balls_in_bins",
                "opaque.api.dpftrl.sampling._balls_in_bins",
            ),
            (
                "gaussian",
                "random_allocation",
                "opaque.api.dpsgd.sampling._random_allocation",
            ),
            ("mf_band", "auto", "opaque.api.dpftrl.sampling._b_min_sep"),
            ("mf_blt", "auto", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_bisr", "auto", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_bsr", "auto", "opaque.api.dpftrl.sampling._balls_in_bins"),
            ("mf_lambda_cgd", "auto", "opaque.api.dpftrl.sampling._balls_in_bins"),
        ],
    )
    def test_sampler_dispatched_via_auto(
        self, tmp_path, mechanism, sampling_mode, expected_sampler_module_name
    ):
        # Argument validation already pins ``sampling_mode``; this test
        # confirms the trainer's ``get_train_dataloader`` builds the
        # right concrete sampler class.  Snapshot the sampler with an
        # ``on_step_begin`` callback (fires after the dataloader and
        # sampler are constructed).
        max_steps = 4 if mechanism == "mf_identity" and sampling_mode == "auto" else 16
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
            sampling_mode=sampling_mode,
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


class TestDpTrainerAllocationModes:
    @pytest.mark.parametrize(
        ("mechanism", "sampling_mode"),
        [
            ("gaussian", "random_allocation"),
            ("mf_identity", "balls_in_bins"),
        ],
    )
    def test_trains_complete_schedule(self, tmp_path, mechanism, sampling_mode):
        args = _args(
            output_dir=str(tmp_path / f"{mechanism}-{sampling_mode}"),
            mechanism=mechanism,
            sampling_mode=sampling_mode,
            max_steps=16,
        )
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )

        out = trainer.train()

        assert out.global_step == 16
        assert "privacy_epsilon" in out.metrics


class TestDpFtrlCheckpointRoundTrip:
    @pytest.mark.parametrize(
        ("mechanism", "max_steps"),
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
            [
                d.name
                for d in Path(outdir).iterdir()
                if d.name.startswith("checkpoint-")
            ],
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


class TestDpFtrlLrScheduleIntegration:
    """The optimizer LR schedule auto-flows into BandMF / BLT strategies.

    Pre-refactor (opaque.scheduling as a closure DSL), the trainer
    silently dropped the schedule because the strategy codec rejected
    callable fields.  Post-refactor, schedules are frozen recipe
    dataclasses, round-trip through the strategy codec via tagged
    sub-dicts, and the trainer's ``_setup_training`` auto-injects the
    live LR schedule into BandMF / BLT.
    """

    def test_band_mf_strategy_receives_schedule(self, tmp_path):
        # The trainer's live LR schedule should appear on
        # ``ctx.mf.strategy.lr_schedule``.  Snapshot via an
        # ``on_step_begin`` callback (fires after ``_setup_training``
        # populates ``_ctx``).
        from transformers import TrainerCallback

        from opaque.scheduling.types import ConstantSchedule, CosineSchedule

        captured: dict[str, type] = {}
        sentinel_trainer: dict[str, object] = {}

        class _Snap(TrainerCallback):
            def on_step_begin(self, args_, state_, ctrl_, **_kw):
                trainer = sentinel_trainer["trainer"]
                mf = getattr(trainer._ctx, "mf", None)
                if mf is not None and "ls_cls" not in captured:
                    captured["ls_cls"] = type(mf.strategy.lr_schedule)

        for sched_type, expected_cls in (
            ("cosine", CosineSchedule),
            ("constant", ConstantSchedule),
        ):
            args = TrainingArguments(
                output_dir=str(tmp_path / sched_type),
                per_device_train_batch_size=4,
                max_steps=16,
                privacy_noise_mechanism="mf_band",
                privacy_noise_mechanism_kwargs={"bands": 4},
                privacy_noise_multiplier=1.0,
                clipping_norm=1.0,
                learning_rate=1e-3,
                optim="sgd",
                lr_scheduler=sched_type,
                report_to=[],
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="no",
                disable_tqdm=True,
                use_cpu=True,
                seed=0,
            )
            torch.manual_seed(0)
            trainer = DPTrainer(
                model=_TinyLM(),
                args=args,
                train_dataset=_TinyDS(),
                data_collator=_collate,
            )
            sentinel_trainer["trainer"] = trainer
            captured.clear()
            trainer.add_callback(_Snap())
            trainer.train()
            assert captured.get("ls_cls") is expected_cls, (
                f"expected {expected_cls.__name__} for "
                f"lr_scheduler={sched_type!r}, got "
                f"{captured.get('ls_cls')}"
            )

    def test_resume_preserves_schedule_in_accountant(self, tmp_path):
        # Save mid-train, resume to completion, verify ε matches a
        # from-scratch run — the saved accountant.json must round-trip
        # the cosine schedule baked into the BandMfStrategy.
        outdir = tmp_path / "bandmf_cosine"
        kwargs = {
            "per_device_train_batch_size": 4,
            "max_steps": 16,
            "save_steps": 4,
            "privacy_noise_mechanism": "mf_band",
            "privacy_noise_mechanism_kwargs": {"bands": 4},
            "privacy_noise_multiplier": 1.0,
            "clipping_norm": 1.0,
            "learning_rate": 1e-3,
            "optim": "sgd",
            "lr_scheduler": "cosine",
            "report_to": [],
            "save_strategy": "steps",
            "eval_strategy": "no",
            "logging_strategy": "no",
            "disable_tqdm": True,
            "use_cpu": True,
            "seed": 0,
        }

        # From-scratch run.
        args = TrainingArguments(output_dir=str(outdir), **kwargs)
        torch.manual_seed(0)
        t1 = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out1 = t1.train()

        # Find a mid-train checkpoint.
        ckpts = sorted(
            [
                d.name
                for d in Path(outdir).iterdir()
                if d.name.startswith("checkpoint-")
            ],
            key=lambda d: int(d.split("-")[1]),
        )
        assert len(ckpts) >= 2
        mid = str(outdir / ckpts[-2])

        # Resume.
        outdir2 = tmp_path / "bandmf_cosine_resumed"
        args2 = TrainingArguments(output_dir=str(outdir2), **kwargs)
        torch.manual_seed(0)
        t2 = DPTrainer(
            model=_TinyLM(),
            args=args2,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out2 = t2.train(resume_from_checkpoint=mid)

        # Identical ε is the strongest signal that the accountant's
        # internal schedule survived the disk round-trip.
        assert out1.metrics["privacy_epsilon"] == pytest.approx(
            out2.metrics["privacy_epsilon"], rel=1e-9
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


class TestNonPrivateZeroNoise:
    """``privacy_noise_multiplier=0`` → non-DP baseline.

    The chosen mechanism and sampler are kept intact; the accountant
    composes a non-private step so ε=∞ is reported, and zero noise is
    added (clipping still applies unless disabled).
    """

    @pytest.mark.parametrize(
        ("mechanism", "max_steps"),
        [("gaussian", 4), ("mf_identity", 4), ("mf_band", 16)],
    )
    def test_zero_noise_reports_inf_epsilon(self, tmp_path, mechanism, max_steps):
        args = _args(
            output_dir=str(tmp_path / mechanism),
            mechanism=mechanism,
            max_steps=max_steps,
            noise_multiplier=0.0,
        )
        torch.manual_seed(0)
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()

        # ε=∞ is the legal non-private output; the multiplier round-trips
        # as a fixed 0.0 with the "fixed" provenance tag.
        assert out.global_step > 0
        assert math.isinf(out.metrics["privacy_epsilon"])
        assert out.metrics["privacy_noise_multiplier"] == 0.0
        assert trainer.state.privacy_resolved_noise_multiplier == 0.0
        assert trainer.state.privacy_calibration_source == "fixed"
        assert math.isfinite(out.metrics["train_loss"])

    def test_zero_noise_is_deterministic(self, tmp_path):
        # σ=0 ⇒ no randomness from the noise stream: two runs at the same
        # seed must produce identical weights.
        def _run(tag):
            args = _args(
                output_dir=str(tmp_path / tag),
                mechanism="gaussian",
                max_steps=4,
                noise_multiplier=0.0,
            )
            torch.manual_seed(0)
            trainer = DPTrainer(
                model=_TinyLM(),
                args=args,
                train_dataset=_TinyDS(),
                data_collator=_collate,
            )
            trainer.train()
            return {n: p.detach().clone() for n, p in trainer.model.named_parameters()}

        a, b = _run("a"), _run("b")
        for n in a:
            assert torch.equal(a[n], b[n]), f"param {n} differs across σ=0 runs"

    def test_disabled_clipping_zero_noise_no_nan(self, tmp_path):
        # clipping_norm=math.inf with σ=0 is true non-private SGD;
        # the 0*inf NaN hazard in the noise std must be guarded.
        args = _args(
            output_dir=str(tmp_path / "noclip"),
            mechanism="gaussian",
            max_steps=4,
            noise_multiplier=0.0,
            clipping_norm=math.inf,
        )
        assert math.isinf(args.clipping_norm)
        torch.manual_seed(0)
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert math.isfinite(out.metrics["train_loss"])
        for n, p in trainer.model.named_parameters():
            assert not torch.isnan(p).any(), f"NaN in param {n}"
