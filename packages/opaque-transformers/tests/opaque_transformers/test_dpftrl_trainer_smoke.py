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

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from opaque.api.transformers.trainer._dp_trainer import DPTrainer
from opaque.exceptions import CheckpointError, ConfigurationError, OperationError
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
    noise_multiplier: float | None = 1.0,
    target_epsilon: float | None = None,
    clipping_norm: float | str = 1.0,
    sampling_mode: str = "auto",
    sampling_kwargs: dict[str, object] | None = None,
    ignore_data_skip: bool = False,
    dataloader_num_workers: int = 0,
    auto_find_microbatch_size: bool = False,
    microbatch_size: int | None = None,
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
        sampling_kwargs=sampling_kwargs,
        ignore_data_skip=ignore_data_skip,
        auto_find_microbatch_size=auto_find_microbatch_size,
        microbatch_size=microbatch_size,
        privacy_noise_multiplier=noise_multiplier,
        privacy_target_epsilon=target_epsilon,
        clipping_norm=clipping_norm,
        learning_rate=1e-3,
        optim="sgd",
        report_to=[],
        eval_strategy="no",
        logging_strategy="no",
        disable_tqdm=True,
        use_cpu=True,
        seed=0,
        dataloader_num_workers=dataloader_num_workers,
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

    def test_step_logs_report_only_full_horizon_epsilon(self, tmp_path):
        args = _args(
            output_dir=str(tmp_path / "full-horizon-logs"),
            mechanism="mf_identity",
            max_steps=4,
        )
        args.logging_strategy = "steps"
        args.logging_steps = 1
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )

        out = trainer.train()
        logged = [
            row["privacy_epsilon"]
            for row in trainer.state.log_history
            if "privacy_epsilon" in row
        ]

        assert logged
        assert logged == pytest.approx([out.metrics["privacy_epsilon"]] * len(logged))


class TestDpFtrlSamplerDispatch:
    @pytest.mark.parametrize(
        ("mechanism", "expected_sampler_module_name"),
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


class TestDpTrainerAllocationModes:
    @pytest.mark.parametrize(
        ("mechanism", "sampling_mode", "max_steps", "allocation"),
        [
            ("gaussian", "k_out_of_t", 18, "block"),
            ("gaussian", "k_out_of_t", 10, "total"),
            ("mf_identity", "balls_in_bins", 16, None),
        ],
    )
    @pytest.mark.slow
    def test_trains_complete_schedule(
        self, tmp_path, mechanism, sampling_mode, max_steps, allocation
    ):
        kwargs = {}
        if sampling_mode == "k_out_of_t":
            kwargs["sampling_kwargs"] = {"k": 2, "allocation": allocation}
        args = _args(
            output_dir=str(tmp_path / f"{mechanism}-{sampling_mode}"),
            mechanism=mechanism,
            sampling_mode=sampling_mode,
            max_steps=max_steps,
            **kwargs,
        )
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        out = trainer.train()
        assert out.global_step == max_steps
        assert "privacy_epsilon" in out.metrics

    def test_callback_cannot_mutate_realized_participation(
        self,
        tmp_path,
        monkeypatch,
    ):
        from transformers import TrainerCallback

        from opaque.api.transformers.trainer import _dpftrl

        outdir = tmp_path / "immutable-participation"
        args = _args(
            output_dir=str(outdir),
            mechanism="mf_band",
            max_steps=16,
            save_steps=4,
        )
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        built: dict[str, object] = {}
        observed_in_order: list[bool] = []
        original_build_sampler = _dpftrl.build_sampler

        def recording_build_sampler(**kwargs):
            sampler = original_build_sampler(**kwargs)
            built.update(plan=kwargs["plan"], sampler=sampler)
            return sampler

        monkeypatch.setattr(_dpftrl, "build_sampler", recording_build_sampler)

        class _MutateSamplingArgs(TrainerCallback):
            def on_train_begin(self, args_, state_, control_, **_kwargs):
                args_.sampling_mode = "poisson"
                args_.sampling_kwargs["truncated_batch_size"] = 1
                args_.dataloader_in_order = False

            def on_step_begin(self, args_, state_, control_, **kwargs):
                observed_in_order.append(kwargs["train_dataloader"].in_order)

        trainer.add_callback(_MutateSamplingArgs())
        trainer.train()

        from opaque.api.transformers.trainer import _checkpoint as checkpoint
        from opaque.dpftrl import BMinSepSampler

        assert isinstance(built["sampler"], BMinSepSampler)
        plan = built["plan"]
        assert plan.sampling_mode == "b_min_sep"
        assert plan.sampling_kwargs == ()
        assert observed_in_order
        assert all(observed_in_order)
        saved = checkpoint.load_dp_runtime_state(
            str(outdir / "checkpoint-4" / checkpoint.DP_STATE_NAME)
        )
        assert saved.participation_plan == plan.to_state_dict()

    def test_oom_retry_reuses_pre_callback_invocation(self, tmp_path, monkeypatch):
        from transformers import TrainerCallback

        from opaque.api.transformers.trainer import _dpftrl

        trainer = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "oom-invocation"),
                mechanism="mf_band",
                max_steps=4,
                auto_find_microbatch_size=True,
                microbatch_size=4,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        plans = []
        strategy_bands = []
        original_build_sampler = _dpftrl.build_sampler
        original_build_strategy = _dpftrl.build_strategy

        def recording_build_sampler(**kwargs):
            plans.append(kwargs["plan"])
            return original_build_sampler(**kwargs)

        monkeypatch.setattr(_dpftrl, "build_sampler", recording_build_sampler)

        def recording_build_strategy(mechanism_kind, mechanism_kwargs, **kwargs):
            strategy_bands.append(mechanism_kwargs["bands"])
            return original_build_strategy(
                mechanism_kind,
                mechanism_kwargs,
                **kwargs,
            )

        monkeypatch.setattr(_dpftrl, "build_strategy", recording_build_strategy)
        original_training_step = trainer.training_step
        attempts = 0

        def oom_once(model, inputs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise torch.OutOfMemoryError("simulated first-attempt OOM")
            return original_training_step(model, inputs)

        monkeypatch.setattr(trainer, "training_step", oom_once)

        class _MutatePrivacyArgs(TrainerCallback):
            def on_train_begin(self, args_, state_, control_, **_kwargs):
                args_.privacy_noise_mechanism = "gaussian"
                args_.privacy_noise_mechanism_kwargs["bands"] = 1
                args_.sampling_mode = "poisson"
                args_.sampling_kwargs = {"truncated_batch_size": 1}
                args_.ignore_data_skip = True
                args_.dataloader_in_order = False

        trainer.add_callback(_MutatePrivacyArgs())
        result = trainer.train()

        assert result.global_step == 4
        assert len(plans) == 2
        assert plans[0] is plans[1]
        assert plans[0].mechanism_kind == "mf_band"
        assert plans[0].sampling_mode == "b_min_sep"
        assert strategy_bands == [4, 4]

    def test_uncopyable_schedule_only_blocks_automatic_retry(self, tmp_path):
        class _UncopyableSchedule:
            def __call__(self, _step):
                return 1e-3

            def __deepcopy__(self, _memo):
                raise TypeError("schedule owns an uncopyable backend handle")

        args = _args(
            output_dir=str(tmp_path / "no-retry"),
            mechanism="gaussian",
            max_steps=1,
        )
        args.lr_scheduler = _UncopyableSchedule()
        trainer = DPTrainer(
            model=_TinyLM(),
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        assert trainer.train().global_step == 1

        retry_args = _args(
            output_dir=str(tmp_path / "with-retry"),
            mechanism="gaussian",
            max_steps=1,
            auto_find_microbatch_size=True,
        )
        retry_args.lr_scheduler = _UncopyableSchedule()
        retry_trainer = DPTrainer(
            model=_TinyLM(),
            args=retry_args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        with pytest.raises(
            ConfigurationError,
            match="deepcopy-compatible",
        ):
            retry_trainer.train()

    def test_oom_after_committed_step_is_not_retried(self, tmp_path, monkeypatch):
        trainer = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "late-oom"),
                mechanism="gaussian",
                sampling_mode="poisson",
                max_steps=3,
                auto_find_microbatch_size=True,
                microbatch_size=4,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        original_training_step = trainer.training_step
        calls = 0

        def oom_on_second_step(model, inputs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise torch.OutOfMemoryError("simulated late OOM")
            return original_training_step(model, inputs)

        monkeypatch.setattr(trainer, "training_step", oom_on_second_step)

        with pytest.raises(
            OperationError,
            match="cannot retry after a privacy release became observable",
        ):
            trainer.train()
        assert calls == 2

    def test_first_step_optimizer_callback_oom_is_not_retried(self, tmp_path):
        from transformers import TrainerCallback

        trainer = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "optimizer-callback-oom"),
                mechanism="gaussian",
                sampling_mode="poisson",
                max_steps=2,
                auto_find_microbatch_size=True,
                microbatch_size=4,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        callbacks = 0

        class _OomAfterOptimizer(TrainerCallback):
            def on_optimizer_step(self, args_, state_, control_, **_kwargs):
                nonlocal callbacks
                callbacks += 1
                raise torch.OutOfMemoryError("post-optimizer callback OOM")

        trainer.add_callback(_OomAfterOptimizer())
        with pytest.raises(
            OperationError,
            match="cannot retry after a privacy release became observable",
        ):
            trainer.train()
        assert callbacks == 1

    def test_cleanup_oom_after_release_is_not_retried(self, tmp_path, monkeypatch):
        trainer = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "cleanup-oom"),
                mechanism="gaussian",
                sampling_mode="poisson",
                max_steps=1,
                auto_find_microbatch_size=True,
                microbatch_size=4,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        restores = 0

        def oom_restore(_params):
            nonlocal restores
            restores += 1
            raise torch.OutOfMemoryError("cleanup OOM")

        monkeypatch.setattr(trainer, "_restore_params", oom_restore)
        with pytest.raises(
            OperationError,
            match="cannot retry after a privacy release became observable",
        ):
            trainer.train()
        assert restores == 1

    def test_on_train_begin_cannot_mutate_executed_step(self, tmp_path, monkeypatch):
        from transformers import TrainerCallback

        outdir = tmp_path / "mutate-step-at-begin"
        trainer = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=2,
                save_steps=1,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )

        class _MutateStep(TrainerCallback):
            def on_train_begin(self, args_, state_, control_, **_kwargs):
                state_.global_step = 1

        trainer.add_callback(_MutateStep())
        monkeypatch.setattr(
            trainer,
            "training_step",
            lambda *_args, **_kwargs: pytest.fail("training step must not run"),
        )

        with pytest.raises(OperationError, match="private executed-step authority"):
            trainer.train()
        assert not list(outdir.glob("checkpoint-*"))

    def test_on_step_end_cannot_forge_checkpoint_progress(self, tmp_path):
        from transformers import TrainerCallback

        outdir = tmp_path / "mutate-step-before-save"
        trainer = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=2,
                save_steps=1,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )

        class _MutateStep(TrainerCallback):
            def on_step_end(self, args_, state_, control_, **_kwargs):
                state_.global_step = 99

        trainer.add_callback(_MutateStep())

        with pytest.raises(OperationError, match="private executed-step authority"):
            trainer.train()
        assert not list(outdir.glob("checkpoint-*"))


class TestDpFtrlCheckpointRoundTrip:
    def test_prefetch_checkpoint_cursor_tracks_executed_steps(
        self, tmp_path, monkeypatch
    ):
        outdir = tmp_path / "prefetched-poisson"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
                dataloader_num_workers=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        live_cursors: list[tuple[int, int]] = []
        original_snapshot = trainer1._execution_aligned_sampler_state

        def record_live_cursor(ctx):
            live_cursors.append(
                (trainer1.state.global_step, ctx.current_sampler.consumed)
            )
            return original_snapshot(ctx)

        monkeypatch.setattr(
            trainer1,
            "_execution_aligned_sampler_state",
            record_live_cursor,
        )
        trainer1.train()
        expected_params = {
            name: value.detach().clone()
            for name, value in trainer1.model.named_parameters()
        }

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        saved = checkpoint.load_dp_runtime_state(
            str(outdir / "checkpoint-2" / checkpoint.DP_STATE_NAME)
        )
        assert saved.sampler_cursor_origin == 0
        assert saved.sampler_state["consumed"] == 2
        assert (2, 4) in live_cursors

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "prefetched-poisson-resumed"),
                mechanism="gaussian",
                max_steps=4,
                dataloader_num_workers=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        result = trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))
        assert result.global_step == 4
        for name, value in trainer2.model.named_parameters():
            torch.testing.assert_close(
                value.detach(),
                expected_params[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_ignore_skip_checkpoint_can_resume_normally(self, tmp_path):
        first_out = tmp_path / "first-stream"
        ds = _TinyDS()
        DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(first_out),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        ).train()

        restarted_out = tmp_path / "restarted-stream"
        restarted = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(restarted_out),
                mechanism="gaussian",
                max_steps=6,
                save_steps=2,
                ignore_data_skip=True,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        restarted.train(resume_from_checkpoint=str(first_out / "checkpoint-2"))
        expected_params = {
            name: value.detach().clone()
            for name, value in restarted.model.named_parameters()
        }

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        chained_path = restarted_out / "checkpoint-4"
        chained = checkpoint.load_dp_runtime_state(
            str(chained_path / checkpoint.DP_STATE_NAME)
        )
        assert chained.sampler_cursor_origin == 2
        assert chained.sampler_state["consumed"] == 2

        resumed = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "normal-resume"),
                mechanism="gaussian",
                max_steps=6,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        result = resumed.train(resume_from_checkpoint=str(chained_path))
        assert result.global_step == 6
        for name, value in resumed.model.named_parameters():
            torch.testing.assert_close(
                value.detach(),
                expected_params[name],
                rtol=0.0,
                atol=0.0,
            )

    def test_callback_cannot_enable_ignore_skip_after_policy_capture(
        self, tmp_path, monkeypatch
    ):
        from transformers import TrainerCallback

        outdir = tmp_path / "capture-false"
        ds = _TinyDS()
        DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        ).train()

        resumed = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "capture-false-resumed"),
                mechanism="gaussian",
                max_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        restore_calls = 0
        original_restore = resumed._restore_sampler_for_resume

        def record_restore(*args, **kwargs):
            nonlocal restore_calls
            restore_calls += 1
            return original_restore(*args, **kwargs)

        monkeypatch.setattr(resumed, "_restore_sampler_for_resume", record_restore)

        class _EnableSkip(TrainerCallback):
            def on_train_begin(self, args_, state_, control_, **_kwargs):
                args_.ignore_data_skip = True

        resumed.add_callback(_EnableSkip())
        resumed.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))
        assert restore_calls == 1

    def test_callback_cannot_disable_ddp_ignore_skip_after_policy_capture(
        self, tmp_path, monkeypatch
    ):
        import dataclasses

        from transformers import TrainerCallback

        first_out = tmp_path / "capture-true"
        ds = _TinyDS()
        DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(first_out),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        ).train()

        resumed_out = tmp_path / "capture-true-resumed"
        resumed = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(resumed_out),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
                ignore_data_skip=True,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        resumed._ddp = dataclasses.replace(
            resumed._ddp,
            world_size=2,
            rank=0,
            local_rank=0,
            is_distributed=False,
        )
        monkeypatch.setattr(
            resumed,
            "_restore_sampler_for_resume",
            lambda *_args, **_kwargs: pytest.fail(
                "callback mutation restored the shared rank-0 sampler"
            ),
        )

        class _DisableSkip(TrainerCallback):
            def on_train_begin(self, args_, state_, control_, **_kwargs):
                args_.ignore_data_skip = False

        resumed.add_callback(_DisableSkip())
        result = resumed.train(resume_from_checkpoint=str(first_out / "checkpoint-2"))
        assert result.global_step == 4

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        saved = checkpoint.load_dp_runtime_state(
            str(resumed_out / "checkpoint-4" / checkpoint.DP_STATE_NAME)
        )
        assert saved.sampler_cursor_origin == 2
        assert saved.sampler_state["consumed"] == 2

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
        assert out1.global_step == max_steps
        expected_params = {
            name: value.detach().clone()
            for name, value in trainer1.model.named_parameters()
        }

        mid_ckpt = outdir / f"checkpoint-{max_steps // 2}"
        assert mid_ckpt.is_dir()

        outdir2 = tmp_path / f"{mechanism}-resumed"
        args2 = _args(
            output_dir=str(outdir2),
            mechanism=mechanism,
            max_steps=max_steps,
            save_steps=4,
        )
        torch.manual_seed(123)
        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=args2,
            train_dataset=ds,
            data_collator=_collate,
        )
        out2 = trainer2.train(resume_from_checkpoint=str(mid_ckpt))
        assert out2.global_step == max_steps

        actual_params = dict(trainer2.model.named_parameters())
        assert list(actual_params) == list(expected_params)
        for name, expected in expected_params.items():
            torch.testing.assert_close(
                actual_params[name].detach(),
                expected,
                rtol=0.0,
                atol=0.0,
                msg=name,
            )

        assert out1.metrics["privacy_epsilon"] == pytest.approx(
            out2.metrics["privacy_epsilon"], rel=1e-3
        )

    def test_resume_rejects_historical_band_poisson_before_accountant_load(
        self,
        tmp_path,
    ):
        outdir = tmp_path / "legacy-band-poisson"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_band",
                max_steps=16,
                save_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        checkpoint_dir = outdir / "checkpoint-4"
        runtime_path = checkpoint_dir / checkpoint.DP_STATE_NAME
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        fixture_path = (
            Path(__file__).parents[1]
            / "fixtures"
            / "issue_776_legacy_sampler_states_v7.json"
        )
        with fixture_path.open() as fixture_file:
            runtime.sampler_state = json.load(fixture_file)["plain_poisson"]
        runtime.__dict__.pop("participation_plan")
        runtime.__dict__.pop("sampler_cursor_origin")
        torch.save(runtime, runtime_path)
        (checkpoint_dir / checkpoint.DP_ACCOUNTANT_NAME).write_text("not-json")

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "legacy-band-poisson-resumed"),
                mechanism="mf_band",
                max_steps=16,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        with pytest.raises(
            CheckpointError,
            match="Changing the sampler cannot repair",
        ):
            trainer2.train(resume_from_checkpoint=str(checkpoint_dir))

    def test_planless_v7_shape_band_b_min_sep_checkpoint_resumes(self, tmp_path):
        outdir = tmp_path / "legacy-band-b-min-sep"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_band",
                max_steps=16,
                save_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        checkpoint_dir = outdir / "checkpoint-4"
        runtime_path = checkpoint_dir / checkpoint.DP_STATE_NAME
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        runtime.__dict__.pop("participation_plan")
        runtime.__dict__.pop("sampler_cursor_origin")
        torch.save(runtime, runtime_path)

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "legacy-band-b-min-sep-resumed"),
                mechanism="mf_band",
                max_steps=16,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        result = trainer2.train(resume_from_checkpoint=str(checkpoint_dir))
        assert result.global_step == 16

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            (lambda state, runtime: state.__setitem__("bands", 5), "saved MF"),
            (
                lambda state, runtime: state.__setitem__("sampling_prob", 0.2),
                "probability contradicts",
            ),
            (lambda state, runtime: state.__setitem__("n_steps", 15), "saved MF"),
            (
                lambda state, runtime: state.__setitem__("consumed", 3),
                "does not match trainer progress",
            ),
        ],
        ids=("bands", "probability", "horizon", "cursor"),
    )
    def test_legacy_band_b_min_sep_mutations_fail_before_training(
        self, tmp_path, monkeypatch, mutation, message
    ):
        outdir = tmp_path / "legacy-band-b-min-sep-invalid"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_band",
                max_steps=16,
                save_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        checkpoint_dir = outdir / "checkpoint-4"
        runtime_path = checkpoint_dir / checkpoint.DP_STATE_NAME
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        runtime.__dict__.pop("participation_plan")
        runtime.__dict__.pop("sampler_cursor_origin")
        mutation(runtime.sampler_state, runtime)
        torch.save(runtime, runtime_path)

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "legacy-band-b-min-sep-invalid-resume"),
                mechanism="mf_band",
                max_steps=16,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        monkeypatch.setattr(
            trainer2,
            "_inner_training_loop",
            lambda *_args, **_kwargs: pytest.fail(
                "invalid legacy sampler reached the training loop"
            ),
        )
        with pytest.raises(CheckpointError, match=message):
            trainer2.train(resume_from_checkpoint=str(checkpoint_dir))

    @pytest.mark.parametrize(
        ("saved_sampling_kwargs", "resumed_sampling_kwargs", "resumed_size"),
        [
            ({}, {}, 32),
            (
                {"truncated_batch_size": 8},
                {"truncated_batch_size": 4},
                64,
            ),
        ],
        ids=("sample-rate", "truncation-cap"),
    )
    def test_gaussian_resume_rejects_restored_sampler_parameter_drift(
        self,
        tmp_path,
        monkeypatch,
        saved_sampling_kwargs,
        resumed_sampling_kwargs,
        resumed_size,
    ):
        outdir = tmp_path / "gaussian-sampler-drift"
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
                sampling_kwargs=saved_sampling_kwargs,
            ),
            train_dataset=_TinyDS(),
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "gaussian-sampler-drift-resume"),
                mechanism="gaussian",
                max_steps=4,
                sampling_kwargs=resumed_sampling_kwargs,
            ),
            train_dataset=_TinyDS(n=resumed_size),
            data_collator=_collate,
        )
        monkeypatch.setattr(
            trainer2,
            "_inner_training_loop",
            lambda *_args, **_kwargs: pytest.fail(
                "drifted sampler reached the training loop"
            ),
        )
        with pytest.raises(CheckpointError, match="participation_plan drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

    def test_gaussian_poisson_resume_allows_horizon_only_extension(self, tmp_path):
        outdir = tmp_path / "gaussian-horizon-extension"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "gaussian-horizon-extension-resumed"),
                mechanism="gaussian",
                max_steps=6,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        result = trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))
        assert result.global_step == 6

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("sample_rate", 0.125),
            ("truncated_batch_size", 2),
        ],
    )
    def test_legacy_gaussian_sampler_is_revalidated_after_restore(
        self, tmp_path, monkeypatch, field, value
    ):
        outdir = tmp_path / "legacy-gaussian-invalid-sampler"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        from opaque.api.transformers.trainer import _checkpoint as checkpoint

        checkpoint_dir = outdir / "checkpoint-2"
        runtime_path = checkpoint_dir / checkpoint.DP_STATE_NAME
        runtime = torch.load(runtime_path, map_location="cpu", weights_only=False)
        runtime.__dict__.pop("participation_plan")
        runtime.__dict__.pop("sampler_cursor_origin")
        runtime.sampler_state[field] = value
        torch.save(runtime, runtime_path)

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "legacy-gaussian-invalid-resume"),
                mechanism="gaussian",
                max_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        monkeypatch.setattr(
            trainer2,
            "training_step",
            lambda *_args, **_kwargs: pytest.fail(
                "invalid restored sampler reached a private training step"
            ),
        )
        with pytest.raises(CheckpointError, match="restored sampler does not match"):
            trainer2.train(resume_from_checkpoint=str(checkpoint_dir))

    def test_k_out_of_t_resume_rejects_parameter_drift(self, tmp_path):
        outdir = tmp_path / "k-out-of-t"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="gaussian",
                max_steps=4,
                save_steps=2,
                sampling_mode="k_out_of_t",
                sampling_kwargs={"k": 2, "allocation": "block"},
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "k-out-of-t-resumed"),
                mechanism="gaussian",
                max_steps=4,
                sampling_mode="k_out_of_t",
                sampling_kwargs={"k": 1, "allocation": "block"},
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="participation_plan"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

    def test_mf_resume_rejects_same_shape_strategy_drift(self, tmp_path):
        outdir = tmp_path / "mf-strategy-drift"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_band",
                max_steps=16,
                save_steps=4,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        resumed_args = _args(
            output_dir=str(tmp_path / "mf-strategy-drift-resumed"),
            mechanism="mf_band",
            max_steps=16,
        )
        resumed_args.privacy_noise_mechanism_kwargs = {
            "bands": 4,
            "momentum": 0.5,
        }
        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=resumed_args,
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="horizon_process_state"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-4"))

    def test_calibrated_horizon_resume_restores_noise_multiplier(self, tmp_path):
        outdir = tmp_path / "calibrated"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=None,
                target_epsilon=5.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        original = trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "calibrated-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=None,
                target_epsilon=5.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        resumed = trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

        assert resumed.metrics["privacy_noise_multiplier"] == pytest.approx(
            original.metrics["privacy_noise_multiplier"]
        )
        assert resumed.metrics["privacy_epsilon"] == pytest.approx(
            original.metrics["privacy_epsilon"]
        )

    def test_calibrated_horizon_resume_rejects_target_drift(self, tmp_path):
        outdir = tmp_path / "calibrated-target-drift"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=None,
                target_epsilon=5.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "calibrated-target-drift-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=None,
                target_epsilon=20.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="privacy_target_epsilon drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

    def test_fixed_horizon_resume_rejects_calibrated_mode(self, tmp_path):
        outdir = tmp_path / "fixed-to-calibrated"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=0.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "fixed-to-calibrated-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=None,
                target_epsilon=1.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="calibration mode drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))

    def test_fixed_horizon_resume_rejects_noise_multiplier_drift(self, tmp_path):
        outdir = tmp_path / "fixed-noise"
        ds = _TinyDS()
        trainer1 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(outdir),
                mechanism="mf_identity",
                max_steps=4,
                save_steps=2,
                noise_multiplier=1.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )
        trainer1.train()

        trainer2 = DPTrainer(
            model=_TinyLM(),
            args=_args(
                output_dir=str(tmp_path / "fixed-noise-resumed"),
                mechanism="mf_identity",
                max_steps=4,
                noise_multiplier=2.0,
            ),
            train_dataset=ds,
            data_collator=_collate,
        )

        with pytest.raises(CheckpointError, match="privacy_noise_multiplier drift"):
            trainer2.train(resume_from_checkpoint=str(outdir / "checkpoint-2"))


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
