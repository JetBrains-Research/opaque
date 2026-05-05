"""HF Trainer contract regressions for DPTrainer."""

from __future__ import annotations

import os
import inspect
import types

import torch
from transformers.trainer_callback import DefaultFlowCallback, TrainerCallback

import opaque.transformers.trainer as trainer_module
import opaque.transformers.trainer._callback as callback_module
from opaque.transformers.trainer import DPTrainer, DPTrainingArguments


class _LogitsOnlyModel(torch.nn.Module):
    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        return {"logits": self.linear(x)}


class _StopOnInitCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        control.should_training_stop = True
        return control


class _ReportingCallback(TrainerCallback):
    pass


class _UserCallback(TrainerCallback):
    pass


def _args(tmp_path, **overrides) -> DPTrainingArguments:
    defaults = dict(
        output_dir=str(tmp_path),
        save_strategy="no",
        use_cpu=True,
        dp_target_epsilon=10.0,
        dp_noise_multiplier=1.0,
    )
    defaults.update(overrides)
    return DPTrainingArguments(**defaults)


def test_constructor_accepts_hf_positional_model_and_optional_datasets(tmp_path):
    model = _LogitsOnlyModel()
    args = _args(tmp_path)

    trainer = DPTrainer(model, args)

    assert trainer.model is model
    assert trainer.train_dataset is None
    assert trainer.eval_dataset is None


def test_constructor_accepts_model_init(tmp_path):
    args = _args(tmp_path)

    trainer = DPTrainer(model_init=_LogitsOnlyModel, args=args)

    assert isinstance(trainer.model, _LogitsOnlyModel)


def test_label_less_predict_returns_logits_not_loss(tmp_path):
    args = _args(tmp_path, per_device_eval_batch_size=2)
    trainer = DPTrainer(model=_LogitsOnlyModel(), args=args)
    dataset = [{"x": torch.zeros(4)}, {"x": torch.ones(4)}]

    output = trainer.predict(dataset)

    assert output.predictions.shape == (2, 2)
    assert output.label_ids is None
    assert "test_loss" not in output.metrics


def test_callback_returned_control_is_preserved(tmp_path):
    trainer = DPTrainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path),
        callbacks=[_StopOnInitCallback()],
    )

    assert trainer.control.should_training_stop is True


def test_reporting_callbacks_precede_user_callbacks_and_see_functional_slots(
    tmp_path, monkeypatch
):
    def fake_reporting_callbacks(report_to):
        assert report_to == ["tensorboard"]
        return [_ReportingCallback]

    monkeypatch.setattr(
        callback_module,
        "get_reporting_integration_callbacks",
        fake_reporting_callbacks,
    )

    trainer = DPTrainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path, report_to="tensorboard"),
        callbacks=[_UserCallback()],
    )

    callbacks = trainer.callback_handler.callbacks
    default_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, DefaultFlowCallback)
    )
    reporting_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, _ReportingCallback)
    )
    user_index = next(
        index
        for index, callback in enumerate(callbacks)
        if isinstance(callback, _UserCallback)
    )

    assert default_index < reporting_index < user_index
    assert trainer.callback_handler.optimizer is None
    assert trainer.callback_handler.lr_scheduler is None


def test_full_determinism_uses_hf_deterministic_seed_helper(tmp_path, monkeypatch):
    calls = []

    def fake_enable_full_determinism(seed):
        calls.append(("full", seed))

    def fake_set_seed(seed):
        calls.append(("seed", seed))

    monkeypatch.setattr(
        trainer_module,
        "enable_full_determinism",
        fake_enable_full_determinism,
    )
    monkeypatch.setattr(trainer_module, "set_seed", fake_set_seed)

    DPTrainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path, full_determinism=True, seed=123),
    )

    assert calls == [("full", 123)]


def test_label_smoothing_recomputes_loss_from_logits_for_vector_case(tmp_path):
    trainer = DPTrainer(
        model=_LogitsOnlyModel(),
        args=_args(tmp_path, label_smoothing_factor=0.2),
    )
    logits = torch.tensor([2.0, -1.0], requires_grad=True)
    labels = torch.tensor(0)
    unsmoothed_loss = torch.nn.functional.cross_entropy(
        logits.reshape(1, -1),
        labels.reshape(1),
    )

    def fmodel(params, **kwargs):
        return types.SimpleNamespace(loss=unsmoothed_loss, logits=logits)

    loss_fn, batch_argnums = trainer._build_per_example_loss(
        fmodel,
        frozen_params={},
        batch_keys=("x", "labels"),
    )

    actual = loss_fn({}, torch.zeros(4), labels)
    expected = torch.nn.functional.cross_entropy(
        logits.reshape(1, -1),
        labels.reshape(1),
        label_smoothing=0.2,
    )

    assert batch_argnums == (1, 2)
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, unsmoothed_loss)


def test_public_save_model_writes_training_args(tmp_path):
    trainer = DPTrainer(model=_LogitsOnlyModel(), args=_args(tmp_path))

    trainer.save_model()

    assert os.path.exists(tmp_path / "training_args.bin")


def test_hpo_public_method_signatures_match_hf_shape():
    from transformers import Trainer

    trainer_train = inspect.signature(Trainer.train)
    dp_train = inspect.signature(DPTrainer.train)
    for name in ["resume_from_checkpoint", "trial", "ignore_keys_for_eval"]:
        assert name in dp_train.parameters
        assert name in trainer_train.parameters
    assert any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in dp_train.parameters.values()
    )

    trainer_hpo = inspect.signature(Trainer.hyperparameter_search)
    dp_hpo = inspect.signature(DPTrainer.hyperparameter_search)
    for name in [
        "hp_space",
        "compute_objective",
        "n_trials",
        "direction",
        "backend",
        "hp_name",
    ]:
        assert name in dp_hpo.parameters
        assert name in trainer_hpo.parameters


def test_process_helpers_are_single_process_true(tmp_path):
    trainer = DPTrainer(model=_LogitsOnlyModel(), args=_args(tmp_path))

    assert trainer.is_world_process_zero()
    assert trainer.is_local_process_zero()
    assert trainer._is_world_process_zero()
    assert trainer._is_local_process_zero()
