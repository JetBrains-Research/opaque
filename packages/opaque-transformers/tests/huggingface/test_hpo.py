"""Hyperparameter-search contract tests for DPTrainer."""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch
from transformers.trainer_callback import TrainerCallback

from opaque.transformers.trainer import DPTrainer, DPTrainingArguments


class _LossModel(torch.nn.Module):
    main_input_name = "x"

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x, labels=None):
        logits = self.linear(x)
        loss = None
        if labels is not None:
            if logits.ndim == 1:
                logits_for_loss = logits.unsqueeze(0)
                labels_for_loss = labels.reshape(1)
            else:
                logits_for_loss = logits
                labels_for_loss = labels
            loss = torch.nn.functional.cross_entropy(logits_for_loss, labels_for_loss)
        return {"loss": loss, "logits": logits}


def _dataset() -> list[dict[str, torch.Tensor]]:
    return [
        {"x": torch.tensor([1.0, 0.0, 0.0, 0.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([0.0, 1.0, 0.0, 0.0]), "labels": torch.tensor(1)},
        {"x": torch.tensor([0.0, 0.0, 1.0, 0.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([0.0, 0.0, 0.0, 1.0]), "labels": torch.tensor(1)},
        {"x": torch.tensor([1.0, 1.0, 0.0, 0.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([0.0, 1.0, 1.0, 0.0]), "labels": torch.tensor(1)},
        {"x": torch.tensor([0.0, 0.0, 1.0, 1.0]), "labels": torch.tensor(0)},
        {"x": torch.tensor([1.0, 0.0, 0.0, 1.0]), "labels": torch.tensor(1)},
    ]


def _args(tmp_path, **overrides) -> DPTrainingArguments:
    defaults = dict(
        output_dir=str(tmp_path),
        use_cpu=True,
        per_device_train_batch_size=7,
        per_device_eval_batch_size=7,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_train_epochs=1,
        learning_rate=1e-3,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1,
        dp_target_epsilon=10.0,
        dp_noise_multiplier=1.0,
        remove_unused_columns=True,
    )
    defaults.update(overrides)
    return DPTrainingArguments(**defaults)


def test_train_trial_dict_reinitializes_model_and_isolates_output_dir(tmp_path):
    seen_trials = []

    def model_init(trial=None):
        seen_trials.append(trial)
        return _LossModel()

    trainer = DPTrainer(
        model_init=model_init,
        args=_args(tmp_path),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    output = trainer.train(trial={"run_id": "manual", "learning_rate": 0.02})

    assert output.global_step == 1
    assert seen_trials[-1] == {"run_id": "manual", "learning_rate": 0.02}
    assert trainer.args.learning_rate == pytest.approx(0.02)
    assert os.path.exists(tmp_path / "run-manual" / "checkpoint-1" / "accountant.json")
    assert not os.path.exists(tmp_path / "checkpoint-1")


def test_train_trial_requires_model_init(tmp_path):
    trainer = DPTrainer(
        model=_LossModel(),
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    with pytest.raises(RuntimeError, match="model_init"):
        trainer.train(trial={"learning_rate": 0.02})


def test_hyperparameter_search_rejects_unsupported_backend(tmp_path):
    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    with pytest.raises(ValueError, match="Ray Tune owns process/actor execution"):
        trainer.hyperparameter_search(backend="ray", n_trials=1)


def test_dict_trials_rebuild_callbacks_with_fresh_state(tmp_path):
    records = []

    class RecordingCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            records.append(
                {
                    "state_id": id(state),
                    "handler_state_id": id(trainer.callback_handler.state),
                    "trial_name": state.trial_name,
                    "trial_params": dict(state.trial_params or {}),
                }
            )

    def model_init(trial=None):
        return _LossModel()

    trainer = DPTrainer(
        model_init=model_init,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
        callbacks=[RecordingCallback()],
    )

    trainer.train(trial={"run_id": "first", "learning_rate": 0.01})
    trainer.train(trial={"run_id": "second", "learning_rate": 0.02})

    assert [record["trial_name"] for record in records] == [
        "run-first",
        "run-second",
    ]
    assert [record["trial_params"]["learning_rate"] for record in records] == [
        0.01,
        0.02,
    ]
    assert records[0]["state_id"] != records[1]["state_id"]
    assert records[0]["state_id"] == records[0]["handler_state_id"]
    assert records[1]["state_id"] == records[1]["handler_state_id"]


def test_hyperparameter_search_wandb_uses_sweep_agent_and_isolates_trials(
    tmp_path, monkeypatch
):
    seen_trials = []

    class _FakeConfig:
        def __init__(self, items):
            self._items = dict(items)

        def update(self, values):
            self._items.update(values)

    class _FakeRun:
        def __init__(self, run_id):
            self.id = run_id
            self.name = f"name-{run_id}"
            self.config = fake_wandb.config

    fake_wandb = types.ModuleType("wandb")
    fake_wandb.run = None
    fake_wandb.config = _FakeConfig({})
    fake_wandb.sweep_calls = []
    fake_wandb.agent_calls = []
    trial_values = [0.01, 0.02]
    fake_wandb._trial_index = 0

    def sweep(config, project=None, entity=None):
        fake_wandb.sweep_calls.append((config, project, entity))
        return "sweep-123"

    def init():
        run_id = f"wandb-{fake_wandb._trial_index}"
        fake_wandb.config = _FakeConfig(
            {"learning_rate": trial_values[fake_wandb._trial_index]}
        )
        fake_wandb.run = _FakeRun(run_id)
        return fake_wandb.run

    def agent(sweep_id, function, count):
        fake_wandb.agent_calls.append((sweep_id, count))
        for index in range(count):
            fake_wandb._trial_index = index
            fake_wandb.run = None
            function()

    fake_wandb.sweep = sweep
    fake_wandb.init = init
    fake_wandb.agent = agent
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    def model_init(trial=None):
        seen_trials.append(trial)
        return _LossModel()

    trainer = DPTrainer(
        model_init=model_init,
        args=_args(
            tmp_path,
            eval_strategy="steps",
            eval_steps=1,
            metric_for_best_model="eval_loss",
        ),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    def hp_space(_trial):
        return {
            "method": "grid",
            "metric": {"name": "objective"},
            "parameters": {"learning_rate": {"values": trial_values}},
        }

    best = trainer.hyperparameter_search(
        backend="wandb",
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_loss"],
        n_trials=2,
        direction="minimize",
        project="opaque-test",
        entity="federated-compute",
        metric="eval/loss",
    )

    sweep_config, project, entity = fake_wandb.sweep_calls[0]
    assert (project, entity) == ("opaque-test", "federated-compute")
    assert sweep_config["metric"] == {"name": "eval/loss", "goal": "minimize"}
    assert fake_wandb.agent_calls == [("sweep-123", 2)]
    assert best.run_id in {"wandb-0", "wandb-1"}
    assert "learning_rate" in best.hyperparameters
    assert len(seen_trials) == 3  # constructor model_init + two W&B trials
    assert seen_trials[1]["run_id"] == "wandb-0"
    assert seen_trials[2]["run_id"] == "wandb-1"
    assert os.path.exists(tmp_path / "run-wandb-0" / "checkpoint-1" / "accountant.json")
    assert os.path.exists(tmp_path / "run-wandb-1" / "checkpoint-1" / "accountant.json")


def test_hyperparameter_search_optuna_returns_best_run_and_isolates_trials(tmp_path):
    pytest.importorskip("optuna")
    seen_trials = []

    def model_init(trial=None):
        seen_trials.append(trial)
        return _LossModel()

    trainer = DPTrainer(
        model_init=model_init,
        args=_args(
            tmp_path,
            eval_strategy="steps",
            eval_steps=1,
            metric_for_best_model="eval_loss",
        ),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.02])
        }

    best = trainer.hyperparameter_search(
        hp_space=hp_space,
        compute_objective=lambda metrics: metrics["eval_loss"],
        n_trials=2,
        direction="minimize",
        sampler=__import__("optuna").samplers.GridSampler(
            {"learning_rate": [0.01, 0.02]}
        ),
    )

    assert best.run_id in {"0", "1"}
    assert "learning_rate" in best.hyperparameters
    assert len(seen_trials) == 3  # constructor model_init + two HPO trials
    assert os.path.exists(tmp_path / "run-0" / "checkpoint-1" / "accountant.json")
    assert os.path.exists(tmp_path / "run-1" / "checkpoint-1" / "accountant.json")
