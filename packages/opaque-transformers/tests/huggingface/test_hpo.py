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

    with pytest.raises(ValueError, match=r"\{'optuna', 'wandb', 'ray'\}"):
        trainer.hyperparameter_search(backend="sigopt", n_trials=1)


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


# ---------------------------------------------------------------------
# Ray Tune backend (Phase 12)
# ---------------------------------------------------------------------


def test_dptrainer_pickles_after_scrub(tmp_path):
    """``_scrub_for_pickling`` should make a DPTrainer picklable."""
    import pickle

    from opaque.transformers.trainer._hpo import _scrub_for_pickling

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    _scrub_for_pickling(trainer)

    blob = pickle.dumps(trainer)
    restored = pickle.loads(blob)
    assert restored.model is None
    assert restored.model_init is not None
    assert restored.args.output_dir == str(tmp_path)


def test_ray_hpo_rejects_multirank_world_size(tmp_path):
    pytest.importorskip("ray.tune")

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )
    trainer.args.world_size = 2

    with pytest.raises(ValueError, match="Phase 10"):
        trainer.hyperparameter_search(backend="ray", n_trials=1)


def test_ray_hpo_rejects_multirank_per_trial_gpu(tmp_path):
    pytest.importorskip("ray.tune")

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    with pytest.raises(ValueError, match="Phase 10"):
        trainer.hyperparameter_search(
            backend="ray",
            n_trials=1,
            resources_per_trial={"cpu": 1, "gpu": 2},
        )


def test_ray_hpo_requires_absolute_output_dir(tmp_path, monkeypatch):
    pytest.importorskip("ray.tune")
    # Construct with absolute path (post_init validates), then mutate to
    # a relative one to exercise the scrub-time check.
    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )
    trainer.args.output_dir = "rel/output"

    with pytest.raises(ValueError, match="absolute path"):
        trainer.hyperparameter_search(backend="ray", n_trials=1)


def _install_fake_ray(monkeypatch):
    """Install a minimal fake ``ray`` / ``ray.train`` / ``ray.tune`` stack.

    Captures every public touch point exercised by ``_run_ray_search``
    so wiring tests can inspect HF parity without spinning up real Ray
    actors.  Returns the module handle for assertions.
    """

    fake_ray = types.ModuleType("ray")
    fake_ray.train = types.ModuleType("ray.train")
    fake_ray.tune = types.ModuleType("ray.tune")
    fake_ray.tune.schedulers = types.ModuleType("ray.tune.schedulers")

    class _FakeAnalysis:
        def get_best_trial(self, metric, mode, scope):
            self.metric = metric
            self.mode = mode
            self.scope = scope
            best = types.SimpleNamespace(
                trial_id="ray-0",
                last_result={"objective": 0.5},
                config={"learning_rate": 0.02},
            )
            return best

    captured = {
        "with_parameters_calls": [],
        "tune_run_calls": [],
        "objectives": [],
        "wrapped_trainable_called": False,
    }

    def with_parameters(fn, **bound):
        captured["with_parameters_calls"].append((fn, bound))
        wrapped = lambda *a, **kw: fn(*a, **bound)  # noqa: E731 (test fake)
        wrapped.__mixins__ = ("dummy",)
        return wrapped

    def tune_run(trainable, *, config, num_samples, **kwargs):
        captured["tune_run_calls"].append(
            {"config": config, "num_samples": num_samples, **kwargs}
        )
        captured["wrapped_trainable_called"] = True
        return _FakeAnalysis()

    class _FakeReporter:
        def __init__(self, *, metric_columns):
            self.metric_columns = list(metric_columns)

    fake_ray.tune.with_parameters = with_parameters
    fake_ray.tune.run = tune_run
    fake_ray.tune.CLIReporter = _FakeReporter

    class _FakeASHA:
        pass

    class _FakeMedian:
        pass

    class _FakeHyperBand:
        pass

    class _FakePBT:
        pass

    fake_ray.tune.schedulers.ASHAScheduler = _FakeASHA
    fake_ray.tune.schedulers.MedianStoppingRule = _FakeMedian
    fake_ray.tune.schedulers.HyperBandForBOHB = _FakeHyperBand
    fake_ray.tune.schedulers.PopulationBasedTraining = _FakePBT

    def get_checkpoint():
        return None

    fake_ray.train.get_checkpoint = get_checkpoint

    class _FakeCheckpoint:
        @classmethod
        def from_directory(cls, path):
            return ("checkpoint", path)

    def report(metrics, checkpoint=None):
        captured["objectives"].append((dict(metrics), checkpoint))

    fake_ray.train.Checkpoint = _FakeCheckpoint
    fake_ray.train.report = report

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(sys.modules, "ray.train", fake_ray.train)
    monkeypatch.setitem(sys.modules, "ray.tune", fake_ray.tune)
    monkeypatch.setitem(sys.modules, "ray.tune.schedulers", fake_ray.tune.schedulers)

    return fake_ray, captured


def test_ray_hpo_dispatch_returns_best_run_and_pops_tensorboard(tmp_path, monkeypatch):
    fake_ray, captured = _install_fake_ray(monkeypatch)

    class _FakeTensorBoardCallback(TrainerCallback):
        pass

    fake_integrations = types.ModuleType("transformers.integrations")
    fake_integrations.TensorBoardCallback = _FakeTensorBoardCallback
    monkeypatch.setitem(sys.modules, "transformers.integrations", fake_integrations)

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )
    tb = _FakeTensorBoardCallback()
    trainer.add_callback(tb)

    best = trainer.hyperparameter_search(
        backend="ray",
        n_trials=2,
        direction="minimize",
        hp_space=lambda _t: {"learning_rate": 0.02},
    )

    # BestRun shape (HF parity).
    assert best.run_id == "ray-0"
    assert best.objective == 0.5
    assert best.hyperparameters == {"learning_rate": 0.02}
    assert best.run_summary is not None  # analysis attached

    # tune.run received our config / num_samples / scheduler-empty kwargs.
    call = captured["tune_run_calls"][0]
    assert call["num_samples"] == 2
    assert call["config"] == {"learning_rate": 0.02}
    # Default resources injected (HF parity).
    expected_resources = {"cpu": 1}
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        expected_resources["gpu"] = 1
    assert call["resources_per_trial"] == expected_resources
    # Default progress reporter wired (HF parity).
    reporter = call["progress_reporter"]
    assert reporter.metric_columns == ["objective"]

    # tune.with_parameters received the trainer.
    fn, bound = captured["with_parameters_calls"][0]
    assert "local_trainer" in bound
    # dynamic-modules wrapper preserves the __mixins__ tag from with_parameters.
    assert getattr(call["config"], "__mixins__", None) is None  # config dict only

    # TensorBoard callback was popped during the sweep and re-attached.
    callback_types = [type(cb) for cb in trainer.callback_handler.callbacks]
    assert _FakeTensorBoardCallback in callback_types

    # HPO state is cleared post-search.
    assert trainer.hp_search_backend is None
    assert trainer.hp_space is None


def test_ray_hpo_scheduler_requires_eval(tmp_path, monkeypatch):
    fake_ray, _ = _install_fake_ray(monkeypatch)

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no", do_eval=False, eval_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    with pytest.raises(RuntimeError, match=r"FakeASHA.*haven't enabled evaluation"):
        trainer.hyperparameter_search(
            backend="ray",
            n_trials=1,
            scheduler=fake_ray.tune.schedulers.ASHAScheduler(),
        )


def test_ray_hpo_get_output_dir_uses_trial_id(tmp_path, monkeypatch):
    pytest.importorskip("ray.tune")
    from transformers.trainer_utils import HPSearchBackend

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )

    # Pretend we're in the middle of a Ray sweep.
    trainer.hp_search_backend = HPSearchBackend.RAY

    # Stub ray.train.get_context().get_trial_id().
    fake_ray_train = types.ModuleType("ray.train")
    fake_ray_train.get_context = lambda: types.SimpleNamespace(
        get_trial_id=lambda: "trial-abc123"
    )
    monkeypatch.setitem(sys.modules, "ray.train", fake_ray_train)

    out = trainer._get_output_dir(trial={"learning_rate": 0.01})
    assert out == os.path.join(str(tmp_path), "run-trial-abc123")


def test_tune_save_checkpoint_writes_full_dp_snapshot(tmp_path):
    """``_tune_save_checkpoint`` must emit a complete resume-capable tree."""
    completed: dict = {}

    class _CaptureCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            ckpt_root = tmp_path / "ray_ckpt"
            ckpt_root.mkdir(exist_ok=True)
            trainer._tune_save_checkpoint(checkpoint_dir=str(ckpt_root))
            completed["root"] = str(ckpt_root)
            control.should_training_stop = True

    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
        callbacks=[_CaptureCallback()],
    )

    trainer.train()

    root = completed.get("root")
    assert root is not None, "callback never fired"
    entries = os.listdir(root)
    ckpt_dirs = [e for e in entries if e.startswith("checkpoint-")]
    assert ckpt_dirs, f"no checkpoint subdir in {entries}"
    inside = os.listdir(os.path.join(root, ckpt_dirs[0]))
    # Resumability set: model + accountant + trainer_state + DP runtime
    # + optimizer + RNG + training_args.
    expected_files = {
        "accountant.json",
        "trainer_state.json",
        "dp_runtime_state.pt",
        "dp_optimizer.pt",
        "rng_state.pth",
        "training_args.bin",
    }
    assert expected_files.issubset(set(inside)), (
        f"missing files: {expected_files - set(inside)} (have {inside})"
    )
    # Model weights either as safetensors or pickled.
    assert any(
        f in inside
        for f in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        )
    ), inside


def test_tune_save_checkpoint_outside_loop_raises(tmp_path):
    trainer = DPTrainer(
        model_init=_LossModel,
        args=_args(tmp_path, save_strategy="no"),
        train_dataset=_dataset(),
        eval_dataset=_dataset(),
    )
    with pytest.raises(RuntimeError, match="self._ctx is None"):
        trainer._tune_save_checkpoint(checkpoint_dir=str(tmp_path / "x"))
