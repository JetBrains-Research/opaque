"""Real Ray Tune smoke test for ``DPTrainer.hyperparameter_search``.

This module lives **outside** ``tests/opaque_transformers/`` on purpose: the
trainer integration tests share a conftest that applies runtime patches at
session scope; keeping this Ray Tune smoke at the package tests root yields a
stable top-level module path (``test_dptrainer_ray_hpo_smoke``) for cloudpickle
when Ray workers unpickle callables.
"""

from __future__ import annotations

import os

import pytest
import torch

from opaque.transformers.trainer import DPTrainer, TrainingArguments


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


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = dict(
        output_dir=str(tmp_path),
        use_cpu=True,
        per_device_train_batch_size=7,
        per_device_eval_batch_size=7,
        max_steps=1,
        num_train_epochs=1,
        learning_rate=1e-3,
        eval_strategy="no",
        save_strategy="no",
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
        remove_unused_columns=True,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


@pytest.mark.slow
def test_ray_tune_real_smoke_one_trial(tmp_path):
    """One CPU Ray trial (cold worker ``uv`` env can take tens of seconds)."""
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    pytest.importorskip("ray.tune")
    import ray

    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=2, include_dashboard=False, logging_level="error")
    try:
        trainer = DPTrainer(
            model_init=lambda trial=None: _LossModel(),
            args=_args(tmp_path),
            train_dataset=_dataset(),
            eval_dataset=_dataset(),
        )
        best = trainer.hyperparameter_search(
            backend="ray",
            n_trials=1,
            direction="minimize",
            hp_space=lambda _t: {"learning_rate": 0.01},
            resources_per_trial={"cpu": 1},
        )
        assert best.run_id
        assert best.objective is not None
    finally:
        if ray.is_initialized():
            ray.shutdown()
