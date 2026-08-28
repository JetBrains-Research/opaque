"""Regression: _restore_params must not re-derive trainability from the live
module's requires_grad flags.

The live module's requires_grad flags are not a reliable record of what was
trained under the functional training path. The old guard re-derived the
trainable set from the live module and could raise on a completing trainer run;
the guard now validates the provided keys against the model's state_dict instead.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.exceptions import ConfigurationError
from opaque.transformers.trainer import DPTrainer, TrainingArguments


def _trainer(tmp_path):
    model = nn.Linear(4, 2)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        max_steps=1,
        save_strategy="no",
        use_cpu=True,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
    )
    return DPTrainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )


def test_restore_params_succeeds_when_module_requires_grad_neutralized(tmp_path):
    trainer = _trainer(tmp_path)
    # Simulate the post-functional state: every param requires_grad=False.
    for p in trainer._model.parameters():
        p.requires_grad_(False)

    trained = {
        name: torch.ones_like(p) for name, p in trainer._model.named_parameters()
    }
    # Must not raise (old guard raised here).
    trainer._restore_params(trained)

    for _, p in trainer._model.named_parameters():
        assert torch.equal(p, torch.ones_like(p))


def test_restore_params_rejects_foreign_keys(tmp_path):
    trainer = _trainer(tmp_path)
    bad = {"not_a_real_param": torch.zeros(1)}
    with pytest.raises(ConfigurationError, match="not present in the model"):
        trainer._restore_params(bad)
