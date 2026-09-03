"""Trainer-level wiring for ``torch_compile``, ``use_performance_kernels``, and
compute-precision flags.

These tests target the *plumbing* — compile / kernel features behave
correctly when flags flip — without running full training (which would
require a complete
data collator, sampler, dataset, accountant). The actual training/eval
behavior is covered by the broader trainer suite.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from opaque.api.transformers.trainer._dp_trainer import (
    _compile_with_fullgraph_fallback,
)
from opaque.exceptions import ConfigurationError
from opaque.transformers.trainer import DPTrainer, TrainingArguments

# ----------------------------------------------------------------------------
# Tiny shared trainer helper
# ----------------------------------------------------------------------------


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = {
        "output_dir": str(tmp_path),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "num_train_epochs": 1,
        "save_strategy": "no",
        "use_cpu": True,
        "privacy_target_epsilon": 10.0,
        "privacy_noise_multiplier": 1.0,
    }
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _tiny_trainer(tmp_path, **arg_overrides) -> tuple[DPTrainer, nn.Module]:
    model = nn.Linear(4, 2)
    args = _args(tmp_path, **arg_overrides)
    trainer = DPTrainer(
        model=model,
        args=args,
        train_dataset=[{"x": torch.zeros(4)}],
        eval_dataset=None,
    )
    return trainer, model


# ----------------------------------------------------------------------------
# torch_compile flag wiring
# ----------------------------------------------------------------------------


def test_torch_compile_default_false_does_not_compile(tmp_path):
    """When torch_compile is unset, the trainer must not pull torch.compile
    onto the loss closure (zero-overhead default)."""
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer.args.torch_compile is False


def test_torch_compile_true_accepted(tmp_path):
    """torch_compile=True initializes without error; backend/mode default
    to inductor/default at the closure-building site."""
    trainer, _ = _tiny_trainer(tmp_path, torch_compile=True)
    assert trainer.args.torch_compile is True


def test_torch_compile_with_backend_and_mode(tmp_path):
    trainer, _ = _tiny_trainer(
        tmp_path,
        torch_compile=True,
        torch_compile_backend="aot_eager",
        torch_compile_mode="reduce-overhead",
    )
    assert trainer.args.torch_compile_backend == "aot_eager"
    assert trainer.args.torch_compile_mode == "reduce-overhead"


def test_torch_compile_invalid_mode_rejected_at_args(tmp_path):
    with pytest.raises(ValueError, match="torch_compile_mode"):
        _args(tmp_path, torch_compile_mode="nonsense")


def test_torch_compile_with_auto_find_microbatch_size_rejected(tmp_path):
    with pytest.raises(
        ConfigurationError, match=r"torch_compile.*auto_find_microbatch_size"
    ):
        _args(tmp_path, torch_compile=True, auto_find_microbatch_size=True)


def test_torch_compile_with_explicit_no_autofind_accepted(tmp_path):
    trainer, _ = _tiny_trainer(
        tmp_path,
        torch_compile=True,
        auto_find_microbatch_size=False,
    )
    assert trainer.args.torch_compile is True
    assert trainer.args.auto_find_microbatch_size is False


# ----------------------------------------------------------------------------
# fullgraph fallback control flow (_compile_with_fullgraph_fallback)
# ----------------------------------------------------------------------------


class _FakeCompiled:
    """Fake ``torch.compile`` artifact: records calls, can raise."""

    def __init__(self, fullgraph: bool, error: BaseException | None = None):
        self.fullgraph = fullgraph
        self.error = error
        self.calls = 0

    def __call__(self, *_args, **_kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return "ok"


def _patch_compile(
    monkeypatch, errors: list[BaseException | None] | None = None
) -> list[_FakeCompiled]:
    """Patch ``torch.compile`` to staged fakes raising ``errors[i]``."""
    made: list[_FakeCompiled] = []
    pending = list(errors or [])

    def fake_compile(fn, *, backend, mode, fullgraph):
        stage = _FakeCompiled(
            fullgraph=fullgraph, error=pending.pop(0) if pending else None
        )
        made.append(stage)
        return stage

    monkeypatch.setattr(torch, "compile", fake_compile)
    return made


def _grad_step(x):  # placeholder body; never executed by the fakes
    return x


_DYNAMO_FAILURES = {
    "unsupported": lambda: torch._dynamo.exc.Unsupported("graph break"),
    "backend-compiler-failed": lambda: torch._dynamo.exc.BackendCompilerFailed(
        _grad_step, RuntimeError("inductor failed"), None
    ),
    "dynamo-exception": lambda: torch._dynamo.exc.TorchDynamoException(
        "compile failure"
    ),
}


@pytest.mark.parametrize("failure", list(_DYNAMO_FAILURES), ids=list(_DYNAMO_FAILURES))
def test_fullgraph_fallback_triggers_on_dynamo_failures(monkeypatch, failure):
    """Dynamo failures recompile with ``fullgraph=False`` once, then reuse it."""
    made = _patch_compile(monkeypatch, [_DYNAMO_FAILURES[failure]()])
    compiled = _compile_with_fullgraph_fallback(
        _grad_step, backend="aot_eager", mode="default"
    )

    assert compiled(1) == "ok"
    assert [stage.fullgraph for stage in made] == [True, False]
    assert made[0].calls == 1

    compiled(2)
    compiled(3)
    assert len(made) == 2
    assert made[1].calls == 3


def test_fullgraph_oom_propagates_without_recompile(monkeypatch):
    """OOM is a runtime failure, not a compile failure: it propagates as-is."""
    made = _patch_compile(
        monkeypatch, [torch.OutOfMemoryError("CUDA out of memory (simulated)")]
    )
    compiled = _compile_with_fullgraph_fallback(
        _grad_step, backend="inductor", mode="default"
    )

    with pytest.raises(torch.OutOfMemoryError, match="simulated"):
        compiled(1)

    assert len(made) == 1
    assert made[0].calls == 1


@pytest.mark.parametrize(
    ("error", "exc_type"),
    [
        (RuntimeError("kernel launch failed mid-step"), RuntimeError),
        (ValueError("shape mismatch in the loss closure"), ValueError),
        (ConfigurationError("unsupported trainer configuration"), ConfigurationError),
    ],
    ids=["runtime", "value", "configuration"],
)
def test_non_compile_failure_propagates_without_recompile(monkeypatch, error, exc_type):
    """Non-Dynamo failures propagate without fallback compilation or retry."""
    made = _patch_compile(monkeypatch, [error])
    compiled = _compile_with_fullgraph_fallback(
        _grad_step, backend="inductor", mode="default"
    )

    with pytest.raises(exc_type, match=str(error)):
        compiled(1)

    assert len(made) == 1
    assert made[0].calls == 1


def test_oom_during_fallback_recompile_propagates(monkeypatch):
    """OOM during the ``fullgraph=False`` re-run propagates, chaining the
    original Dynamo error as ``__context__``."""
    made = _patch_compile(
        monkeypatch,
        [
            torch._dynamo.exc.Unsupported("graph break"),
            torch.OutOfMemoryError("CUDA out of memory (simulated)"),
        ],
    )
    compiled = _compile_with_fullgraph_fallback(
        _grad_step, backend="inductor", mode="default"
    )

    with pytest.raises(torch.OutOfMemoryError, match="simulated") as excinfo:
        compiled(1)

    assert [stage.fullgraph for stage in made] == [True, False]
    assert isinstance(excinfo.value.__context__, torch._dynamo.exc.Unsupported)


# ----------------------------------------------------------------------------
# use_performance_kernels — wiring through to apply_model_patches
# ----------------------------------------------------------------------------


def test_use_performance_kernels_default_keeps_kv_cache_and_compat_on(
    tmp_path, monkeypatch
):
    """Default-off ``use_performance_kernels`` still applies compat and the
    ``performance`` bucket (kv_cache); only the Triton ``kernels`` group is
    disabled."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path)  # use_performance_kernels default is False
    assert len(calls) == 1
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is False
    assert calls[0]["kwargs"]["compat"] is True


def test_use_performance_kernels_true_enables_kernels_group(tmp_path, monkeypatch):
    """``use_performance_kernels=True`` flips ``kernels`` on at the
    ``apply_model_patches`` call (alongside the always-on ``performance``
    and ``compat`` umbrellas)."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"model": model, "kwargs": kwargs})

    # _performance_kernels.py imports apply_model_patches lazily inside the function body,
    # so patch the source location rather than the consumer's module.
    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _trainer, model = _tiny_trainer(tmp_path, use_performance_kernels=True)
    assert len(calls) == 1
    assert calls[0]["model"] is model
    assert calls[0]["kwargs"]["performance"] is True
    assert calls[0]["kwargs"]["kernels"] is True
    assert calls[0]["kwargs"]["compat"] is True


def test_performance_kernels_config_forwards_opaque_keys_as_is(tmp_path, monkeypatch):
    """``performance_kernels_config`` is a flat dict forwarded as-is to
    ``apply_model_patches`` kwargs — no key translation, opaque-patches
    keys used directly."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(
        tmp_path,
        use_performance_kernels=True,
        performance_kernels_config={
            "rope": True,
            "rms_norm": True,
            "fused_linear_cross_entropy": True,
        },
    )
    assert len(calls) == 1
    assert calls[0]["kwargs"]["rope"] is True
    assert calls[0]["kwargs"]["rms_norm"] is True
    assert calls[0]["kwargs"]["fused_linear_cross_entropy"] is True


def test_performance_kernels_config_can_disable_kv_cache(tmp_path, monkeypatch):
    """``kv_cache`` stays on by default but can be opted out via the config
    dict for models whose forward depends on HF's DynamicCache."""
    calls: list[dict] = []

    def _spy(model, **kwargs):
        calls.append({"kwargs": kwargs})

    monkeypatch.setattr("opaque.patches.apply_model_patches", _spy)

    _tiny_trainer(tmp_path, performance_kernels_config={"kv_cache": False})
    assert len(calls) == 1
    assert calls[0]["kwargs"]["kv_cache"] is False
    assert calls[0]["kwargs"]["performance"] is True


# ----------------------------------------------------------------------------
# Precision: bf16 is the only mixed-precision mode; fp16 is unsupported
# ----------------------------------------------------------------------------


def test_bf16_trainer_enables_autocast(tmp_path):
    """bf16=True sets the autocast dtype (no loss scaler — bf16's wider
    exponent range needs none)."""
    trainer, _ = _tiny_trainer(tmp_path, bf16=True)
    assert trainer._amp_dtype == torch.bfloat16


def test_fp32_trainer_has_no_autocast(tmp_path):
    trainer, _ = _tiny_trainer(tmp_path)
    assert trainer._amp_dtype is None


def test_fp16_training_is_rejected(tmp_path):
    """fp16 training (autocast + dynamic loss scaling) is unsupported."""
    with pytest.raises(TypeError):
        _tiny_trainer(tmp_path, fp16=True)
