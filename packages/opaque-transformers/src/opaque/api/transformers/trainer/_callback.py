"""Callback handler construction for DPTrainer.

DPTrainer uses HF's :class:`transformers.trainer_callback.CallbackHandler`
and :class:`~transformers.trainer_callback.DefaultFlowCallback` directly.
This module just bundles the construction pattern HF's ``Trainer``
applies internally:

1. Prepend ``DEFAULT_CALLBACKS = [DefaultFlowCallback]``.
2. Append reporting integration callbacks from ``args.report_to``
   (``WandbCallback``, ``TensorBoardCallback``, etc.) — HF's
   :func:`transformers.integrations.get_reporting_integration_callbacks`
   returns the right set for each backend string.
3. Append user-supplied callbacks.
4. Construct the handler.
5. Register a progress callback (``ProgressCallback`` / ``PrinterCallback``)
   based on ``args.disable_tqdm``.

The handler stores ``optimizer=None`` and ``lr_scheduler=None`` because the
functional optimizer + ``Callable[[int], float]`` schedule have no
``torch.optim`` instance HF callbacks can read; callbacks that introspect
those slots see ``None``.

DP semantic note: ``on_substep_end`` is part of HF's hook surface but is
**not fired** by DPTrainer.  Each iteration of DP-SGD is one atomic
optimizer step (the Poisson round); there is no substep concept.  HF
callbacks that override ``on_substep_end`` will simply not be invoked.
"""

from __future__ import annotations

import logging
from typing import Any

from transformers import TrainingArguments
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
)


__all__ = [
    "DEFAULT_CALLBACKS",
    "build_callback_handler",
    "rewrite_logs_for_reporting",
    "wrap_reporting_callback_class",
]


log = logging.getLogger(__name__)


DEFAULT_CALLBACKS: list[type[TrainerCallback]] = [DefaultFlowCallback]


_GROUP_PRIVACY_SUFFIXES: tuple[tuple[str, str], ...] = (
    ("_clipping_norm", "clipping_norm"),
    ("_grad_norm", "grad_norm"),
    ("_clip_rate", "clip_rate"),
    ("_noise_std", "noise_std"),
)
_PRIVACY_SUMMARY_KEYS = {
    "privacy_epsilon",
    "privacy_delta",
    "privacy_noise_multiplier",
    "privacy_calibration_converged",
    "privacy_calibration_achieved_epsilon",
}
_WRAPPED_CALLBACK_CLASSES: dict[type[Any], type[Any]] = {}


def _rewrite_privacy_group_key(key: str) -> str | None:
    group_and_metric = key.removeprefix("privacy_group_")
    for suffix, metric_name in _GROUP_PRIVACY_SUFFIXES:
        if group_and_metric.endswith(suffix):
            group_name = group_and_metric[: -len(suffix)]
            if group_name:
                return f"privacy/group_{group_name}/{metric_name}"
    return None


def _rewrite_privacy_key(key: str) -> str | None:
    if key.startswith("privacy_group_"):
        group_key = _rewrite_privacy_group_key(key)
        if group_key is not None:
            return group_key
    if key.startswith("privacy_"):
        return "privacy/" + key.removeprefix("privacy_")
    return None


def rewrite_logs_for_reporting(logs: dict[str, Any] | None) -> dict[str, Any]:
    """Rewrite HF raw metric keys for hierarchy-aware reporting backends."""
    rewritten: dict[str, Any] = {}
    for key, value in (logs or {}).items():
        privacy_key = _rewrite_privacy_key(key)
        if privacy_key is not None:
            rewritten[privacy_key] = value
        elif key.startswith("eval_"):
            rewritten["eval/" + key.removeprefix("eval_")] = value
        elif key.startswith("test_"):
            rewritten["test/" + key.removeprefix("test_")] = value
        else:
            rewritten["train/" + key] = value
    return rewritten


def wrap_reporting_callback_class(callback_cls: type[Any]) -> type[Any]:
    """Wrap HF reporting callbacks whose log rewrite needs privacy keys."""
    if callback_cls in _WRAPPED_CALLBACK_CLASSES:
        return _WRAPPED_CALLBACK_CLASSES[callback_cls]

    class_name = getattr(callback_cls, "__name__", "")
    if class_name == "WandbCallback":

        class OpaqueWandbCallback(callback_cls):  # type: ignore[misc, valid-type]
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                single_value_scalars = {
                    "train_runtime",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "train_loss",
                    "total_flos",
                }

                if self._wandb is None:
                    return
                if not self._initialized:
                    self.setup(args, state, model)
                if state.is_world_process_zero:
                    raw_logs = dict(logs or {})
                    for key, value in raw_logs.items():
                        if key in single_value_scalars:
                            self._wandb.run.summary[key] = value
                        privacy_key = _rewrite_privacy_key(key)
                        if key in _PRIVACY_SUMMARY_KEYS and privacy_key is not None:
                            self._wandb.run.summary[privacy_key] = value
                    non_scalar_logs = {
                        key: value
                        for key, value in raw_logs.items()
                        if key not in single_value_scalars
                    }
                    self._wandb.log(
                        {
                            **rewrite_logs_for_reporting(non_scalar_logs),
                            "train/global_step": state.global_step,
                        }
                    )

        OpaqueWandbCallback.__name__ = "OpaqueWandbCallback"
        wrapped = OpaqueWandbCallback
    elif class_name == "TensorBoardCallback":

        class OpaqueTensorBoardCallback(callback_cls):  # type: ignore[misc, valid-type]
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not state.is_world_process_zero:
                    return

                if self.tb_writer is None:
                    self._init_summary_writer(args)

                if self.tb_writer is not None:
                    for key, value in rewrite_logs_for_reporting(logs).items():
                        if isinstance(value, (int, float)):
                            self.tb_writer.add_scalar(key, value, state.global_step)
                        elif isinstance(value, str):
                            self.tb_writer.add_text(key, value, state.global_step)
                        else:
                            log.warning(
                                "Trainer is attempting to log a value of %r of "
                                "type %s for key %r as a scalar. This invocation "
                                "of TensorBoard's writer.add_scalar() is incorrect "
                                "so we dropped this attribute.",
                                value,
                                type(value),
                                key,
                            )

        OpaqueTensorBoardCallback.__name__ = "OpaqueTensorBoardCallback"
        wrapped = OpaqueTensorBoardCallback
    else:
        wrapped = callback_cls

    _WRAPPED_CALLBACK_CLASSES[callback_cls] = wrapped
    return wrapped


def build_callback_handler(
    args: TrainingArguments,
    model: Any,
    processing_class: Any,
    callbacks: list[TrainerCallback] | None = None,
) -> CallbackHandler:
    """Construct an HF ``CallbackHandler`` populated like ``Trainer.__init__``.

    Mirrors :class:`transformers.Trainer`'s init pattern:

    - ``DEFAULT_CALLBACKS`` (currently just :class:`DefaultFlowCallback`)
      come first so default cadence runs before user logic.
    - Reporting integration callbacks (W&B, TensorBoard, MLflow, …) are
      registered next, derived from ``args.report_to`` via HF's
      :func:`~transformers.integrations.get_reporting_integration_callbacks`.
    - User-supplied callbacks come after.
    - A progress callback is registered after construction:
      :class:`PrinterCallback` when ``args.disable_tqdm`` else
      :class:`ProgressCallback`.

    The handler stores ``optimizer=None`` and ``lr_scheduler=None`` since
    the DP path uses a functional optimizer + callable schedule.
    """
    reporting_callbacks = [
        wrap_reporting_callback_class(callback_cls)
        for callback_cls in get_reporting_integration_callbacks(
            getattr(args, "report_to", None)
        )
    ]
    user_callbacks = list(callbacks) if callbacks else []
    callback_list: list[Any] = [
        *DEFAULT_CALLBACKS,
        *reporting_callbacks,
        *user_callbacks,
    ]
    handler = CallbackHandler(
        callback_list,
        model,
        processing_class,
        None,  # optimizer
        None,  # lr_scheduler
    )
    handler.add_callback(PrinterCallback() if args.disable_tqdm else ProgressCallback())
    return handler
