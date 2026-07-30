"""Evaluation helpers for :class:`DPTrainer`.

This module hosts the Opaque-owned eval container plus pure helpers used
by the eval loop:

- :class:`EvaluationResult` — single dataclass returned by
  :meth:`DPTrainer.evaluate`, :meth:`DPTrainer.predict`, and
  :meth:`DPTrainer.evaluation_loop`.  Replaces HF's split
  ``EvalLoopOutput`` / ``PredictionOutput`` pair.
- :class:`_PredictionAccumulator` — collects per-batch losses, predictions,
  labels, and (optionally) inputs across an eval loop, with a hot/cold
  buffer split that keeps CPU-offload flushes O(K·N) (one move per batch
  group, no re-flush of already-frozen chunks).
- :func:`should_run_eval_at_step` — pure decision helper for the
  training-loop-driven eval trigger; encodes ``eval_strategy`` × ``eval_delay``
  semantics.
- :func:`with_metric_prefix` — adds ``{prefix}_`` to keys that don't already
  start with it (HF parity).
- :func:`speed_metrics` — pure helper mirroring HF's
  ``transformers.trainer_utils.speed_metrics`` so eval/predict reports
  expose ``{prefix}_runtime``, ``{prefix}_samples_per_second``,
  ``{prefix}_steps_per_second``.

``EvalPrediction`` is re-exported from ``transformers.trainer_utils`` as the
canonical input shape for user-supplied ``compute_metrics`` callbacks.
"""

from __future__ import annotations

import dataclasses
import time
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from transformers.trainer_pt_utils import (
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
)
from transformers.trainer_utils import (
    EvalPrediction,
    denumpify_detensorize,
)

from .types import EvaluationResult  # re-export; canonical home is types.py

if TYPE_CHECKING:
    from ._training_arguments import TrainingArguments


__all__ = [
    "EvalPrediction",
    "EvaluationResult",
    "_PredictionAccumulator",
    "denumpify_detensorize",
    "find_batch_size",
    "nested_numpify",
    "nested_truncate",
    "resolve_eval_num_samples",
    "should_run_eval_at_step",
    "speed_metrics",
    "with_metric_prefix",
]


# ``-100`` is HF's universal padding sentinel for eval tensors — labels,
# logits, and inputs alike.  ``compute_metrics`` users mask this value to
# skip ignored positions; padding logits with ``0`` would silently leak
# ignored positions into accuracy / perplexity computations.
_HF_PAD_VALUE = -100


# ---------------------------------------------------------------------------
# _PredictionAccumulator
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _PredictionAccumulator:
    """Collects per-batch eval tensors with optional CPU-offload cadence.

    The accumulator is bypassed when ``prediction_loss_only`` is ``True``:
    only per-example losses are retained and ``finalize`` returns
    ``predictions=None`` and ``label_ids=None``.

    Storage layout (HF parity):

    - ``_hot_*`` lists hold the *current* on-device batch group.
    - ``_cold_*`` lists hold previously-flushed chunks already moved to CPU
      and concatenated so each entry is a single tensor (a "frozen chunk").

    ``flush_to_cpu`` concatenates and moves the hot buffers to CPU,
    appends each result as one tensor to its cold list, then resets the
    hot buffers.  This makes per-flush work O(N) in the size of the
    *current* group (not the full accumulated history) — overall
    ``K`` flushes over a run that produces ``K·N`` batches cost
    ``O(K·N)`` rather than the ``O(K²·N)`` of repeatedly moving
    already-flushed tensors.

    When ``eval_do_concat_batches`` is ``True`` (HF default), ``finalize``
    concatenates the cold + trailing-hot chunks into one tensor.  When
    ``False``, the per-batch list is returned (no concat) so
    ``compute_metrics`` can introspect each batch separately.

    ``EvalPrediction.losses`` is populated with **per-example** losses —
    each batch's reduced loss is repeated by its batch size before being
    appended (HF parity, mirroring ``losses.repeat(batch_size)``).  This
    removes the silent corruption that would otherwise occur when a
    custom head returns a sum-reduced loss.
    """

    prediction_loss_only: bool = False
    eval_accumulation_steps: int | None = None
    eval_do_concat_batches: bool = True
    include_inputs: bool = False
    include_losses: bool = False

    # On-device "hot" buffers — current batch group; flushed periodically.
    _hot_losses: list[Tensor] = dataclasses.field(default_factory=list)
    _hot_logits: list[Any] = dataclasses.field(default_factory=list)
    _hot_labels: list[Any] = dataclasses.field(default_factory=list)
    # ``inputs`` is the *bare* main-input tensor (HF parity:
    # ``EvalLoopContainer`` collects ``inputs_decode = inputs[main_input_name]``
    # — a single tensor, not a dict).
    _hot_inputs: list[Tensor] = dataclasses.field(default_factory=list)

    # On-CPU "cold" buffers — list of frozen, already-concatenated chunks.
    _cold_losses: list[Tensor] = dataclasses.field(default_factory=list)
    _cold_logits: list[Tensor] = dataclasses.field(default_factory=list)
    _cold_labels: list[Tensor] = dataclasses.field(default_factory=list)
    _cold_inputs: list[Tensor] = dataclasses.field(default_factory=list)

    _num_batches: int = 0

    def add(
        self,
        loss: Tensor | None,
        logits: Any | None,
        labels: Any | None,
        inputs: Tensor | None,
        *,
        batch_size: int,
    ) -> None:
        """Append one batch's tensors to the on-device "hot" buffers.

        ``batch_size`` is required so per-example losses can be produced
        even when the model returns a scalar, batch-reduced loss
        (``losses.repeat(bs)``, HF parity).

        ``inputs`` is the bare main-input tensor (HF parity:
        ``inputs_decode = inputs[main_input_name]``), or ``None`` if the
        collator didn't emit the primary input.
        """
        # ``loss`` is either scalar (the standard ``prediction_step``
        # path: a batch-mean reduced by the model's ``forward``) or 1-D
        # of length ``batch_size`` (the vmap'd eval closure path
        # triggered by ``'loss' in include_for_metrics``).  We store
        # only the real per-example track; scalar losses are discarded
        # at this point because they carry no per-example information
        # — populating ``EvalPrediction.losses`` from a replicated
        # batch-mean is fake-by-construction and HF-misleading.
        if self.include_losses and loss is not None and loss.ndim > 0:
            self._hot_losses.append(loss.detach())

        if not self.prediction_loss_only:
            if logits is not None:
                self._hot_logits.append(nested_detach(logits))
            if labels is not None:
                self._hot_labels.append(nested_detach(labels))
            if self.include_inputs and inputs is not None:
                self._hot_inputs.append(inputs.detach())

        self._num_batches += 1

        # In ``eval_do_concat_batches=False`` mode we MUST keep per-batch
        # tensors as separate cold chunks (the contract is "list of
        # per-batch tensors").  Force a flush after every add so each
        # cold chunk corresponds to exactly one batch.
        flush_now = not self.eval_do_concat_batches
        if not flush_now and self.eval_accumulation_steps is not None:
            flush_now = self._num_batches % self.eval_accumulation_steps == 0
        if flush_now:
            self.flush_to_cpu()

    def flush_to_cpu(self) -> None:
        """Move the *current* hot buffer group to CPU and freeze it.

        Each per-tensor list is concatenated into a single CPU tensor
        appended to the corresponding ``_cold_*`` list; the hot buffers
        are then reset.  Already-cold chunks are not touched — this is
        the O(K·N) optimization over the prior ``[t.to("cpu") for t in ...]``
        re-flush.
        """
        if self._hot_losses:
            self._cold_losses.append(_freeze_hot_chunk(self._hot_losses))
            self._hot_losses = []

        if self._hot_logits:
            self._cold_logits.append(
                _freeze_hot_chunk(self._hot_logits, pad_value=_HF_PAD_VALUE)
            )
            self._hot_logits = []

        if self._hot_labels:
            self._cold_labels.append(
                _freeze_hot_chunk(self._hot_labels, pad_value=_HF_PAD_VALUE)
            )
            self._hot_labels = []

        if self._hot_inputs:
            self._cold_inputs.append(
                _freeze_hot_chunk(self._hot_inputs, pad_value=_HF_PAD_VALUE)
            )
            self._hot_inputs = []

    def finalize(
        self,
        *,
        num_samples: int | None = None,
        gather: bool = False,
    ) -> tuple[
        Any | None,
        Any | None,
        Any | None,
        Any | None,
    ]:
        """Return ``(predictions, label_ids, inputs, losses)`` as numpy arrays.

        ``predictions`` / ``label_ids`` are ``None`` when
        ``prediction_loss_only`` is ``True`` or no logits / labels were
        ever added.  ``inputs`` is a single tensor (HF parity:
        ``inputs_decode`` collected from ``inputs[main_input_name]``)
        or ``None`` when the user didn't request it / the collator
        didn't emit a primary input.

        ``losses`` is a 1-D numpy array of length ``total_samples`` when
        ``include_losses`` is ``True``, otherwise ``None``.  Per-example
        — *not* per-batch — semantics (HF parity).

        ``num_samples`` (when set) truncates each leading-dim of the
        returned tensors to that length via HF's
        :func:`~transformers.trainer_pt_utils.nested_truncate` — drops
        gather-padding rows distributed gather may introduce.  Pass
        ``None`` to skip truncation.

        ``gather=True`` all-gathers each tensor pytree
        across DDP ranks via :func:`opaque.distributed.gather_pytree`
        *before* numpify, so per-rank shards are concatenated into the
        cluster-wide result.  Single-process eval should leave the
        default (``False``).

        Tensors are converted to numpy arrays via HF's
        :func:`~transformers.trainer_pt_utils.nested_numpify` so user
        ``compute_metrics`` callbacks receive the same types as HF
        ``Trainer`` would deliver (works for the ``evaluate`` /
        ``sklearn`` / ``seqeval`` ecosystem).  When
        ``eval_do_concat_batches=False`` the per-batch chunks are
        preserved as a list (mirrors HF's ``EvalLoopContainer`` with
        ``do_nested_concat=False``).
        """
        # Final flush so any trailing hot tensors land on CPU before concat.
        self.flush_to_cpu()

        predictions = self._collect_chunks(
            self._cold_logits,
            empty_ok=self.prediction_loss_only,
            pad_value=_HF_PAD_VALUE,
        )
        labels = self._collect_chunks(
            self._cold_labels,
            empty_ok=self.prediction_loss_only,
            pad_value=_HF_PAD_VALUE,
        )

        inputs: Any | None
        if self.include_inputs and self._cold_inputs:
            inputs = self._collect_chunks(
                self._cold_inputs,
                empty_ok=False,
                pad_value=_HF_PAD_VALUE,
            )
        else:
            inputs = None

        losses: Any | None
        if self.include_losses and self._cold_losses:
            # Per-example losses are 1-D; stacking via cat is correct.
            losses = torch.cat(self._cold_losses, dim=0)
        else:
            losses = None

        # all-gather each tensor pytree across ranks so the
        # numpy outputs are cluster-wide, not per-rank shards.
        if gather:
            from opaque.api.engine.distributed._state import gather_pytree

            if predictions is not None:
                predictions = gather_pytree(predictions)
            if labels is not None:
                labels = gather_pytree(labels)
            if inputs is not None:
                inputs = gather_pytree(inputs)
            if losses is not None:
                losses = gather_pytree(losses)

        # HF parity: ``compute_metrics`` consumes numpy arrays, not
        # ``torch.Tensor``.  ``nested_numpify`` recurses into lists / dicts
        # / tuples so all four containers are converted uniformly.
        if predictions is not None:
            predictions = nested_numpify(predictions)
        if labels is not None:
            labels = nested_numpify(labels)
        if inputs is not None:
            inputs = nested_numpify(inputs)
        if losses is not None:
            losses = nested_numpify(losses)

        # HF parity (trainer.py: end of ``evaluation_loop``): truncate
        # leading dim to the dataset's true sample count so user
        # ``compute_metrics`` callbacks see ``predictions.shape[0] ==
        # num_samples`` regardless of any padding the gather/pad path
        # introduced upstream.  No-op for single-process eval (where
        # ``sum(batch_sizes) == num_samples`` already); needed under
        # distributed eval where the pad makes the gather rectangular.
        if num_samples is not None:
            if predictions is not None:
                predictions = nested_truncate(predictions, num_samples)
            if labels is not None:
                labels = nested_truncate(labels, num_samples)
            if inputs is not None:
                inputs = nested_truncate(inputs, num_samples)
            if losses is not None:
                losses = nested_truncate(losses, num_samples)

        return predictions, labels, inputs, losses

    def _collect_chunks(
        self,
        cold: list[Any],
        *,
        empty_ok: bool,
        pad_value: int | float,
    ) -> Any | list[Any] | None:
        if not cold:
            # An empty container always collapses to ``None`` (HF parity:
            # ``compute_metrics`` receives ``None`` for predictions/labels in
            # loss-only mode).  ``empty_ok`` is retained as caller intent —
            # predictions/labels pass ``prediction_loss_only``; ``inputs`` is
            # only collected behind a non-empty guard.
            del empty_ok
            return None
        if self.eval_do_concat_batches:
            return _concat_nested_chunks(cold, padding_value=pad_value)
        return list(cold)


def _to_cpu_nested(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.to("cpu")
    if isinstance(value, tuple):
        return tuple(_to_cpu_nested(v) for v in value)
    if isinstance(value, list):
        return [_to_cpu_nested(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_cpu_nested(v) for k, v in value.items()}
    return value


def _concat_nested_chunks(
    tensors: list[Any],
    *,
    padding_value: int | float = _HF_PAD_VALUE,
) -> Any:
    if len(tensors) == 1:
        return tensors[0]
    concatenated = tensors[0]
    for tensor in tensors[1:]:
        concatenated = nested_concat(
            concatenated,
            tensor,
            padding_index=padding_value,
        )
    return concatenated


def _freeze_hot_chunk(
    tensors: list[Any],
    *,
    pad_value: int | float = 0,
) -> Any:
    """Concatenate a hot list and move the result to CPU as a single tensor.

    Variable trailing-dim sizes (common for causal-LM eval logits) are
    right-padded with ``pad_value`` before concat.
    """
    if len(tensors) == 1:
        return _to_cpu_nested(tensors[0])
    concatenated = _concat_nested_chunks(tensors, padding_value=pad_value)
    return _to_cpu_nested(concatenated)


# ---------------------------------------------------------------------------
# Trigger / formatting helpers
# ---------------------------------------------------------------------------


def should_run_eval_at_step(
    args: TrainingArguments,
    global_step: int,
    epoch: float,
    eval_steps_resolved: int,
) -> bool:
    """Return ``True`` if the training loop should fire eval at this step.

    Encodes HF's ``eval_strategy`` × ``eval_delay`` semantics:

    - ``eval_strategy="no"`` always returns ``False`` (the per-loop trigger
      never fires; ``eval_on_start`` is handled separately by the caller).
    - ``eval_strategy="steps"``: fires every ``eval_steps_resolved`` steps,
      with ``eval_delay`` interpreted in **steps** (skip until
      ``global_step >= eval_delay``).
    - ``eval_strategy="epoch"``: fires only at integer epoch boundaries (the
      caller is expected to invoke this helper at the end of each epoch),
      with ``eval_delay`` interpreted in **epochs** (skip until
      ``epoch >= eval_delay``).

    This helper is intentionally pure so it can be unit-tested without a
    model.
    """
    strategy = args.eval_strategy
    if strategy == "no":
        return False

    delay = float(args.eval_delay or 0)

    if strategy == "steps":
        if global_step < delay:
            return False
        if eval_steps_resolved <= 0:
            return False
        return global_step > 0 and global_step % eval_steps_resolved == 0

    if strategy == "epoch":
        # Only at integer-epoch boundaries.  The caller passes the running
        # ``epoch`` value; we accept floats and require integer-equality.
        if not float(epoch).is_integer():
            return False
        return epoch > 0 and epoch >= delay

    # Unknown strategy — be conservative.
    return False


def speed_metrics(
    prefix: str,
    start_time: float,
    *,
    num_samples: int | None = None,
    num_steps: int | None = None,
) -> dict[str, float]:
    """Compute throughput metrics for an eval / predict pass.

    Pure helper mirroring ``transformers.trainer_utils.speed_metrics`` —
    we don't import HF's version because it is part of an unstable
    private surface.  Returned keys (when the corresponding count is
    available):

    - ``{prefix}_runtime`` — wall time in seconds (always emitted).
    - ``{prefix}_samples_per_second`` — ``num_samples / runtime``.
    - ``{prefix}_steps_per_second`` — ``num_steps / runtime``.

    All values are rounded to three decimals (HF parity —
    ``transformers.trainer_utils.speed_metrics`` rounds with
    ``round(x, 4)`` for runtime but ``round(x, 3)`` for derived rates;
    we round all three uniformly to 3 to keep dashboard parity tight).
    """
    runtime = max(time.monotonic() - start_time, 1e-9)
    out: dict[str, float] = {f"{prefix}_runtime": round(runtime, 4)}
    if num_samples is not None:
        out[f"{prefix}_samples_per_second"] = round(float(num_samples) / runtime, 3)
    if num_steps is not None:
        out[f"{prefix}_steps_per_second"] = round(float(num_steps) / runtime, 3)
    return out


def resolve_eval_num_samples(dataloader: Any, *, observed: int) -> int:
    """Resolve the ``num_samples`` field of an :class:`EvaluationResult`.

    HF parity (``transformers.trainer.Trainer.evaluation_loop``,
    trainer.py:4757-4769): prefer the dataset's reported length, then the
    dataloader-driven count, then the observed batch sums.  Streaming
    iterables expose neither length so we land on the observed count.

    ``observed`` is the running batch-size sum recorded by the eval loop;
    we use it as both the final fallback and as a non-zero rescue when
    upstream length probes return zero (HF does the same).
    """
    dataset = getattr(dataloader, "dataset", None)
    # 1. dataset.__len__ (map-style or finite IterableDataset).
    try:
        if dataset is not None:
            return len(dataset)
    except TypeError:
        pass
    # 2. Sharded iterable datasets carry a usable ``num_examples`` attr.
    if dataset is not None:
        n = getattr(dataset, "num_examples", 0)
        if isinstance(n, int) and n > 0:
            return n
    # 3. Dataloader's own length × declared batch size.
    try:
        n_batches = len(dataloader)
        bs = int(getattr(dataloader, "batch_size", 0) or 0)
        if n_batches > 0 and bs > 0:
            return n_batches * bs
    except TypeError:
        pass
    # 4. Observed batch sums.
    return int(observed)


def with_metric_prefix(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return a copy of ``metrics`` with ``{prefix}_`` prepended where missing.

    Mirrors HF's behavior: keys that already start with ``{prefix}_`` pass
    through unchanged.  This keeps user-supplied ``compute_metrics`` outputs
    HF-compatible regardless of whether the user pre-prefixes their keys.
    """
    if not prefix:
        return dict(metrics)
    out: dict[str, Any] = {}
    head = f"{prefix}_"
    for k, v in metrics.items():
        out[k if k.startswith(head) else f"{head}{k}"] = v
    return out
