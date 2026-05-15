"""DP trainer state — standalone dataclass for :class:`DPTrainer`.

Mirrors HF :class:`transformers.TrainerState`'s field *names* for the
fields we use (so HF reporting callbacks duck-typing
``state.global_step``, ``state.epoch``, ``state.log_history`` keep
working unchanged) without inheriting from it. The field surface is
the subset we actually use, plus DPTrainer-specific privacy bookkeeping
and an explicit serialisation ``version``.

Dropped vs HF:

- ``total_flos`` — FLOP counter never written by the DP path.
- ``is_hyper_param_search``, ``trial_name``, ``trial_params`` — HPO
  was removed in Phase C.

Schema divergence: ``trainer_state.json`` written by DPTrainer may
include privacy telemetry keys in ``log_history``. HF's strict
:meth:`transformers.TrainerState.load_from_json` would reject; ours
filters unknown top-level keys (forward compat).
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

__all__ = ["DPTrainerState"]


_STATE_VERSION = 1


@dataclasses.dataclass
class DPTrainerState:
    """Trainer state for DPTrainer (standalone; not a ``TrainerState`` subclass).

    HF callbacks (TensorBoard, WandB, EarlyStoppingCallback, …) read
    ``state.global_step``, ``state.epoch``, ``state.log_history`` etc.
    via attribute access — not ``isinstance(state, TrainerState)`` —
    so duck-typing parity is preserved by keeping the field names.
    """

    # HF-named fields we use (parity for callbacks)
    epoch: float = 0.0
    global_step: int = 0
    max_steps: int = 0
    logging_steps: float = 500
    eval_steps: float | None = 500
    save_steps: int = 500
    train_batch_size: int | None = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    log_history: list[dict[str, float]] = dataclasses.field(default_factory=list)
    best_metric: float | None = None
    best_global_step: int | None = None
    best_model_checkpoint: str | None = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    stateful_callbacks: dict[str, Any] = dataclasses.field(default_factory=dict)

    # DPTrainer-specific privacy bookkeeping
    privacy_resolved_delta: float | None = None
    privacy_resolved_noise_multiplier: float | None = None
    privacy_noise_multiplier_source: str | None = None
    privacy_sample_rate: float | None = None
    privacy_expected_batch_size: int | None = None
    privacy_total_steps: int | None = None

    # Serialisation version; bumped on schema changes.
    version: int = _STATE_VERSION

    def compute_steps(self, args: Any, max_steps: int) -> None:
        """Resolve absolute step counts for logging/eval/save.

        HF parity: fractional values < 1 are interpreted as fractions of
        ``max_steps`` and rounded up; ``None`` is left alone.
        """
        for kind in ("logging", "eval", "save"):
            num_steps = getattr(args, f"{kind}_steps", None)
            if num_steps is None:
                continue
            if num_steps < 1:
                num_steps = math.ceil(max_steps * num_steps)
            setattr(self, f"{kind}_steps", num_steps)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for ``trainer_state.json``."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DPTrainerState":
        """Reconstruct from a dict loaded from ``trainer_state.json``.

        Unknown keys are filtered (forward-compat with newer writers).
        Missing ``version`` is tolerated and treated as legacy v0;
        a version greater than our own raises ``ValueError``.
        """
        version = data.get("version")
        if version is not None and version > _STATE_VERSION:
            raise ValueError(
                f"checkpoint trainer_state.json version {version} is not "
                f"supported by this DPTrainer (expected <= {_STATE_VERSION})."
            )
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
