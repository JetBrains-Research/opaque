"""DP trainer state — re-export of HF ``TrainerState``.

Inherits HF's full ``TrainerState`` field surface and methods
(``compute_steps`` / ``save_to_json`` / ``load_from_json`` /
``init_training_references``).  Adds dict-based :meth:`to_json` /
:meth:`from_json` used by the trainer's checkpoint plumbing — HF's
``save_to_json`` / ``load_from_json`` are file-based and strict on
unknown keys; :meth:`from_json` filters unknown keys so checkpoints
written by a future version remain readable.

Schema divergence vs HF: ``trainer_state.json`` written by DPTrainer may include
privacy telemetry keys in ``log_history`` that trip HF's strict
:meth:`transformers.TrainerState.load_from_json`.  Use :meth:`from_json`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from transformers.trainer_callback import TrainerControl as DPTrainerControl
from transformers.trainer_callback import TrainerState

__all__ = ["DPTrainerControl", "DPTrainerState"]


@dataclasses.dataclass
class DPTrainerState(TrainerState):
    """Trainer state for DPTrainer (subclass of HF ``TrainerState``).

    Adds run-resolved privacy bookkeeping populated by :class:`DPTrainer`
    during ``_setup_training`` (so callbacks and tooling can read stable
    values from ``state`` rather than mutable ``args`` fields).
    """

    privacy_resolved_delta: float | None = None
    privacy_resolved_noise_multiplier: float | None = None
    privacy_noise_multiplier_source: str | None = None
    privacy_sample_rate: float | None = None
    privacy_expected_batch_size: int | None = None
    privacy_total_steps: int | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible dict for ``trainer_state.json``."""
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "DPTrainerState":
        """Reconstruct from a dict loaded from ``trainer_state.json``.

        Unknown keys are ignored so checkpoints written by a future
        version of the trainer can still be loaded by older code (lossy
        but tolerant).
        """
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ``DPTrainerControl`` is a direct re-export of HF's ``TrainerControl``.
# Re-exported so callers can rely on ``isinstance(control,
# DPTrainerControl)`` while also benefitting from HF's
# ``ExportableState`` protocol (``state()`` / ``from_state()``) and the
# ``_new_step`` / ``_new_epoch`` / ``_new_training`` reset helpers used
# by the callback flag lifecycle.
