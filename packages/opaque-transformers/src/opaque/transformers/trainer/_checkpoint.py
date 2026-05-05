"""Checkpoint helpers for DPTrainer.

Defines the on-disk layout (parallel to HuggingFace ``Trainer``), discovery and
rotation utilities, an RNG snapshot helper, and a thin DP-runtime bundle that
delegates serialization to each Opaque type's ``state_dict`` / ``from_state_dict``.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
from typing import Any

import numpy as np
import torch
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from opaque.clipping.types import FixedClipState
from opaque.dpsgd.clipping.adaptive import AdaptiveClipState
from opaque.dpsgd.noise.gaussian import GaussianNoiseState

log = logging.getLogger(__name__)

# Filename layout.  ``training_args.bin`` deliberately matches HF's
# ``TRAINING_ARGS_NAME`` so HF tooling that reads it back via
# ``torch.load(.../training_args.bin)`` works on DPTrainer checkpoints —
# ``DPTrainingArguments`` is a subclass of ``TrainingArguments``, so the
# field surface is a superset and HF's reader accepts it.  Genuinely
# DP-specific files keep the ``dp_`` prefix because their contents are
# *not* HF-compatible (torchopt pytree vs torch.optim state_dict;
# clip/noise/sampler bundle vs nothing in HF).
TRAINING_ARGS_NAME = "training_args.bin"
DP_OPTIMIZER_NAME = "dp_optimizer.pt"
DP_RUNTIME_STATE_NAME = "dp_runtime_state.pt"
DP_ACCOUNTANT_NAME = "accountant.json"
# rng_state.pth keeps HF's filename AND matches HF's schema (cpu/cuda keys),
# so DPTrainer-saved checkpoints can be read by HF Trainer without crashing.
RNG_STATE_NAME = "rng_state.pth"

_CHECKPOINT_RE = re.compile(rf"^{re.escape(PREFIX_CHECKPOINT_DIR)}\-(\d+)$")

__all__ = [
    "PREFIX_CHECKPOINT_DIR",
    "WEIGHTS_NAME",
    "SAFE_WEIGHTS_NAME",
    "DP_OPTIMIZER_NAME",
    "TRAINING_ARGS_NAME",
    "TRAINER_STATE_NAME",
    "DP_RUNTIME_STATE_NAME",
    "DP_ACCOUNTANT_NAME",
    "RNG_STATE_NAME",
    "parse_checkpoint_step",
    "list_checkpoints",
    "get_last_checkpoint",
    "rotate_checkpoints",
    "rng_state_path",
    "snapshot_rng_state",
    "restore_rng_state",
    "save_dp_runtime_state",
    "load_dp_runtime_state",
]


def rng_state_path(ckpt_dir: str, *, rank: int = 0, world_size: int = 1) -> str:
    """Resolve the RNG-snapshot path for ``rank`` in a ``world_size``-process run.

    Single-process: ``rng_state.pth`` (HF parity, unchanged on disk).
    Multi-process: ``rng_state_{rank}.pth`` (forward-compat with HF DDP).

    Lives here so save / load go through one canonical resolver — when
    Phase 9 lands DDP support, only the call sites that pass a non-default
    ``world_size`` need to change.
    """
    if world_size <= 1:
        return os.path.join(ckpt_dir, RNG_STATE_NAME)
    return os.path.join(ckpt_dir, f"rng_state_{int(rank)}.pth")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def parse_checkpoint_step(path: str) -> int | None:
    """Extract the step number from a ``checkpoint-N`` directory name."""
    m = _CHECKPOINT_RE.match(os.path.basename(path.rstrip(os.sep)))
    return int(m.group(1)) if m is not None else None


def list_checkpoints(folder: str) -> list[str]:
    """Return ``checkpoint-N`` subdirectories of ``folder`` sorted by step ascending."""
    if not os.path.isdir(folder):
        return []
    found: list[tuple[int, str]] = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if not os.path.isdir(full):
            continue
        step = parse_checkpoint_step(name)
        if step is not None:
            found.append((step, full))
    return [path for _, path in sorted(found)]


def get_last_checkpoint(folder: str) -> str | None:
    """Return the ``checkpoint-N`` with the highest step, or ``None``."""
    paths = list_checkpoints(folder)
    return paths[-1] if paths else None


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def rotate_checkpoints(
    output_dir: str,
    save_total_limit: int | None,
    best_model_checkpoint: str | None = None,
) -> None:
    """Delete oldest checkpoints to honor ``save_total_limit``.

    Always protects the most recent checkpoint and the best-model checkpoint
    (when supplied). Effective keep count is
    ``max(save_total_limit, len(protected))`` to avoid deleting either.
    """
    if save_total_limit is None or save_total_limit <= 0:
        return

    checkpoints = list_checkpoints(output_dir)
    if len(checkpoints) <= save_total_limit:
        return

    protected: set[str] = {checkpoints[-1]}
    if best_model_checkpoint is not None:
        best_abs = os.path.abspath(best_model_checkpoint)
        for path in checkpoints:
            if os.path.abspath(path) == best_abs:
                protected.add(path)
                break

    keep = max(save_total_limit, len(protected))
    num_to_delete = max(0, len(checkpoints) - keep)
    deleted = 0
    for path in checkpoints:
        if deleted >= num_to_delete:
            break
        if path in protected:
            continue
        log.info("Deleting older checkpoint %s due to save_total_limit=%d", path, save_total_limit)
        shutil.rmtree(path, ignore_errors=True)
        deleted += 1


# ---------------------------------------------------------------------------
# RNG snapshot
# ---------------------------------------------------------------------------


def snapshot_rng_state() -> dict[str, Any]:
    """Capture python / numpy / torch CPU + per-CUDA-device + MPS RNG states.

    Schema matches HuggingFace ``Trainer``: keys are ``python``, ``numpy``,
    ``cpu``, ``cuda`` (list of per-device tensors when CUDA is available),
    and ``mps`` (when MPS is available).
    """
    snap: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.random.get_rng_state_all()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        snap["mps"] = torch.mps.get_rng_state()
    return snap


def restore_rng_state(snap: dict[str, Any]) -> None:
    """Apply a previously captured RNG snapshot.

    Tolerates HF's schema (``cpu``/``cuda``/``mps``) and the legacy keys
    (``torch_cpu``/``torch_cuda``) used by very early DPTrainer builds.
    """
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    cpu_state = snap.get("cpu", snap.get("torch_cpu"))
    if cpu_state is not None:
        torch.set_rng_state(cpu_state)
    cuda_states = snap.get("cuda", snap.get("torch_cuda"))
    if cuda_states is not None and torch.cuda.is_available():
        for i, st in enumerate(cuda_states):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(st, i)
    mps_state = snap.get("mps")
    if (
        mps_state is not None
        and hasattr(torch, "mps")
        and torch.backends.mps.is_available()
    ):
        torch.mps.set_rng_state(mps_state)


# ---------------------------------------------------------------------------
# DP runtime bundle (clip_state + noise_state + scheduling + sampler state)
# ---------------------------------------------------------------------------


def _serialize_clip_state(state: Any) -> dict[str, Any]:
    if isinstance(state, FixedClipState):
        return {"type": "FixedClipState", "data": state.state_dict()}
    if isinstance(state, AdaptiveClipState):
        return {"type": "AdaptiveClipState", "data": state.state_dict()}
    raise TypeError(f"Unsupported clip_state type: {type(state).__name__}")


def _deserialize_clip_state(blob: dict[str, Any]) -> Any:
    kind = blob["type"]
    data = blob["data"]
    if kind == "FixedClipState":
        return FixedClipState.from_state_dict(data)
    if kind == "AdaptiveClipState":
        return AdaptiveClipState.from_state_dict(data)
    raise ValueError(f"Unknown clip_state type: {kind}")


def _serialize_noise_state(state: Any) -> dict[str, Any]:
    if isinstance(state, GaussianNoiseState):
        return {"type": "GaussianNoiseState", "data": state.state_dict()}
    raise TypeError(f"Unsupported noise_state type: {type(state).__name__}")


def _deserialize_noise_state(blob: dict[str, Any]) -> Any:
    kind = blob["type"]
    data = blob["data"]
    if kind == "GaussianNoiseState":
        return GaussianNoiseState.from_state_dict(data)
    raise ValueError(f"Unknown noise_state type: {kind}")


def save_dp_runtime_state(
    path: str,
    *,
    clip_state: Any,
    noise_state: Any,
    sampler_state: dict[str, Any] | None,
    sample_rate: float,
    target_delta: float,
    noise_multiplier: float,
    expected_steps_per_epoch: int,
    expected_batch_size: int,
    total_steps: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save the DP runtime bundle (everything under DP_RUNTIME_STATE_NAME)."""
    payload: dict[str, Any] = {
        "version": 1,
        "clip_state": _serialize_clip_state(clip_state),
        "noise_state": _serialize_noise_state(noise_state),
        "sampler_state": sampler_state,
        "sample_rate": float(sample_rate),
        "target_delta": float(target_delta),
        "noise_multiplier": float(noise_multiplier),
        "expected_steps_per_epoch": int(expected_steps_per_epoch),
        "expected_batch_size": int(expected_batch_size),
        "total_steps": int(total_steps),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)


def load_dp_runtime_state(path: str) -> dict[str, Any]:
    """Load the DP runtime bundle. ``clip_state`` and ``noise_state`` come back deserialized.

    ``weights_only=False`` is required: the bundle includes a torchopt
    pytree (optimizer state) and our serialized clip/noise state
    objects — none are plain tensor maps that PyTorch's safe-load
    path can reconstruct.  PyTorch 2.6+ flips the default to ``True``;
    keeping the explicit ``False`` here pins the behaviour we tested
    against.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["clip_state"] = _deserialize_clip_state(payload["clip_state"])
    payload["noise_state"] = _deserialize_noise_state(payload["noise_state"])
    return payload
