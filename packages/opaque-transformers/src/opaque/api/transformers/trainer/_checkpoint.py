"""Checkpoint helpers for DPTrainer.

Defines the on-disk layout (parallel to HuggingFace ``Trainer``), discovery and
rotation utilities, an RNG snapshot helper, and the DP-side runtime bundle
stored under ``dp_state.pt``.  Clip / noise slices use
:func:`opaque.serialization.state_dict`; resume merges them with the live
training context via :func:`opaque.serialization.from_state_dict`.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import random
import re
import shutil
from collections.abc import Mapping
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from opaque.exceptions import CheckpointError, ConfigurationError
from opaque.serialization import state_dict as opaque_state_dict
from opaque.types import ClipState, NoiseState
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME

from ._participation import ResolvedParticipationPlan

log = logging.getLogger(__name__)

# Filename layout: ``training_args.bin`` matches HF's ``TRAINING_ARGS_NAME``.
TRAINING_ARGS_NAME = "training_args.bin"
DP_OPTIMIZER_NAME = "dp_optimizer.pt"
DP_STATE_NAME = "dp_state.pt"
DP_ACCOUNTANT_NAME = "accountant.json"
RNG_STATE_NAME = "rng_state.pth"

# Version 7 records calibration provenance plus versioned MF runtime state,
# including bounded BISR history and sensitivity latches. Participation-plan
# provenance and the execution-aligned sampler-cursor origin are optional
# additive fields so safe historical v7 bundles remain inspectable and
# resumable.
DP_STATE_BUNDLE_VERSION = 7

_CHECKPOINT_RE = re.compile(rf"^{re.escape(PREFIX_CHECKPOINT_DIR)}\-(\d+)$")

__all__ = [
    "DP_ACCOUNTANT_NAME",
    "DP_OPTIMIZER_NAME",
    "DP_STATE_BUNDLE_VERSION",
    "DP_STATE_NAME",
    "PREFIX_CHECKPOINT_DIR",
    "RNG_STATE_NAME",
    "SAFE_WEIGHTS_NAME",
    "TRAINER_STATE_NAME",
    "TRAINING_ARGS_NAME",
    "WEIGHTS_NAME",
    "RuntimeCheckpoint",
    "get_last_checkpoint",
    "list_checkpoints",
    "load_dp_runtime_state",
    "parse_checkpoint_step",
    "restore_rng_state",
    "rng_state_path",
    "rotate_checkpoints",
    "save_dp_runtime_state",
    "snapshot_rng_state",
    "validate_dp_runtime_for_resume",
    "validate_sampler_cursor_for_resume",
]


def rng_state_path(ckpt_dir: str, *, rank: int = 0, world_size: int = 1) -> str:
    """Resolve the RNG-snapshot path for ``rank`` in a ``world_size``-process run."""
    if world_size <= 1:
        return str(Path(ckpt_dir) / RNG_STATE_NAME)
    return str(Path(ckpt_dir) / f"rng_state_{int(rank)}.pth")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def parse_checkpoint_step(path: str) -> int | None:
    """Extract the step number from a ``checkpoint-N`` directory name."""
    m = _CHECKPOINT_RE.match(Path(path.rstrip("/\\")).name)
    return int(m.group(1)) if m is not None else None


def list_checkpoints(folder: str) -> list[str]:
    """Return ``checkpoint-N`` subdirectories of ``folder`` sorted by step ascending."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        return []
    found: list[tuple[int, str]] = []
    for child in folder_path.iterdir():
        if not child.is_dir():
            continue
        step = parse_checkpoint_step(child.name)
        if step is not None:
            found.append((step, str(child)))
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
        best_abs = Path(best_model_checkpoint).resolve()
        for path in checkpoints:
            if Path(path).resolve() == best_abs:
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
        log.info(
            "Deleting older checkpoint %s due to save_total_limit=%d",
            path,
            save_total_limit,
        )
        shutil.rmtree(path, ignore_errors=True)
        deleted += 1


# ---------------------------------------------------------------------------
# RNG snapshot
# ---------------------------------------------------------------------------


def snapshot_rng_state() -> dict[str, Any]:
    """Capture python / numpy / torch CPU + per-CUDA-device + MPS RNG states."""
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
    """Apply a previously captured RNG snapshot."""
    random.setstate(snap["python"])
    np.random.set_state(snap["numpy"])
    torch.set_rng_state(snap["cpu"])
    cuda_states = snap.get("cuda")
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


@dataclasses.dataclass
class RuntimeCheckpoint:
    """Typed payload of ``dp_state.pt`` — the DP-side runtime bundle.

    Replaces the prior ad-hoc ``dict[str, Any]`` so the resume code is
    type-driven (attribute access, ``dataclasses.fields(...)`` iteration)
    instead of string-keyed.  Adding a new field is a single edit on
    this dataclass; ``_apply_runtime_state`` and ``_warn_on_arg_drift``
    in the trainer pick it up automatically.

    Each field tagged ``compare_on_resume=True`` carries a ``drift``
    disposition that controls what happens when phase-2's value differs
    from phase-1's saved value:

    - ``"dp_relevant"`` — affects privacy accounting.  DP-SGD warns
      (heterogeneous RDP composition still yields a correct ε); DP-FTRL
      raises (the matrix-factorization strategy is shape-locked for the
      original composition, so drift would silently compose a different
      ε).
    - ``"shape"`` — affects training trajectory (LR schedule, etc.) but
      not privacy.  Warns.
    - ``"intentional_extend"`` — silently allowed.  Used for ``total_steps``
      in the DP-SGD path where extending training is a normal user action.
    - dict form, e.g. ``{"gaussian": "intentional_extend", "default":
      "dp_relevant"}`` — per-mechanism override, resolved by looking up
      the saved ``mechanism_kind`` (the ``"default"`` key catches
      anything not listed; ``"gaussian"`` is the DP-SGD path).
    """

    version: int
    clip_state: dict[str, Any]
    noise_state: dict[str, Any]
    sampler_state: dict[str, Any] | None

    # --- DP-accounting scalars (privacy-relevant) ---------------------
    sample_rate: float = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    target_delta: float = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    noise_multiplier: float = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    expected_steps_per_epoch: int = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    expected_batch_size: int = field(
        metadata={"compare_on_resume": True, "drift": "dp_relevant"}
    )
    # ``total_steps`` differs by mechanism: DP-SGD users extend training
    # routinely (RDP composes step-by-step); DP-FTRL builds an MF
    # strategy for a specific T, so extending means a different strategy
    # and a different ε.
    total_steps: int = field(
        metadata={
            "compare_on_resume": True,
            "drift": {
                "gaussian": "intentional_extend",
                "default": "dp_relevant",
            },
        }
    )

    # --- DP-FTRL provenance + MF strategy params ---------------------
    mechanism_kind: str = field(
        default="gaussian",
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    participation_plan: dict[str, Any] | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    # Global trainer step at which this sampler stream began. Usually zero;
    # non-zero only after ``ignore_data_skip=True`` intentionally starts a new
    # Poisson stream. ``sampler_state['consumed']`` is stored relative to this
    # origin so DataLoader worker prefetch cannot move the persisted cursor past
    # the number of optimizer steps that actually executed.
    sampler_cursor_origin: int = 0
    is_horizon_process: bool = field(
        default=False,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    calibration_source: str = "fixed"
    target_epsilon: float | None = None
    horizon_process_state: dict[str, Any] | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    mf_n_steps: int | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    mf_min_sep: int | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )
    mf_max_participations: int | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "dp_relevant"},
    )

    # --- LR-schedule shape (trajectory-relevant, privacy-neutral) ---
    lr_scheduler: str | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )
    learning_rate: float | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )
    warmup_steps: int | float | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )
    lr_scheduler_kwargs: dict[str, Any] | None = field(
        default=None,
        metadata={"compare_on_resume": True, "drift": "shape"},
    )


def save_dp_runtime_state(  # noqa: PLR0913
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
    mechanism_kind: str = "gaussian",
    participation_plan: dict[str, Any] | None = None,
    sampler_cursor_origin: int = 0,
    is_horizon_process: bool = False,
    calibration_source: str = "fixed",
    target_epsilon: float | None = None,
    horizon_process_state: dict[str, Any] | None = None,
    mf_n_steps: int | None = None,
    mf_min_sep: int | None = None,
    mf_max_participations: int | None = None,
    lr_scheduler: str | None = None,
    learning_rate: float | None = None,
    warmup_steps: int | float | None = None,
    lr_scheduler_kwargs: dict[str, Any] | None = None,
) -> None:
    """Save the DP runtime bundle as a :class:`RuntimeCheckpoint`."""
    if not isinstance(clip_state, ClipState):
        raise CheckpointError(
            *(
                f"clip_state must be a ClipState instance, got {type(clip_state).__name__}",
            )
        )
    if not isinstance(noise_state, NoiseState):
        raise CheckpointError(
            *(
                f"noise_state must be a NoiseState instance, got {type(noise_state).__name__}",
            )
        )
    if type(sampler_cursor_origin) is not int or sampler_cursor_origin < 0:
        raise CheckpointError(
            *(
                "sampler_cursor_origin must be a non-negative integer, got "
                f"{sampler_cursor_origin!r}.",
            )
        )
    bundle = RuntimeCheckpoint(
        version=DP_STATE_BUNDLE_VERSION,
        clip_state=opaque_state_dict(clip_state),
        noise_state=opaque_state_dict(noise_state),
        sampler_state=sampler_state,
        sample_rate=float(sample_rate),
        target_delta=float(target_delta),
        noise_multiplier=float(noise_multiplier),
        expected_steps_per_epoch=int(expected_steps_per_epoch),
        expected_batch_size=int(expected_batch_size),
        total_steps=int(total_steps),
        mechanism_kind=str(mechanism_kind),
        participation_plan=participation_plan,
        sampler_cursor_origin=sampler_cursor_origin,
        is_horizon_process=bool(is_horizon_process),
        calibration_source=str(calibration_source),
        target_epsilon=(float(target_epsilon) if target_epsilon is not None else None),
        horizon_process_state=horizon_process_state,
        mf_n_steps=int(mf_n_steps) if mf_n_steps is not None else None,
        mf_min_sep=int(mf_min_sep) if mf_min_sep is not None else None,
        mf_max_participations=(
            int(mf_max_participations) if mf_max_participations is not None else None
        ),
        lr_scheduler=lr_scheduler,
        learning_rate=(float(learning_rate) if learning_rate is not None else None),
        warmup_steps=(float(warmup_steps) if warmup_steps is not None else None),
        lr_scheduler_kwargs=lr_scheduler_kwargs,
    )
    if participation_plan is not None:
        validate_dp_runtime_for_resume(bundle)
    # ``torch.save`` of a dataclass round-trips via pickle.  Kept as
    # pickle to handle the heterogeneous types (tensors inside
    # ``clip_state`` / ``noise_state``, Python objects in
    # ``sampler_state``).  A future migration to safetensors+JSON
    # sidecar would eliminate the ``weights_only=False`` requirement.
    torch.save(bundle, path)


def load_dp_runtime_state(path: str) -> RuntimeCheckpoint:
    """Load the DP runtime bundle from disk as a typed :class:`RuntimeCheckpoint`.

    ``clip_state`` and ``noise_state`` are flat serialisation dicts; merge them
    into live objects with :func:`opaque.serialization.from_state_dict` using
    the templates produced by the current training setup.

    ``weights_only=False`` is required for PyTorch 2.6+ defaults when tensors
    or dataclasses appear in the bundle.
    """
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, RuntimeCheckpoint):
        raise CheckpointError(
            *(
                f"dp_state.pt at {path} did not deserialize to RuntimeCheckpoint "
                f"(got {type(bundle).__name__}); checkpoint may be from an older "
                "trainer version.",
            )
        )
    if bundle.version != DP_STATE_BUNDLE_VERSION:
        raise CheckpointError(
            *(
                f"unsupported dp_state bundle version {bundle.version} "
                f"(expected {DP_STATE_BUNDLE_VERSION})",
            )
        )
    return bundle


_POISSON_SAMPLER_FIELDS = frozenset(
    {
        "key_seed",
        "key_impl",
        "consumed",
        "num_samples",
        "sample_rate",
        "n_steps",
        "truncated_batch_size",
    }
)
_K_OUT_OF_T_SAMPLER_FIELDS = frozenset(
    {
        "key_seed",
        "key_impl",
        "consumed",
        "num_samples",
        "k",
        "t",
        "allocation",
    }
)
_B_MIN_SEP_SAMPLER_FIELDS = frozenset(
    {
        "key_seed",
        "key_impl",
        "consumed",
        "num_examples",
        "bands",
        "sampling_prob",
        "n_steps",
    }
)
_BALLS_IN_BINS_SAMPLER_FIELDS = frozenset(
    {
        "key_seed",
        "key_impl",
        "consumed",
        "num_samples",
        "num_bins",
        "n_steps",
    }
)
_SAMPLER_KIND_BY_FIELDS = {
    _POISSON_SAMPLER_FIELDS: "whole_dataset_poisson",
    _K_OUT_OF_T_SAMPLER_FIELDS: "k_out_of_t",
    _B_MIN_SEP_SAMPLER_FIELDS: "b_min_sep",
    _BALLS_IN_BINS_SAMPLER_FIELDS: "balls_in_bins",
}


def _sampler_state_kind(state: object) -> str | None:
    if not isinstance(state, Mapping):
        return None
    return _SAMPLER_KIND_BY_FIELDS.get(frozenset(state))


def _affected_band_poisson_error() -> CheckpointError:
    return CheckpointError(
        *(
            "Cannot resume an mf_band checkpoint written with plain Poisson "
            "sampling: its sampler and accountant describe different participation "
            "processes. Changing the sampler cannot repair the executed prefix. "
            "Restart from the original public/pre-training initialization with "
            "sampling_mode='b_min_sep', or use privacy_noise_mechanism='mf_identity' "
            "for a new whole-dataset Poisson run; do not treat affected DP-trained "
            "weights as a zero-cost fresh initialization.",
        )
    )


def _validate_sampler_state_against_plan(
    plan: ResolvedParticipationPlan,
    state: Mapping[str, Any],
) -> None:
    kind = _sampler_state_kind(state)
    if kind != plan.contract_id:
        if plan.mechanism_kind == "mf_band" and kind == "whole_dataset_poisson":
            raise _affected_band_poisson_error()
        raise CheckpointError(
            *(
                "checkpoint sampler state does not match its participation plan: "
                f"sampler={kind!r}, plan={plan.contract_id!r}.",
            )
        )

    expected_horizon_field = "t" if kind == "k_out_of_t" else "n_steps"
    if type(state[expected_horizon_field]) is not int:
        raise CheckpointError(*("checkpoint sampler horizon has an invalid type.",))
    if state[expected_horizon_field] != plan.total_steps:
        raise CheckpointError(
            *("checkpoint sampler horizon does not match its participation plan.",)
        )
    _validate_sampler_cursor_range(state, horizon=plan.total_steps)
    if type(state["key_seed"]) is not int or type(state["key_impl"]) is not str:
        raise CheckpointError(*("checkpoint sampler RNG identity has invalid types.",))
    population_field = "num_examples" if kind == "b_min_sep" else "num_samples"
    if type(state[population_field]) is not int:
        raise CheckpointError(*("checkpoint sampler population has an invalid type.",))
    if state[population_field] != plan.local_population_size:
        raise CheckpointError(
            *("checkpoint sampler population does not match its participation plan.",)
        )

    kwargs = plan.sampler_kwargs_dict()
    if kind == "whole_dataset_poisson":
        if type(state["sample_rate"]) is not float:
            raise CheckpointError(
                *("checkpoint Poisson sampler rate has an invalid type.",)
            )
        expected_cap = kwargs.get("truncated_batch_size")
        actual_cap = state["truncated_batch_size"]
        if (actual_cap is not None and type(actual_cap) is not int) or (
            expected_cap is not None and type(actual_cap) is not int
        ):
            raise CheckpointError(
                *("checkpoint Poisson sampler cap has an invalid type.",)
            )
        if (
            state["sample_rate"] != plan.sample_rate
            or state["truncated_batch_size"] != expected_cap
        ):
            raise CheckpointError(
                *(
                    "checkpoint Poisson sampler parameters do not match its "
                    "participation plan.",
                )
            )
    elif kind == "k_out_of_t":
        if type(state["k"]) is not int or type(state["allocation"]) is not str:
            raise CheckpointError(
                *("checkpoint k-out-of-t sampler parameters have invalid types.",)
            )
        if state["k"] != kwargs["k"] or state["allocation"] != kwargs["allocation"]:
            raise CheckpointError(
                *(
                    "checkpoint k-out-of-t sampler parameters do not match its "
                    "participation plan.",
                )
            )
    elif kind == "balls_in_bins":
        if type(state["num_bins"]) is not int:
            raise CheckpointError(
                *("checkpoint Balls-in-Bins num_bins has an invalid type.",)
            )
        if state["num_bins"] != plan.num_bins:
            raise CheckpointError(
                *(
                    "checkpoint Balls-in-Bins sampler parameters do not match its "
                    "participation plan.",
                )
            )


def _validate_bundle_against_plan(
    bundle: RuntimeCheckpoint,
    plan: ResolvedParticipationPlan,
) -> None:
    cursor_origin = _validated_sampler_cursor_origin(bundle)
    if cursor_origin > plan.total_steps:
        raise CheckpointError(
            *(
                "checkpoint sampler_cursor_origin exceeds its participation "
                f"horizon: {cursor_origin} > {plan.total_steps}.",
            )
        )
    if cursor_origin != 0 and plan.contract_id != "whole_dataset_poisson":
        raise CheckpointError(
            *(
                "checkpoint sampler_cursor_origin may be non-zero only for a "
                "whole-dataset Poisson stream.",
            )
        )
    expected_horizon = (
        plan.mechanism_kind != "gaussian" or plan.sampling_mode == "k_out_of_t"
    )
    scalar_pairs = (
        ("sample_rate", bundle.sample_rate, plan.sample_rate, float),
        (
            "expected_batch_size",
            bundle.expected_batch_size,
            plan.expected_batch_size,
            int,
        ),
        ("total_steps", bundle.total_steps, plan.total_steps, int),
        (
            "expected_steps_per_epoch",
            bundle.expected_steps_per_epoch,
            plan.num_bins,
            int,
        ),
        (
            "is_horizon_process",
            bundle.is_horizon_process,
            expected_horizon,
            bool,
        ),
    )
    for field_name, actual, expected, expected_type in scalar_pairs:
        if type(actual) is not expected_type or actual != expected:
            raise CheckpointError(
                *(
                    f"checkpoint {field_name} contradicts its participation plan: "
                    f"{actual!r} != {expected!r}.",
                )
            )

    if plan.mechanism_kind != "gaussian":
        if type(bundle.mf_n_steps) is not int or bundle.mf_n_steps != plan.total_steps:
            raise CheckpointError(
                *("checkpoint MF horizon contradicts its participation plan.",)
            )
    elif bundle.mf_n_steps is not None:
        raise CheckpointError(
            *("checkpoint Gaussian mechanism unexpectedly carries an MF horizon.",)
        )

    state = bundle.sampler_state
    if plan.sampling_mode == "b_min_sep":
        if not isinstance(state, Mapping):
            raise CheckpointError(
                *("checkpoint has a participation plan but no sampler state.",)
            )
        bands = state["bands"]
        sampling_prob = state["sampling_prob"]
        if type(bands) is not int or type(sampling_prob) is not float:
            raise CheckpointError(
                *("checkpoint b-min-separation parameters have invalid types.",)
            )
        if type(bundle.mf_min_sep) is not int or bundle.mf_min_sep != bands:
            raise CheckpointError(
                *(
                    "checkpoint b-min-separation sampler contradicts its MF "
                    "participation parameters.",
                )
            )
        from opaque.api.accounting.dpftrl.amplification._b_min_sep import (
            participation_p_from_per_example_rate,
        )

        try:
            expected_probability = participation_p_from_per_example_rate(
                plan.sample_rate,
                bands,
            )
        except ConfigurationError as exc:
            raise CheckpointError(
                *("checkpoint b-min-separation parameters are invalid.",)
            ) from exc
        if sampling_prob != expected_probability:
            raise CheckpointError(
                *(
                    "checkpoint b-min-separation probability contradicts its "
                    "participation plan.",
                )
            )


def _validate_sampler_cursor_range(
    state: Mapping[str, Any],
    *,
    horizon: int,
) -> int:
    consumed = state.get("consumed")
    if type(consumed) is not int or not 0 <= consumed <= horizon:
        raise CheckpointError(
            *(
                "checkpoint sampler cursor is invalid: "
                f"consumed={consumed!r}, horizon={horizon!r}.",
            )
        )
    return consumed


def _validated_sampler_cursor_origin(bundle: RuntimeCheckpoint) -> int:
    """Return the additive cursor origin, defaulting historical bundles to zero."""
    origin = getattr(bundle, "sampler_cursor_origin", 0)
    if type(origin) is not int or origin < 0:
        raise CheckpointError(
            *(f"checkpoint sampler_cursor_origin is invalid: {origin!r}.",)
        )
    return origin


def _validate_legacy_b_min_sep_bundle(bundle: RuntimeCheckpoint) -> None:
    """Prove every participation invariant available in a plan-less v7 bundle."""
    state = bundle.sampler_state
    if not isinstance(state, Mapping) or _sampler_state_kind(state) != "b_min_sep":
        raise CheckpointError(
            *(
                "Cannot determine the participation process of this legacy mf_band "
                "checkpoint; refusing to guess. Inspect the checkpoint separately "
                "or restart from the original public/pre-training initialization.",
            )
        )

    typed_ints = (
        state["key_seed"],
        state["num_examples"],
        state["bands"],
        state["n_steps"],
        bundle.total_steps,
        bundle.mf_n_steps,
        bundle.mf_min_sep,
        bundle.mf_max_participations,
    )
    if (
        any(type(value) is not int for value in typed_ints)
        or type(state["key_impl"]) is not str
        or type(state["sampling_prob"]) is not float
        or type(bundle.sample_rate) is not float
        or state["num_examples"] < 1
        or state["bands"] < 1
        or state["n_steps"] < 1
        or not math.isfinite(state["sampling_prob"])
        or not 0.0 <= state["sampling_prob"] <= 1.0
        or not math.isfinite(bundle.sample_rate)
        or not 0.0 < bundle.sample_rate <= 1.0
    ):
        raise CheckpointError(
            *("legacy b-min-separation checkpoint has invalid parameters.",)
        )

    _validate_sampler_cursor_range(state, horizon=state["n_steps"])
    bands = state["bands"]
    total_steps = state["n_steps"]
    expected_max_participations = (total_steps + bands - 1) // bands
    if (
        total_steps != bundle.total_steps
        or total_steps != bundle.mf_n_steps
        or bands != bundle.mf_min_sep
        or bundle.mf_max_participations != expected_max_participations
        or bundle.is_horizon_process is not True
    ):
        raise CheckpointError(
            *(
                "legacy b-min-separation sampler contradicts its saved MF "
                "participation parameters.",
            )
        )

    from opaque.api.accounting.dpftrl.amplification._b_min_sep import (
        participation_p_from_per_example_rate,
    )

    try:
        expected_probability = participation_p_from_per_example_rate(
            bundle.sample_rate,
            bands,
        )
    except ConfigurationError as exc:
        raise CheckpointError(
            *("legacy b-min-separation checkpoint has invalid parameters.",)
        ) from exc
    if not math.isclose(
        state["sampling_prob"],
        expected_probability,
        rel_tol=1e-15,
        abs_tol=0.0,
    ):
        raise CheckpointError(
            *(
                "legacy b-min-separation probability contradicts its saved "
                "sample rate and band count.",
            )
        )


def validate_sampler_cursor_for_resume(
    bundle: RuntimeCheckpoint,
    *,
    global_step: int,
    ignore_data_skip: bool,
) -> None:
    """Bind an execution-aligned sampler cursor to ``trainer_state.json``."""
    if type(ignore_data_skip) is not bool:
        raise CheckpointError(
            *(f"ignore_data_skip must be a bool, got {ignore_data_skip!r}.",)
        )
    if type(global_step) is not int or global_step < 0:
        raise CheckpointError(
            *(f"checkpoint trainer global_step is invalid: {global_step!r}.",)
        )
    cursor_origin = _validated_sampler_cursor_origin(bundle)
    if ignore_data_skip:
        return
    state = bundle.sampler_state
    kind = _sampler_state_kind(state)
    if kind is None or not isinstance(state, Mapping):
        raise CheckpointError(
            *("checkpoint has no recognized sampler cursor to resume.",)
        )
    horizon_field = "t" if kind == "k_out_of_t" else "n_steps"
    horizon = state[horizon_field]
    if type(horizon) is not int or horizon < 1:
        raise CheckpointError(*("checkpoint sampler horizon is invalid.",))
    consumed = _validate_sampler_cursor_range(state, horizon=horizon)
    if cursor_origin != 0 and kind != "whole_dataset_poisson":
        raise CheckpointError(
            *(
                "checkpoint sampler_cursor_origin may be non-zero only for a "
                "whole-dataset Poisson stream.",
            )
        )
    if (
        cursor_origin > global_step
        or global_step > horizon
        or cursor_origin + consumed != global_step
    ):
        raise CheckpointError(
            *(
                "checkpoint sampler cursor does not match trainer progress: "
                f"origin={cursor_origin}, consumed={consumed}, "
                f"global_step={global_step}. Refusing to resume a different "
                "participation prefix.",
            )
        )


def validate_dp_runtime_for_resume(bundle: RuntimeCheckpoint) -> None:
    """Reject contradictory or historically unsafe participation provenance.

    Loading remains available for inspection. The Trainer invokes this stricter
    validator only when it intends to continue the recorded execution.
    """
    cursor_origin = _validated_sampler_cursor_origin(bundle)
    saved_plan = getattr(bundle, "participation_plan", None)
    sampler_kind = _sampler_state_kind(bundle.sampler_state)
    if saved_plan is not None:
        if not isinstance(saved_plan, Mapping):
            raise CheckpointError(*("checkpoint participation plan is invalid.",))
        plan = ResolvedParticipationPlan.from_state_dict(saved_plan)
        if plan.mechanism_kind != bundle.mechanism_kind:
            raise CheckpointError(
                *(
                    "checkpoint mechanism_kind contradicts its participation plan: "
                    f"{bundle.mechanism_kind!r} != {plan.mechanism_kind!r}.",
                )
            )
        if not isinstance(bundle.sampler_state, Mapping):
            raise CheckpointError(
                *("checkpoint has a participation plan but no sampler state.",)
            )
        _validate_sampler_state_against_plan(plan, bundle.sampler_state)
        _validate_bundle_against_plan(bundle, plan)
        return

    if cursor_origin != 0:
        raise CheckpointError(
            *(
                "checkpoint has a non-zero sampler_cursor_origin but no "
                "participation plan proving a whole-dataset Poisson stream.",
            )
        )

    if bundle.mechanism_kind != "mf_band":
        return
    if sampler_kind == "whole_dataset_poisson":
        raise _affected_band_poisson_error()
    _validate_legacy_b_min_sep_bundle(bundle)
