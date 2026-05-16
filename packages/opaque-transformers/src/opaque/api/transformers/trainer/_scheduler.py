"""HuggingFace-style LR scheduler dispatch for DPTrainer.

Translates :attr:`TrainingArguments.lr_scheduler_type` plus
``lr_scheduler_kwargs`` into a ``Callable[[int], float]`` built from
:mod:`opaque.scheduling` primitives.

Supported types: ``constant``, ``constant_with_warmup``, ``linear``,
``cosine``, ``polynomial``, ``inverse_sqrt``, ``cosine_with_restarts``,
``cosine_with_min_lr``, ``cosine_warmup_with_min_lr``,
``warmup_stable_decay``, ``reduce_lr_on_plateau``.
"""

from __future__ import annotations

import math
from typing import Any, Callable

from opaque.scheduling import (
    constant_schedule,
    cosine_schedule,
    inverse_sqrt_schedule,
    linear_schedule,
    one_minus_sqrt_schedule,
    polynomial_schedule,
    with_restarts,
    with_warmup,
)


__all__ = [
    "build_lr_schedule",
    "get_warmup_steps",
    "parse_optim_args",
    "ReduceLROnPlateauSchedule",
]


_DEFERRED: set[str] = set()

_WSD_KWARGS = frozenset(
    {
        "num_decay_steps",
        "num_stable_steps",
        "warmup_type",
        "decay_type",
        "min_lr_ratio",
        "num_cycles",
    }
)

_PLATEAU_KWARGS = frozenset(
    {
        "factor",
        "patience",
        "mode",
        "threshold",
        "threshold_mode",
        "cooldown",
        "min_lr",
        "eps",
    }
)

_ALLOWED_KWARGS: dict[str, frozenset[str]] = {
    "constant": frozenset(),
    "constant_with_warmup": frozenset(),
    "linear": frozenset(),
    "cosine": frozenset({"num_cycles"}),
    "polynomial": frozenset({"lr_end", "power"}),
    "inverse_sqrt": frozenset({"timescale"}),
    "cosine_with_restarts": frozenset({"num_cycles"}),
    "cosine_with_min_lr": frozenset({"num_cycles", "min_lr", "min_lr_rate"}),
    "cosine_warmup_with_min_lr": frozenset(
        {"num_cycles", "min_lr", "min_lr_rate", "warmup_lr_rate"}
    ),
    "warmup_stable_decay": _WSD_KWARGS,
    "reduce_lr_on_plateau": _PLATEAU_KWARGS,
}


def get_warmup_steps(
    num_training_steps: int,
    warmup_steps: int,
    warmup_ratio: float,
) -> int:
    """Resolve HF's ``{warmup_steps, warmup_ratio}`` pair to an integer
    step count.

    ``warmup_steps`` wins when ``> 0``; otherwise ``warmup_ratio`` is
    applied to ``num_training_steps``.
    """
    if warmup_steps > 0:
        return int(warmup_steps)
    return math.ceil(num_training_steps * warmup_ratio)


def _validate_kwargs(name: str, kwargs: dict[str, Any]) -> None:
    allowed = _ALLOWED_KWARGS[name]
    extra = set(kwargs) - allowed
    if extra:
        raise ValueError(
            f"lr_scheduler_kwargs contains unsupported keys for "
            f"lr_scheduler_type={name!r}: {sorted(extra)}. "
            f"Allowed: {sorted(allowed) or '<none>'}."
        )


def build_lr_schedule(
    args: Any,
    num_training_steps: int,
) -> Callable[[int], float]:
    """Build a step -> learning-rate callable from HF-style training arguments.

    Reads ``args.lr_scheduler_type``, ``args.warmup_steps``,
    ``args.warmup_ratio``, ``args.learning_rate``, and
    ``args.lr_scheduler_kwargs``.  Returns a callable suitable for
    passing as the ``lr`` argument of any torchopt optimizer factory.
    """
    name = args.lr_scheduler_type
    if hasattr(name, "value"):
        name = name.value
    base_lr = args.learning_rate
    # Tolerate ``None`` for callers that pre-date the
    # ``field(default_factory=dict)`` migration (HF parity:
    # ``TrainingArguments`` accepts JSON-string here too, but we don't —
    # see ``TrainingArguments`` field docstring).
    kwargs = dict(args.lr_scheduler_kwargs or {})

    if name in _DEFERRED:
        raise NotImplementedError(
            f"lr_scheduler_type={name!r} is a recognized HuggingFace scheduler "
            f"that DPTrainer doesn't implement yet. If you need it, please open "
            f"an issue at https://github.com/JetBrains-Research/opaque/issues. "
            f"Currently supported: {sorted(_ALLOWED_KWARGS)}."
        )
    if name not in _ALLOWED_KWARGS:
        raise ValueError(
            f"Unknown lr_scheduler_type={name!r}. Supported: {sorted(_ALLOWED_KWARGS)}."
        )

    _validate_kwargs(name, kwargs)

    W = get_warmup_steps(num_training_steps, args.warmup_steps, args.warmup_ratio)
    decay_steps = max(1, num_training_steps - W)

    if name == "constant":
        return constant_schedule(base_lr)
    if name == "constant_with_warmup":
        return (
            with_warmup(base_lr, transition_steps=W)
            if W > 0
            else constant_schedule(base_lr)
        )
    if name == "warmup_stable_decay":
        return _build_wsd(base_lr, W, num_training_steps, kwargs)
    if name == "reduce_lr_on_plateau":
        return _build_plateau(args, base_lr, kwargs)

    warmup_init = 0.0  # warmup ramp starts at 0 unless overridden below.

    if name == "linear":
        decay = linear_schedule(base_lr, 0.0, decay_steps, transition_begin=W)
    elif name == "cosine":
        decay = cosine_schedule(
            base_lr,
            0.0,
            decay_steps,
            transition_begin=W,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "polynomial":
        lr_end = float(kwargs.get("lr_end", 1e-7))
        if lr_end >= base_lr:
            raise ValueError(
                f"lr_end ({lr_end}) must be smaller than initial lr ({base_lr})"
            )
        decay = polynomial_schedule(
            base_lr,
            lr_end,
            kwargs.get("power", 1.0),
            decay_steps,
            transition_begin=W,
        )
    elif name == "inverse_sqrt":
        timescale = kwargs.get("timescale") or W or 10_000
        decay = inverse_sqrt_schedule(
            base_lr, transition_steps=timescale, transition_begin=W
        )
    elif name == "cosine_with_restarts":
        cycles = int(kwargs.get("num_cycles", 1))
        cycle_length = decay_steps / cycles
        inner = cosine_schedule(base_lr, 0.0, transition_steps=cycle_length)
        decay = with_restarts(
            inner, transition_steps=decay_steps, num_cycles=cycles, transition_begin=W
        )
    elif name == "cosine_with_min_lr":
        end_value = _resolve_min_lr(base_lr, kwargs)
        decay = cosine_schedule(
            base_lr,
            end_value,
            decay_steps,
            transition_begin=W,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
    elif name == "cosine_warmup_with_min_lr":
        # Warmup-ramp divergence vs HF.  HF's
        # ``_get_cosine_with_min_lr_schedule_with_warmup_lr_rate_lambda``
        # (``transformers/optimization.py``) computes the warmup factor
        # as ``r + (1 - r) * step / (W - 1)`` (linear ramp anchored at
        # exact endpoints ``r`` at step 0 and ``1.0`` at step ``W - 1``).
        # We compose ``with_warmup`` instead, whose factor is
        # ``r + (1 - r) * progress`` with ``progress = step / W``.  Both
        # ramps agree at step 0 (``r``) and at the post-warmup boundary
        # (``decay(W) == base_lr``), but the interior values differ
        # whenever ``warmup_lr_rate > 0``.  Strict numerical parity with
        # HF's ``cosine_warmup_with_min_lr`` would require an
        # ``(W - 1)``-anchored ramp primitive that doesn't currently
        # exist in :mod:`opaque.scheduling`; the current behavior is
        # accepted as a documented divergence.
        end_value = _resolve_min_lr(base_lr, kwargs)
        decay = cosine_schedule(
            base_lr,
            end_value,
            decay_steps,
            transition_begin=W,
            num_cycles=kwargs.get("num_cycles", 0.5),
        )
        warmup_init = float(kwargs.get("warmup_lr_rate") or 0.0)
    else:  # pragma: no cover — guarded above.
        raise AssertionError(name)

    if W == 0:
        return decay
    return with_warmup(decay, transition_steps=W, ramp="linear", init_value=warmup_init)


def _resolve_min_lr(base_lr: float, kwargs: dict[str, Any]) -> float:
    """Translate HF's mutually-exclusive ``min_lr`` / ``min_lr_rate`` to an
    absolute end value for our cosine_schedule."""
    min_lr = kwargs.get("min_lr")
    min_lr_rate = kwargs.get("min_lr_rate")
    if min_lr is not None and min_lr_rate is not None:
        raise ValueError("Set only one of min_lr or min_lr_rate.")
    if min_lr is not None:
        return float(min_lr)
    if min_lr_rate is not None:
        return float(min_lr_rate) * base_lr
    raise ValueError(
        "lr_scheduler_kwargs must include exactly one of {'min_lr': ...} "
        "or {'min_lr_rate': ...}."
    )


_WSD_RAMP_TYPES = {"linear", "cosine", "1-sqrt"}
_WSD_DECAY_TYPES = {"linear", "cosine", "1-sqrt"}


def _build_wsd(
    base_lr: float,
    W: int,
    num_training_steps: int,
    kwargs: dict[str, Any],
) -> Callable[[int], float]:
    """Build HF's warmup_stable_decay (WSD) schedule.

    Three phases, each modulated by ``min_lr_ratio`` so the schedule's
    floor is ``min_lr_ratio * base_lr``:

    1. warmup: ramp from floor to ``base_lr`` over ``W`` steps, shape
       chosen by ``warmup_type``.
    2. stable: ``base_lr`` for ``num_stable_steps``.
    3. decay: ``base_lr`` → floor over ``num_decay_steps``, shape chosen
       by ``decay_type``.  After decay, holds at floor.

    ``warmup_type`` and ``decay_type`` accept ``"linear"``, ``"cosine"``,
    or ``"1-sqrt"``.
    """
    warmup_type = kwargs.get("warmup_type", "linear")
    if warmup_type not in _WSD_RAMP_TYPES:
        raise ValueError(
            f"warmup_stable_decay warmup_type must be one of "
            f"{sorted(_WSD_RAMP_TYPES)}; got {warmup_type!r}."
        )
    decay_type = kwargs.get("decay_type", "cosine")
    if decay_type not in _WSD_DECAY_TYPES:
        raise ValueError(
            f"warmup_stable_decay decay_type must be one of "
            f"{sorted(_WSD_DECAY_TYPES)}; got {decay_type!r}."
        )

    if "num_decay_steps" not in kwargs:
        raise ValueError(
            "warmup_stable_decay requires lr_scheduler_kwargs={'num_decay_steps': ...}."
        )
    D = int(kwargs["num_decay_steps"])
    S = int(kwargs.get("num_stable_steps", num_training_steps - W - D))
    min_lr_ratio = float(kwargs.get("min_lr_ratio", 0.0))
    num_cycles = float(kwargs.get("num_cycles", 0.5))

    floor = min_lr_ratio * base_lr

    # Decay curve: base_lr at step (W+S), floor at step (W+S+D), held at floor after.
    if decay_type == "linear":
        decay = linear_schedule(
            base_lr, floor, transition_steps=D, transition_begin=W + S
        )
    elif decay_type == "cosine":
        decay = cosine_schedule(
            base_lr,
            floor,
            transition_steps=D,
            transition_begin=W + S,
            num_cycles=num_cycles,
        )
    else:  # "1-sqrt"
        decay = one_minus_sqrt_schedule(
            base_lr,
            floor,
            transition_steps=D,
            transition_begin=W + S,
        )

    # Cosine post-decay continues oscillating; linear/1-sqrt clip naturally.
    # Clamp at floor for the trailing region.
    end_of_decay = W + S + D

    def core(step: int) -> float:
        if step >= end_of_decay:
            return floor
        return decay(step)

    if W <= 0:
        return core

    return with_warmup(
        core,
        transition_steps=W,
        ramp=warmup_type,
        init_value=min_lr_ratio,
    )


# ---------------------------------------------------------------------------
# ReduceLROnPlateau schedule (metric-driven, not step-indexed).
# ---------------------------------------------------------------------------


_PLATEAU_MODES = {"min", "max"}
_PLATEAU_THRESHOLD_MODES = {"rel", "abs"}


class ReduceLROnPlateauSchedule:
    """Metric-driven LR schedule mirroring ``torch.optim.lr_scheduler.ReduceLROnPlateau``.

    Behaves as a ``Callable[[int], float]`` so it slots into the same
    ``scale_by_schedule`` plumbing as the step-indexed schedules; the
    step argument is ignored — the LR is updated only via :meth:`update`,
    which DPTrainer calls after each evaluation with the value of
    ``state.best_metric`` (or, equivalently, the latest eval metric).

    Use ``mode="min"`` for losses (default) and ``mode="max"`` for
    accuracy-style metrics.  ``factor`` (``< 1``) multiplies the current
    LR each time ``patience`` consecutive bad epochs accumulate.  ``min_lr``
    is the absolute floor of the LR.
    """

    def __init__(
        self,
        base_lr: float,
        *,
        factor: float = 0.1,
        patience: int = 10,
        mode: str = "min",
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ) -> None:
        if mode not in _PLATEAU_MODES:
            raise ValueError(
                f"mode must be one of {sorted(_PLATEAU_MODES)}; got {mode!r}"
            )
        if threshold_mode not in _PLATEAU_THRESHOLD_MODES:
            raise ValueError(
                f"threshold_mode must be one of {sorted(_PLATEAU_THRESHOLD_MODES)}; "
                f"got {threshold_mode!r}"
            )
        if not (0.0 < factor < 1.0):
            raise ValueError(f"factor must be in (0, 1); got {factor}")
        if patience < 0:
            raise ValueError(f"patience must be >= 0; got {patience}")

        self.base_lr = float(base_lr)
        self.factor = float(factor)
        self.patience = int(patience)
        self.mode = mode
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.cooldown = int(cooldown)
        self.min_lr = float(min_lr)
        self.eps = float(eps)

        self._lr: float = float(base_lr)
        self._best: float | None = None
        self._num_bad: int = 0
        self._cooldown_counter: int = 0

    # --- callable interface ------------------------------------------------

    def __call__(self, step: int) -> float:  # noqa: ARG002 — step ignored
        return self._lr

    # --- metric updates ----------------------------------------------------

    def _is_better(self, current: float, best: float) -> bool:
        if self.mode == "min":
            if self.threshold_mode == "rel":
                return current < best * (1.0 - self.threshold)
            return current < best - self.threshold
        # mode == "max"
        if self.threshold_mode == "rel":
            return current > best * (1.0 + self.threshold)
        return current > best + self.threshold

    def update(self, metric: float | None) -> None:
        """Feed a fresh eval metric value; may reduce the LR."""
        if metric is None:
            return
        metric = float(metric)
        if self._best is None or self._is_better(metric, self._best):
            self._best = metric
            self._num_bad = 0
        else:
            self._num_bad += 1

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            self._num_bad = 0  # cooldown swallows bad-epoch counter (HF parity)

        if self._num_bad > self.patience:
            new_lr = max(self._lr * self.factor, self.min_lr)
            if self._lr - new_lr > self.eps:
                self._lr = new_lr
            self._cooldown_counter = self.cooldown
            self._num_bad = 0

    # --- (de)serialization -------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_lr": self.base_lr,
            "factor": self.factor,
            "patience": self.patience,
            "mode": self.mode,
            "threshold": self.threshold,
            "threshold_mode": self.threshold_mode,
            "cooldown": self.cooldown,
            "min_lr": self.min_lr,
            "eps": self.eps,
            "_lr": self._lr,
            "_best": self._best,
            "_num_bad": self._num_bad,
            "_cooldown_counter": self._cooldown_counter,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        # Configuration stays as-constructed; only mutable state is restored
        # (HF / torch convention).  Mismatched configuration is the caller's
        # responsibility — the drift-detection helper in _checkpoint.py
        # surfaces it loudly.
        if "_lr" in sd:
            self._lr = float(sd["_lr"])
        if "_best" in sd:
            self._best = None if sd["_best"] is None else float(sd["_best"])
        if "_num_bad" in sd:
            self._num_bad = int(sd["_num_bad"])
        if "_cooldown_counter" in sd:
            self._cooldown_counter = int(sd["_cooldown_counter"])


def _build_plateau(
    args: Any,
    base_lr: float,
    kwargs: dict[str, Any],
) -> ReduceLROnPlateauSchedule:
    """Construct a :class:`ReduceLROnPlateauSchedule` from HF-style args.

    ``mode`` defaults from ``metric_for_best_model`` when unset:
    ``"min"`` for ``*loss*`` metrics, ``"max"`` otherwise.

    ``TrainingArguments.__post_init__`` defaults
    ``metric_for_best_model="loss"`` whenever ``lr_scheduler_type ==
    "reduce_lr_on_plateau"``, so this builder always sees a
    non-``None`` metric name.

    DP caveat: the schedule's metric input must come from a
    held-out / public eval set.  Feeding train-data eval metrics into
    the LR schedule makes the LR trajectory data-dependent in a way
    the privacy accountant doesn't track.
    """
    metric = args.metric_for_best_model or "loss"
    mode = kwargs.get("mode")
    if mode is None:
        # HF convention: minimize loss, maximize anything else.
        mode = "min" if metric.endswith("loss") else "max"

    return ReduceLROnPlateauSchedule(
        base_lr=base_lr,
        factor=float(kwargs.get("factor", 0.1)),
        patience=int(kwargs.get("patience", 10)),
        mode=mode,
        threshold=float(kwargs.get("threshold", 1e-4)),
        threshold_mode=str(kwargs.get("threshold_mode", "rel")),
        cooldown=int(kwargs.get("cooldown", 0)),
        min_lr=float(kwargs.get("min_lr", 0.0)),
        eps=float(kwargs.get("eps", 1e-8)),
    )


# ---------------------------------------------------------------------------
# optim_args parsing (HF parity).
# ---------------------------------------------------------------------------


def _coerce_value(raw: str) -> Any:
    """Best-effort literal coercion of a user-supplied option string."""
    s = raw.strip()
    lowered = s.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def parse_optim_args(spec: str | None) -> dict[str, Any]:
    """Parse HF's ``optim_args`` string into a kwargs ``dict``.

    Format: ``"key1=value1,key2=value2"``.  Whitespace around keys and
    values is stripped.  Values are coerced to ``bool`` / ``int`` /
    ``float`` when possible, otherwise kept as ``str``.

    Empty / ``None`` input returns ``{}``.

    Raises:
        ValueError: malformed entry (missing ``=`` or empty key).
    """
    if not spec:
        return {}
    out: dict[str, Any] = {}
    for entry in spec.split(","):
        if not entry.strip():
            continue
        if "=" not in entry:
            raise ValueError(f"optim_args entry {entry!r} is not in 'key=value' form")
        key, _, value = entry.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"optim_args entry {entry!r} has an empty key")
        out[key] = _coerce_value(value)
    return out
