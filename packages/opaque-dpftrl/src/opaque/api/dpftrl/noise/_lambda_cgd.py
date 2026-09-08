"""DP-lambda-CGD strategy and noise — correlated noise via PRNG replay.

The DP-lambda-CGD mechanism (Kalinin et al., 2026) uses a lower-triangular
Toeplitz strategy matrix :math:`C_\\lambda` whose inverse is bidiagonal: 1 on the
diagonal, :math:`-\\lambda` on the subdiagonal.  The correlated noise at step t is::

    n_t = z_t - lambda * z_{t-1}              (unnormalized)
    n_t = d_t * (z_t - lambda * z_{t-1})      (column-normalized, default)

where :math:`z_t \\sim N(0, \\sigma^2 I)` are i.i.d. Gaussians, and :math:`d_t`
is the column norm of :math:`C_\\lambda` at step t.  Instead of storing
:math:`z_{t-1}`, we regenerate it from the previous step's PRNG seed —
without retaining a noise-history buffer.

References:
    - Kalinin et al. (2026) "DP-λCGD: Efficient Noise Correlation for
      Differentially Private Model Training" https://arxiv.org/abs/2601.22334
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from numbers import Real
from typing import TYPE_CHECKING, Any

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy
from opaque.exceptions import CheckpointError, ConfigurationError, InputTypeError
from opaque.pytree import tree_flatten_with_paths, tree_map
from opaque.random import fold_in as rng_fold_in
from opaque.random import generator_from_key
from opaque.random.types import RngKey
from opaque.serialization import register_serializer
from opaque.types import PerGroup

from ._engine import (
    MFNoiseState,
    _check_mf_horizon,
    _iid_normal_noise,
    _require_positive_int_horizon,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from opaque.api.engine.scheduling.types import Schedule

    from ._streaming_matrix import StreamingMatrix


def _native():
    from opaque.api.accounting.core import _native as _n

    return _n


_COLUMN_NORM_TAIL_CUTOFF = 1e-30
LAMBDA_CGD_STREAM_FOLD = "opaque.dpftrl.lambda_cgd"
# Bump whenever key derivation, generator replay, layout traversal, or scale
# encoding changes. A mixed-version prefix and suffix is not one calibrated
# correlated mechanism.
_LAMBDA_CGD_REPLAY_VERSION = 1
_INCOMPATIBLE_REPLAY_IDENTITY = (
    "Lambda-CGD state uses an incompatible RNG topology or execution identity "
    "and cannot be resumed. Restart with the configuration and Opaque version "
    "that created it, or restart with the current version from the original "
    "public or pre-training initialization; do not treat already DP-trained "
    "weights as a zero-cost fresh initialization."
)
_SHA256_HEX_LENGTH = 64


@dataclass(frozen=True, slots=True)
class _LambdaCgdExecutionIdentity:
    """Immutable inputs to a λ-CGD replay sequence."""

    replay_version: int
    stream_root: str
    lambda_: float
    normalized: bool
    n_steps: int
    compute_dtype: str
    rng_seed: int
    rng_impl: str
    grad_layout_digest: str


@dataclass(frozen=True, slots=True)
class _LambdaCgdReplayState:
    """State required to regenerate the preceding IID draw."""

    step: int
    base_stddev_digest: str | None
    execution_identity: _LambdaCgdExecutionIdentity


_IDENTITY_FIELDS = (
    "replay_version",
    "stream_root",
    "lambda_",
    "normalized",
    "n_steps",
    "compute_dtype",
    "rng_seed",
    "rng_impl",
    "grad_layout_digest",
)
_RNG_IDENTITY_FIELDS = {"rng_seed", "rng_impl"}
_CONFIGURATION_IDENTITY_FIELDS = tuple(
    name for name in _IDENTITY_FIELDS if name not in _RNG_IDENTITY_FIELDS
)
_REPLAY_STATE_FIELDS = {
    "step",
    "base_stddev_digest",
    *(f"execution_identity.{name}" for name in _IDENTITY_FIELDS),
}


def _is_sha256_digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _path_part_payload(part: object) -> list[str | int]:
    if type(part) is int:
        return ["int", part]
    if type(part) is str:
        return ["str", part]
    raise InputTypeError(
        *(f"Lambda-CGD gradient path contains {type(part).__name__}.",)
    )


def _path_payload(path: tuple[str | int, ...]) -> list[list[str | int]]:
    return [_path_part_payload(part) for part in path]


def _treespec_payload(treespec: Any) -> dict[str, Any]:
    node_type = treespec.type
    return {
        "type": (
            None
            if node_type is None
            else [node_type.__module__, node_type.__qualname__]
        ),
        "namespace": treespec.namespace,
        "none_is_leaf": treespec.none_is_leaf,
        "entries": [_path_part_payload(entry) for entry in treespec.entries()],
        "children": [_treespec_payload(child) for child in treespec.children()],
    }


def _gradient_layout(
    tree: Any,
) -> tuple[
    Any,
    tuple[tuple[str | int, ...], ...],
    tuple[tuple[tuple[int, ...], str], ...],
]:
    paths, leaves, treedef = tree_flatten_with_paths(tree)
    leaf_layouts: list[tuple[tuple[int, ...], str]] = []
    for path, leaf in zip(paths, leaves, strict=True):
        if not isinstance(leaf, torch.Tensor):
            raise InputTypeError(
                *(
                    "Lambda-CGD noise expects tensor leaves; "
                    f"got {type(leaf).__name__} at path {path!r}.",
                )
            )
        leaf_layouts.append((tuple(leaf.shape), str(leaf.dtype)))
    return treedef, tuple(paths), tuple(leaf_layouts)


def _gradient_layout_digest(
    treedef: Any,
    paths: tuple[tuple[str | int, ...], ...],
    leaf_layouts: tuple[tuple[tuple[int, ...], str], ...],
) -> str:
    leaves = [
        {"path": _path_payload(path), "shape": list(shape), "dtype": dtype}
        for path, (shape, dtype) in zip(paths, leaf_layouts, strict=True)
    ]
    return _digest_payload({"treespec": _treespec_payload(treedef), "leaves": leaves})


def _float_payload(value: object, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputTypeError(*(f"{label} must be a real number.",))
    return float(value).hex()


def _base_stddev_digest(
    stddev: float | PerGroup,
    paths: tuple[tuple[str | int, ...], ...],
) -> str:
    if isinstance(stddev, PerGroup):
        payload: object = {
            "kind": "per_leaf",
            "values": [
                {
                    "path": _path_payload(path),
                    "value": _float_payload(
                        stddev.for_path(path),
                        label=f"Lambda-CGD stddev at path {path!r}",
                    ),
                }
                for path in paths
            ],
        }
    else:
        payload = {
            "kind": "scalar",
            "value": _float_payload(stddev, label="Lambda-CGD stddev"),
        }
    return _digest_payload(payload)


def _validate_execution_identity(value: object) -> None:
    if (
        type(value) is not _LambdaCgdExecutionIdentity
        or type(value.replay_version) is not int
        or value.replay_version != _LAMBDA_CGD_REPLAY_VERSION
        or type(value.stream_root) is not str
        or value.stream_root != LAMBDA_CGD_STREAM_FOLD
        or type(value.lambda_) is not float
        or not math.isfinite(value.lambda_)
        or not 0.0 <= value.lambda_ < 1.0
        or type(value.normalized) is not bool
        or type(value.n_steps) is not int
        or value.n_steps < 1
        or type(value.compute_dtype) is not str
        or type(value.rng_seed) is not int
        or type(value.rng_impl) is not str
        or not _is_sha256_digest(value.grad_layout_digest)
    ):
        raise CheckpointError(*(_INCOMPATIBLE_REPLAY_IDENTITY,))


def _validate_replay_state(
    value: object,
    *,
    expected_identity: _LambdaCgdExecutionIdentity | None = None,
) -> _LambdaCgdReplayState:
    if type(value) is not _LambdaCgdReplayState:
        raise CheckpointError(*(_INCOMPATIBLE_REPLAY_IDENTITY,))
    _validate_execution_identity(value.execution_identity)
    if expected_identity is not None:
        mismatched = [
            name
            for name in _CONFIGURATION_IDENTITY_FIELDS
            if getattr(value.execution_identity, name)
            != getattr(expected_identity, name)
        ]
        if mismatched:
            raise CheckpointError(*(_INCOMPATIBLE_REPLAY_IDENTITY,))
    if (
        type(value.step) is not int
        or value.step < 0
        or value.step > value.execution_identity.n_steps
        or (
            value.base_stddev_digest is not None
            and not _is_sha256_digest(value.base_stddev_digest)
        )
        or (value.step == 0) != (value.base_stddev_digest is None)
    ):
        raise CheckpointError(
            *("Lambda-CGD replay state has an invalid step or noise-scale latch.",)
        )
    return value


def _save_lambda_cgd_replay_state(value: _LambdaCgdReplayState) -> dict[str, Any]:
    replay = _validate_replay_state(value)
    identity = replay.execution_identity
    return {
        "step": replay.step,
        "base_stddev_digest": replay.base_stddev_digest,
        **{
            f"execution_identity.{name}": getattr(identity, name)
            for name in _IDENTITY_FIELDS
        },
    }


def _load_lambda_cgd_replay_state(
    template: _LambdaCgdReplayState,
    saved: Mapping[str, Any],
) -> _LambdaCgdReplayState:
    expected = _validate_replay_state(template).execution_identity
    if set(saved) != _REPLAY_STATE_FIELDS:
        raise CheckpointError(*(_INCOMPATIBLE_REPLAY_IDENTITY,))
    restored_identity = _LambdaCgdExecutionIdentity(
        **{name: saved[f"execution_identity.{name}"] for name in _IDENTITY_FIELDS}
    )
    _validate_execution_identity(restored_identity)
    mismatched = [
        name
        for name in _CONFIGURATION_IDENTITY_FIELDS
        if type(saved[f"execution_identity.{name}"])
        is not type(getattr(expected, name))
        or saved[f"execution_identity.{name}"] != getattr(expected, name)
    ]
    if mismatched:
        raise CheckpointError(
            *(
                "Lambda-CGD checkpoint execution identity does not match the "
                f"configured runtime; mismatched fields={mismatched}.",
            )
        )
    restored = _LambdaCgdReplayState(
        step=saved["step"],
        base_stddev_digest=saved["base_stddev_digest"],
        execution_identity=restored_identity,
    )
    return _validate_replay_state(restored, expected_identity=expected)


register_serializer(
    _LambdaCgdReplayState,
    _save_lambda_cgd_replay_state,
    _load_lambda_cgd_replay_state,
)


def _lambda_cgd_replay_sync_token(value: object) -> str | None:
    if type(value) is not _LambdaCgdReplayState:
        return None
    try:
        payload = _save_lambda_cgd_replay_state(value)
    except CheckpointError:
        return f"lambda_cgd:invalid:{value!r}"
    return f"lambda_cgd:{_digest_payload(payload)}"


def _validate_lambda_cgd_noise_state(state: MFNoiseState) -> _LambdaCgdReplayState:
    """Validate the relationship between the outer and replay state."""
    replay = _validate_replay_state(state._inner_state)
    if type(state._step_counter) is not int or state._step_counter != replay.step:
        raise CheckpointError(
            *("Lambda-CGD inner and outer replay steps do not match.",)
        )
    if not isinstance(state._rng_key, RngKey):
        raise CheckpointError(*("Lambda-CGD replay state has an invalid RNG key.",))
    if state._rng_key.seed != replay.execution_identity.rng_seed:
        raise CheckpointError(
            *("Lambda-CGD replay state has an incompatible RNG seed.",)
        )
    if state._rng_key.impl != replay.execution_identity.rng_impl:
        raise CheckpointError(
            *("Lambda-CGD replay state has an incompatible RNG implementation.",)
        )
    return replay


def _require_lambda_cgd_replay_state(
    state: MFNoiseState,
    expected_state: MFNoiseState,
) -> _LambdaCgdReplayState:
    expected = expected_state._inner_state
    if type(expected) is not _LambdaCgdReplayState:
        raise CheckpointError(*(_INCOMPATIBLE_REPLAY_IDENTITY,))
    replay = _validate_lambda_cgd_noise_state(state)
    return _validate_replay_state(
        replay,
        expected_identity=expected.execution_identity,
    )


def _validate_lambda_cgd_replay_call(
    clipped_grads: Any,
    state: MFNoiseState,
    expected_state: MFNoiseState,
    *,
    stddev: float | PerGroup,
) -> tuple[_LambdaCgdReplayState, str]:
    """Validate every replay-dependent input before constructing a generator."""
    replay = _require_lambda_cgd_replay_state(state, expected_state)
    _check_mf_horizon(state._step_counter, replay.execution_identity.n_steps)
    treedef, paths, leaf_layouts = _gradient_layout(clipped_grads)
    if (
        _gradient_layout_digest(treedef, paths, leaf_layouts)
        != replay.execution_identity.grad_layout_digest
    ):
        raise InputTypeError(
            *(
                "Lambda-CGD gradients must match the construction template's "
                "pytree structure, shapes, and dtypes.",
            )
        )
    scale_digest = _base_stddev_digest(stddev, paths)
    if (
        replay.base_stddev_digest is not None
        and replay.base_stddev_digest != scale_digest
    ):
        raise CheckpointError(
            *(
                "Lambda-CGD base noise scale changed after replay began; "
                "restart with the original noise and clipping configuration.",
            )
        )
    return replay, scale_digest


@lru_cache(maxsize=256)
def _lambda_cgd_gram_matrix_cached(
    lambda_: float,
    normalized: bool,
    n_steps: int,
    min_sep: int,
    max_participations: int | None,
) -> tuple[float, ...]:
    """Gram sequence for λ-CGD; cached across repeated σ / PLD probes."""
    return tuple(
        _native().lambda_cgd_gram_matrix(
            lambda_, n_steps, min_sep, max_participations, normalized
        )
    )


def _column_norm(lambda_: float, n_steps: int, step: int) -> float:
    """Column norm :math:`d_t` of :math:`C_\\lambda` at 0-indexed step t.

    ``step`` must be in ``[0, n_steps)``.  At ``step == n_steps`` the
    closed form collapses to 0 (and beyond it is undefined), which would
    zero out the released noise under ``normalized=True``.
    """
    if step < 0 or step >= n_steps:
        raise ConfigurationError(
            *(
                f"column-norm step {step} is outside the calibrated horizon [0, {n_steps}).",
            )
        )
    if lambda_ == 0.0:
        return 1.0
    remaining = n_steps - step
    lambda2 = lambda_ * lambda_
    lambda2r = lambda2**remaining
    if lambda2r < _COLUMN_NORM_TAIL_CUTOFF:
        return math.sqrt(1.0 / (1.0 - lambda2))
    return math.sqrt((1.0 - lambda2r) / (1.0 - lambda2))


@register_strategy
@dataclass(frozen=True, slots=True)
class LambdaCgdStrategy:
    """DP-lambda-CGD strategy — recipe only (PRNG-replay noise)."""

    lambda_: float
    normalized: bool = True
    # Compatibility tombstone for legacy state dictionaries. Non-None values
    # are rejected because optimizer LR schedules are not part of this encoder.
    lr_schedule: Schedule | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.lr_schedule is not None:
            raise ConfigurationError(
                *(
                    "LambdaCgdStrategy does not support lr_schedule. Learning-rate "
                    "schedules are optimizer post-processing and cannot weight its "
                    "Balls-in-Bins privacy accounting. Remove lr_schedule from the "
                    "strategy, pass it only to the optimizer, and recalibrate privacy "
                    "and noise for any result previously computed with this option.",
                )
            )
        if not math.isfinite(self.lambda_) or not 0.0 <= self.lambda_ < 1.0:
            raise ConfigurationError(
                *(f"lambda_ must be finite and in [0, 1), got {self.lambda_}",)
            )

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        # [1, λ, λ², ..., λ^{n_steps-1}].
        return torch.tensor(
            [self.lambda_**i for i in range(n_steps)], dtype=torch.float64
        )

    def gram_matrix(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> tuple[float, ...]:
        return _lambda_cgd_gram_matrix_cached(
            self.lambda_,
            self.normalized,
            n_steps,
            min_sep,
            max_participations,
        )

    def streaming_matrix(self, **_) -> StreamingMatrix:
        # Lambda-CGD never materializes a streaming matrix — it uses
        # PRNG replay via :func:`_make_lambda_cgd_noise` instead.  The
        # mf_gaussian_noise dispatcher special-cases this strategy.
        raise NotImplementedError(
            "LambdaCgdStrategy uses PRNG-replay noise; the noise factory "
            "dispatches to _make_lambda_cgd_noise directly."
        )

    def raw_noise_factory(
        self,
        grad_template: Any,
        *,
        n_steps: int,
        min_sep: int,
        max_participations: int | None,
        key: RngKey,
        compute_dtype: torch.dtype,
    ):
        del min_sep, max_participations
        return _make_lambda_cgd_noise(
            grad_template,
            self,
            n_steps=n_steps,
            key=key,
            compute_dtype=compute_dtype,
        )

    @staticmethod
    def validate_raw_noise_state(
        state: MFNoiseState,
        expected_state: MFNoiseState,
    ) -> None:
        _require_lambda_cgd_replay_state(state, expected_state)

    @staticmethod
    def validate_raw_noise_call(
        clipped_grads: Any,
        state: MFNoiseState,
        expected_state: MFNoiseState,
        *,
        stddev: float | PerGroup,
    ) -> None:
        _validate_lambda_cgd_replay_call(
            clipped_grads,
            state,
            expected_state,
            stddev=stddev,
        )

    def sensitivity(
        self, *, n_steps: int, min_sep: int, max_participations: int | None
    ) -> float:
        if self.normalized:
            sens_sq = _native().lambda_cgd_normalized_sensitivity_squared(
                self.lambda_, n_steps, min_sep, max_participations
            )
        else:
            sens_sq = _native().lambda_cgd_sensitivity_squared(
                self.lambda_, n_steps, min_sep, max_participations
            )
        return float(sens_sq**0.5)

    def max_column_norm(self, *, n_steps: int) -> float:
        """Max L2 column norm of the strategy matrix at this horizon."""
        if self.normalized:
            return 1.0
        return float(_native().lambda_cgd_max_column_norm(self.lambda_, n_steps))


def lambda_cgd_strategy(
    *,
    lambda_: float,
    normalized: bool = True,
    lr_schedule: Schedule | None = None,
) -> LambdaCgdStrategy:
    """Create a DP-lambda-CGD strategy recipe (bandwidth=2, PRNG-replay noise).

    Args:
        lambda_: Correlation coefficient in [0, 1).
        normalized: Use column-normalized matrix (default True).
        lr_schedule: Deprecated compatibility argument. Only ``None`` is
            accepted. Pass learning-rate schedules to the optimizer instead.

    Returns:
        A :class:`LambdaCgdStrategy` recipe.
    """
    return LambdaCgdStrategy(
        lambda_=lambda_,
        normalized=normalized,
        lr_schedule=lr_schedule,
    )


# ---------------------------------------------------------------------------
# Internal noise builder (called by mf_gaussian_noise() dispatcher)
# ---------------------------------------------------------------------------


def _lambda_cgd_row_l2(strategy: LambdaCgdStrategy, n_steps: int, step: int) -> float:
    """Per-step row L2 norm of the λ-CGD effective C^{-1}.

    The realized per-coordinate noise at step ``t`` is
    ``base_σ · row_l2(t)`` (post-normalization when ``normalized=True``).
    At step 0 there is no previous-step term so the factor is just the
    optional column-norm multiplier; at step t≥1 the unnormalized factor
    is ``sqrt(1 + λ²)`` from ``z_t − λ z_{t−1}``.

    ``step`` must be in ``[0, n_steps)`` — the noise function raises on
    past-horizon calls, so this lookup is never asked to invent a factor
    outside the calibrated matrix.
    """
    lam = strategy.lambda_
    col = _column_norm(lam, n_steps, step) if strategy.normalized else 1.0
    if step == 0 or lam == 0.0:
        return col
    return col * math.sqrt(1.0 + lam * lam)


def _make_lambda_cgd_noise(
    grad_template: Any,
    strategy: LambdaCgdStrategy,
    *,
    n_steps: int,
    key: RngKey,
    compute_dtype: torch.dtype = torch.float32,
) -> tuple[
    Callable[..., tuple[Any, MFNoiseState]],
    MFNoiseState,
    Callable[[int], float],
]:
    """DP-lambda-CGD noise via PRNG replay without a noise-history buffer.

    Returns ``(noise_fn, state, row_l2_at)`` where ``row_l2_at(step)``
    gives ``‖row_t(C^{-1})‖`` so the wrapping :func:`mf_gaussian_noise`
    factory can publish the realized per-step σ on
    :class:`NoisedPytree.noise_stddev` (= ``base_σ · row_l2_at(step)``).
    Adam-family bias correction reads that realized σ.
    """
    n_steps = _require_positive_int_horizon(n_steps)

    lambda_ = float(strategy.lambda_)
    normalized = bool(strategy.normalized)
    expected_treedef, expected_paths, expected_leaf_layouts = _gradient_layout(
        grad_template
    )
    grad_layout_digest = _gradient_layout_digest(
        expected_treedef,
        expected_paths,
        expected_leaf_layouts,
    )
    execution_identity = _LambdaCgdExecutionIdentity(
        replay_version=_LAMBDA_CGD_REPLAY_VERSION,
        stream_root=LAMBDA_CGD_STREAM_FOLD,
        lambda_=lambda_,
        normalized=normalized,
        n_steps=n_steps,
        compute_dtype=str(compute_dtype),
        rng_seed=key.seed,
        rng_impl=key.impl,
        grad_layout_digest=grad_layout_digest,
    )
    _validate_execution_identity(execution_identity)

    state = MFNoiseState(
        _inner_state=_LambdaCgdReplayState(
            step=0,
            base_stddev_digest=None,
            execution_identity=execution_identity,
        ),
        _step_counter=0,
        _rng_key=key,
    )

    def noise_fn(
        clipped_grads: Any,
        st: MFNoiseState,
        *,
        stddev: float | PerGroup,
    ) -> tuple[Any, MFNoiseState]:
        replay, scale_digest = _validate_lambda_cgd_replay_call(
            clipped_grads,
            st,
            state,
            stddev=stddev,
        )
        step = st._step_counter

        current_key = rng_fold_in(st._rng_key, LAMBDA_CGD_STREAM_FOLD, step)
        g_current = generator_from_key(current_key)
        z_t = _iid_normal_noise(
            clipped_grads,
            stddev,
            generator=g_current,
            compute_dtype=compute_dtype,
        )

        if step == 0 or lambda_ == 0.0:
            corr_noise = z_t
        else:
            prev_key = rng_fold_in(st._rng_key, LAMBDA_CGD_STREAM_FOLD, step - 1)
            g_prev = generator_from_key(prev_key)
            z_prev = _iid_normal_noise(
                clipped_grads,
                stddev,
                generator=g_prev,
                compute_dtype=compute_dtype,
            )
            corr_noise = tree_map(
                lambda zt, zp: zt - lambda_ * zp,
                z_t,
                z_prev,
            )

        if normalized:
            d_t = _column_norm(lambda_, n_steps, step)
            corr_noise = tree_map(lambda n: n * d_t, corr_noise)

        noisy_grads = tree_map(
            lambda grad, n: (grad + n).to(grad.dtype),
            clipped_grads,
            corr_noise,
        )

        new_state = MFNoiseState(
            _inner_state=_LambdaCgdReplayState(
                step=step + 1,
                base_stddev_digest=scale_digest,
                execution_identity=replay.execution_identity,
            ),
            _step_counter=step + 1,
            _rng_key=st._rng_key,
        )
        return noisy_grads, new_state

    def row_l2_at(step: int) -> float:
        return _lambda_cgd_row_l2(strategy, n_steps, step)

    return noise_fn, state, row_l2_at


__all__ = ["LambdaCgdStrategy", "lambda_cgd_strategy"]
