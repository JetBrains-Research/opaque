"""Schedule-Free wrapper (Defazio et al., 2024).

Composes around any base ``GradientTransformation`` (Adam, AdamW,
Lion, AdEMAMix, …) to give it Defazio's schedule-free averaging — a
training mode that produces published checkpoints from a Polyak-Ruppert-
style uniform average rather than relying on an external LR schedule
(warmup, cosine, …).

Reference:
    Defazio, Yaida, Cutkosky, "The Road Less Scheduled",
    arXiv:2405.15682.

Three weight sequences::

    z_t  : raw iterate updated by the wrapped optimizer
    x_t  : uniform-averaged "published" weights (saved checkpoint)
    y_t  : interpolation y_t = (1 − β) z_t + β x_{t-1} — used for
           forward passes during training

Update step (gradient computed at ``y_t``)::

    g_t   = ∇L(y_t)
    z_{t+1} = z_t − wrapped_optimizer_update(g_t)
    x_{t+1} = (1 − 1/(t+1)) x_t + (1/(t+1)) z_{t+1}
    y_{t+1} = (1 − β) z_{t+1} + β x_{t+1}

The wrapper presents the standard ``GradientTransformation`` interface,
but its update returns a delta that takes the trainer's ``params``
(treated as ``y_t``) to ``y_{t+1}``.  Call :func:`get_eval_params` at
save / evaluation boundaries to read the published ``x_t`` sequence;
:func:`get_train_params` reconstructs the interpolated ``y_t`` used
during training.

DP / privacy notes.

- The privacy mechanism (clipping, noise) attaches to the gradient at
  ``y_t`` exactly as in vanilla DP-SGD; the published average ``x_t``
  is a deterministic function of the (already-private) ``z`` trajectory,
  so by post-processing the privacy guarantee is unchanged.
- Wrapped optimizers consume ``NoisedPytree`` / ``SecondMomentNoiseOutput`` on
  ``updates`` as usual (same as without this wrapper); the public ``update``
  surface does not add DP metadata kwargs.

Trainer-integration caveat.  The wrapper's published params are
``x_t``, not the ``params`` argument the trainer passes in (``y_t``).
Saving / evaluating against ``y_t`` would defeat the purpose of
schedule-free averaging.  Use :func:`get_eval_params` at save / eval
boundaries (and :func:`get_train_params` if a later training step
needs the interpolated iterate restored from optimizer state).
``DPTrainer`` does not yet switch sequences automatically.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

try:
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opaque.pytree import tree_map

if TYPE_CHECKING:
    from opaque.types import TensorPytree


@dataclasses.dataclass(frozen=True)
class ScheduleFreeState:
    """State for the schedule-free wrapper.

    Attributes:
        z: Raw iterate (pytree matching params).
        x: Uniformly-averaged published weights (pytree matching params).
            Prefer :func:`get_eval_params` over reading this field
            directly; the accessor returns a detached snapshot.
        inner: State of the wrapped ``GradientTransformation``.
        step: Number of completed updates.
        beta: Interpolation coefficient β.
    """

    z: TensorPytree
    x: TensorPytree
    inner: Any  # opaque ChainState — the inner GradientTransformation's state
    step: int
    beta: float


def schedule_free(
    base: GradientTransformation,
    *,
    beta: float = 0.9,
    warmup_steps: int = 0,
) -> GradientTransformation:
    """Wrap a base optimizer with schedule-free averaging.

    The trainer treats the returned object as a normal optimizer: it
    feeds in ``y_t`` as ``params``, the wrapper internally maintains
    ``z_t`` and ``x_t``, and the returned update advances ``y_t`` to
    ``y_{t+1}``.

    Args:
        base: Any ``GradientTransformation`` (e.g.
            :func:`opaque.optimizers._adamw`, :func:`opaque.optimizers._lion`,
            :func:`torchopt.sgd`, …).  Its ``init`` is called on
            ``params`` and its ``update`` consumes the gradient at
            ``y_t``.
        beta: Interpolation coefficient between ``z`` and ``x`` for the
            forward-pass weights ``y``.  Paper default 0.9.
        warmup_steps: When > 0, the averaging weight on the first
            ``warmup_steps`` is set to 0 (i.e. ``x`` follows ``z``
            exactly during warm-up).  Useful for early instability;
            disable by leaving at 0.

    Returns:
        A ``torchopt.base.GradientTransformation`` whose state is a
        :class:`ScheduleFreeState`.  Use :func:`get_eval_params` to
        retrieve the published weights for saving / evaluation, and
        :func:`get_train_params` to reconstruct the interpolated
        training iterate.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must satisfy 0 <= beta <= 1, got {beta}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")

    def init_fn(params: Any) -> ScheduleFreeState:
        # ``params`` represents y₀; we initialise z = x = y₀ so the
        # interpolation matches the input on the first step.
        z = tree_map(lambda p: p.clone(), params)
        x = tree_map(lambda p: p.clone(), params)
        return ScheduleFreeState(
            z=z,
            x=x,
            inner=base.init(params),
            step=0,
            beta=beta,
        )

    def update_fn(
        updates: Any,
        state: ScheduleFreeState,
        *,
        params: Any = None,
        inplace: bool = False,
    ) -> tuple[Any, ScheduleFreeState]:
        # ``updates`` is the gradient ∇L(y_t); ``params`` is y_t.
        if params is None:
            raise ValueError(
                "schedule_free requires `params` (interpreted as y_t) "
                "to be passed at update time."
            )
        # Step the wrapped optimizer to get its delta (already
        # negative-LR-scaled by the wrapped chain).  Crucially, the
        # base optimizer is told ``params=state.z`` rather than ``y_t``:
        # decoupled / L2 weight decay must reference the raw iterate
        # ``z`` (which is what the inner optimizer's update is being
        # added to), not the interpolated ``y`` we use for the forward
        # pass.  Mismatching this regularises the wrong tensor and
        # quietly changes the algorithm.
        inner_update, new_inner = base.update(
            updates, state.inner, params=state.z, inplace=inplace
        )
        # z_{t+1} = z_t + inner_update  (inner_update is the negative
        # step the wrapped optimizer would have applied to params).
        new_z = tree_map(lambda z, du: z + du, state.z, inner_update)
        # Uniform average over post-warmup iterates::
        #
        #   x_{t+1} = (1 − w) x_t + w z_{t+1}
        #
        # During warm-up (``state.step < warmup_steps``) we set ``x=z``
        # so the average doesn't anchor to early-training noise.  After
        # warm-up we use ``w = 1 / post_warmup_t``: the first averaged
        # iterate has ``w=1`` (start a fresh average), and subsequent
        # iterates get the standard Polyak-Ruppert ``1/n`` weight.
        # Using the global ``t`` here (the bug) leaves ``x`` anchored to
        # the warmup-end ``z`` because the first post-warmup step would
        # only move it by ``1/(warmup_steps+1)``.
        t = state.step + 1
        if state.step < warmup_steps:
            new_x = tree_map(lambda _, z: z.clone(), state.x, new_z)
        else:
            post_warmup_t = t - warmup_steps  # 1 on the first post-warmup step
            w = 1.0 / float(post_warmup_t)
            new_x = tree_map(lambda x, z: (1.0 - w) * x + w * z, state.x, new_z)
        # y_{t+1} = (1 − β) z_{t+1} + β x_{t+1}
        new_y = tree_map(
            lambda z, x: (1.0 - state.beta) * z + state.beta * x, new_z, new_x
        )
        # Delta: y_{t+1} − y_t (current params).  The trainer applies
        # this as ``params + delta``, recovering y_{t+1}.
        delta = tree_map(lambda y_new, y_old: y_new - y_old, new_y, params)
        return delta, ScheduleFreeState(
            z=new_z, x=new_x, inner=new_inner, step=t, beta=state.beta
        )

    return GradientTransformation(init_fn, update_fn)


def _require_schedule_free_state(state: Any, *, caller: str) -> ScheduleFreeState:
    if not isinstance(state, ScheduleFreeState):
        raise TypeError(
            f"{caller} expects ScheduleFreeState, got {type(state).__name__}"
        )
    return state


def get_eval_params(state: ScheduleFreeState) -> TensorPytree:
    """Return a snapshot of the published schedule-free weights ``x_t``.

    Evaluation and checkpointed model weights should use this sequence,
    not the trainer's interpolated iterate ``y_t``.  The returned pytree
    is cloned, so in-place evaluation mutation cannot corrupt optimizer
    state.

    Args:
        state: Optimizer state produced by :func:`schedule_free`.

    Returns:
        A pytree matching ``state.x``.

    Raises:
        TypeError: If ``state`` is not a :class:`ScheduleFreeState`.
    """
    state = _require_schedule_free_state(state, caller="get_eval_params")
    return tree_map(lambda p: p.detach().clone(), state.x)


def get_train_params(state: ScheduleFreeState) -> TensorPytree:
    """Reconstruct the interpolated training iterate ``y_t``.

    ``y_t = (1 − β) z_t + β x_t``.  After restoring optimizer state
    from a checkpoint, this is the iterate a trainer should resume
    from; it matches the ``params`` pytree :func:`schedule_free`
    ``update`` last produced.

    Args:
        state: Optimizer state produced by :func:`schedule_free`.

    Returns:
        A pytree matching the trainer's ``params`` / ``y_t``.

    Raises:
        TypeError: If ``state`` is not a :class:`ScheduleFreeState`.
    """
    state = _require_schedule_free_state(state, caller="get_train_params")
    return tree_map(
        lambda z, x: ((1.0 - state.beta) * z + state.beta * x).detach().clone(),
        state.z,
        state.x,
    )


__all__ = [
    "ScheduleFreeState",
    "get_eval_params",
    "get_train_params",
    "schedule_free",
]
