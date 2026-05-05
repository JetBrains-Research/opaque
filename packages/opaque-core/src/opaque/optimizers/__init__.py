"""Functional optimizers for Opaque.

Single import path for every functional optimizer used in the Opaque
DP training pipeline — Opaque-built factories with DP-aware paths,
plus the stateless ``torchopt`` primitives we don't need to extend.

Opaque-built (DP-aware modes selectable at ``update()`` time):

- :func:`adamw` — universal Adam / AdamW; accepts both DP kwargs.
  Knobs: ``decoupled_weight_decay``, ``update_rms_clip`` (StableAdamW),
  ``noise_stddev`` (DP-AdamW-BC default).
- :func:`lion` — Lion (sign-of-momentum); no DP-aware mode (no v).
- :func:`ademamix` — AdEMAMix (two first moments, single v); accepts
  both DP kwargs.
- :func:`adafactor` — Adafactor with factored second moment.  Phase A
  ships vanilla + WD only; DP-aware modes deferred (see module
  docstring for the per-axis bias derivation that's pending).
- :func:`rmsprop` — RMSprop with optional DP-aware φ-EMA correction
  on the second moment.
- :func:`adagrad` — Adagrad with optional DP-aware cumulative noise
  variance subtraction.  Vanilla Adagrad's denominator runs away
  under DP noise; the ``noise_stddev`` kwarg activates the
  ``v_acc - Φ_acc`` correction so the optimizer stays usable.
- :func:`schedule_free` — wrapper around any base
  ``GradientTransformation`` (Opaque-built or torchopt-imported)
  implementing Defazio's schedule-free averaging.

DP-aware behavior is selected at ``update()`` time via two optional
kwargs:

- ``noise_stddev`` — tells the optimizer the per-step noise σ;
  activates whatever noise-aware path it has (φ-EMA on ``v̂`` for
  Adam-family / RMSprop, cumulative Φ subtraction for Adagrad,
  planned sign gating for Lion, …).  Per-step override of the
  constructor default.
- ``noisy_squared_grads`` — substitutes a private squared-gradient
  stream in place of ``g²`` (post-processing argument; no extra privacy
  work needed in the optimizer itself).

Re-exported from ``torchopt`` (no DP-aware modes; vanilla but safe
under DP noise — slow without bias correction but converges):

- :func:`sgd` — vanilla / Polyak-momentum SGD.  Update is unbiased
  under noise; canonical DP baseline.
- :func:`adam` — for users who specifically want torchopt's Adam
  without DP modes.  For DP, prefer :func:`adamw` with
  ``decoupled_weight_decay=False`` and ``noise_stddev``.
- :func:`adadelta` — has two EMAs; the ratio partially self-corrects
  under noise so it's not broken, just non-optimal.  Re-exported
  for users who specifically want it.
- :func:`radam` — Rectified Adam.  Has v with EMA decay; same DP-BC
  story as Adam, just not yet implemented as an Opaque-built variant.

Each factory returns a ``torchopt.base.GradientTransformation`` and is
state-isolated; multiple optimizers can coexist in the same process
without RNG / global state collisions.

Less-common building blocks are reachable via the submodules and are
intentionally not part of ``__all__`` (mirroring the
:mod:`opaque.clipping` / :mod:`opaque.random` convention — the public
surface is functional):

- State dataclasses — ``AdamState``, ``LionState``, ``AdEMAMixState``,
  ``AdafactorState``, ``RMSpropState``, ``AdagradState``,
  ``ScheduleFreeState`` — re-exported here for type annotations.
- ``get_eval_params(state)`` from :mod:`opaque.optimizers.schedule_free`
  — returns the published ``x`` weights from a schedule-free state.
- ``state_dict`` / ``load_state_dict`` from
  :mod:`opaque.optimizers.serialization` — flatten / rebuild any chain
  state for ``torch.save`` / ``torch.load``.
"""

# Re-exports from torchopt: stateless primitives we don't extend
# because their vanilla behaviour is acceptable (if non-optimal) under
# DP noise.  ``adagrad``, ``rmsprop`` are *not* re-exported — Opaque
# ships its own factories with DP-aware corrections at the same names
# below.  ``adamax`` is omitted because the max-norm structurally
# misbehaves under DP (half-normal noise mean is permanently absorbed);
# users who want it can ``from torchopt import adamax`` directly.
from torchopt import adadelta as adadelta
from torchopt import adam as adam
from torchopt import radam as radam
from torchopt import sgd as sgd

# Functional surface — listed in ``__all__``.
from opaque.optimizers.adafactor import adafactor
from opaque.optimizers.adagrad import adagrad
from opaque.optimizers.adam import adamw
from opaque.optimizers.ademamix import ademamix
from opaque.optimizers.lion import lion
from opaque.optimizers.rmsprop import rmsprop
from opaque.optimizers.schedule_free import schedule_free

# State dataclasses — re-exported with ``as X`` for type annotation
# discoverability, intentionally not part of ``__all__``.  Same
# convention as ``opaque.clipping.ClipState`` / ``opaque.random.RngKey``.
from opaque.optimizers.adafactor import AdafactorState as AdafactorState
from opaque.optimizers.adagrad import AdagradState as AdagradState
from opaque.optimizers.adam import AdamState as AdamState
from opaque.optimizers.ademamix import AdEMAMixState as AdEMAMixState
from opaque.optimizers.lion import LionState as LionState
from opaque.optimizers.rmsprop import RMSpropState as RMSpropState
from opaque.optimizers.schedule_free import ScheduleFreeState as ScheduleFreeState


__all__ = [
    # Opaque-built factories.
    "adamw",
    "lion",
    "ademamix",
    "adafactor",
    "rmsprop",
    "adagrad",
    "schedule_free",
    # Re-exported torchopt primitives.
    "sgd",
    "adam",
    "adadelta",
    "radam",
]
