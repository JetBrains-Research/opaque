"""Functional optimizers for Opaque.

Single import path for every functional optimizer used in the Opaque
DP training pipeline — Opaque-built factories with DP-aware paths,
plus the stateless ``torchopt`` primitives we don't need to extend.
Same shape as :mod:`opaque.scheduling`.

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
- :func:`schedule_free` — wrapper around any of the above (or any
  re-exported ``torchopt`` primitive) implementing Defazio's
  schedule-free averaging.

DP-aware behavior on the Opaque-built factories is selected at
``update()`` time via two optional kwargs:

- ``noise_stddev`` — activates the DP-AdamW-BC φ-EMA correction on
  ``v̂``.  Per-step override of the constructor default.
- ``noisy_squared_grads`` — substitutes a JME paired-stream second
  moment in place of ``g²`` (post-processing argument; no extra
  privacy work needed in the optimizer itself).

Re-exported from ``torchopt`` (no DP-aware modes; use directly when
the standard update is enough):

- :func:`sgd` — vanilla / Polyak-momentum SGD.
- :func:`adam`, :func:`adagrad`, :func:`adadelta`, :func:`adamax`,
  :func:`radam`, :func:`rmsprop`.

Each factory returns a ``torchopt.base.GradientTransformation`` and is
state-isolated; multiple optimizers can coexist in the same process
without RNG / global state collisions.

Less-common building blocks are reachable via the submodules and are
intentionally not part of ``__all__`` (mirroring the
:mod:`opaque.clipping` / :mod:`opaque.random` convention — the public
surface is functional):

- State dataclasses — ``AdamState``, ``LionState``, ``AdEMAMixState``,
  ``AdafactorState``, ``ScheduleFreeState`` — re-exported here for type
  annotations.
- ``get_eval_params(state)`` from :mod:`opaque.optimizers.schedule_free`
  — returns the published ``x`` weights from a schedule-free state.
- ``state_dict`` / ``load_state_dict`` from
  :mod:`opaque.optimizers.serialization` — flatten / rebuild any chain
  state for ``torch.save`` / ``torch.load``.
"""

# Re-exports from torchopt: stateless primitives we don't extend.
# Same simplicity pattern as ``opaque.scheduling`` — give users one
# place to import optimizers from.
from torchopt import adadelta as adadelta
from torchopt import adagrad as adagrad
from torchopt import adam as adam
from torchopt import adamax as adamax
from torchopt import radam as radam
from torchopt import rmsprop as rmsprop
from torchopt import sgd as sgd

# Functional surface — listed in ``__all__``.
from opaque.optimizers.adafactor import adafactor
from opaque.optimizers.adam import adamw
from opaque.optimizers.ademamix import ademamix
from opaque.optimizers.lion import lion
from opaque.optimizers.schedule_free import schedule_free

# State dataclasses — re-exported with ``as X`` for type annotation
# discoverability, intentionally not part of ``__all__``.  Same
# convention as ``opaque.clipping.ClipState`` / ``opaque.random.RngKey``.
from opaque.optimizers.adafactor import AdafactorState as AdafactorState
from opaque.optimizers.adam import AdamState as AdamState
from opaque.optimizers.ademamix import AdEMAMixState as AdEMAMixState
from opaque.optimizers.lion import LionState as LionState
from opaque.optimizers.schedule_free import ScheduleFreeState as ScheduleFreeState


__all__ = [
    # Opaque-built factories.
    "adamw",
    "lion",
    "ademamix",
    "adafactor",
    "schedule_free",
    # Re-exported torchopt primitives.
    "sgd",
    "adam",
    "adagrad",
    "adadelta",
    "adamax",
    "radam",
    "rmsprop",
]
