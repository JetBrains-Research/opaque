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

Serialisation: :func:`state_dict` flattens any chain optimizer state
into a ``dict[str, Any]`` of tensors / primitives suitable for
``torch.save``; :func:`load_state_dict` rebuilds a state by applying
the dict to a fresh template (``opt.init(params)``).
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

from opaque.optimizers._state_dict import load_state_dict, state_dict
from opaque.optimizers.adafactor import AdafactorState, adafactor
from opaque.optimizers.adam import AdamState, adamw
from opaque.optimizers.ademamix import AdEMAMixState, ademamix
from opaque.optimizers.lion import LionState, lion
from opaque.optimizers.schedule_free import (
    ScheduleFreeState,
    get_eval_params,
    schedule_free,
)


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
    # State types (for type annotations / introspection).
    "AdamState",
    "LionState",
    "AdEMAMixState",
    "AdafactorState",
    "ScheduleFreeState",
    # Helpers.
    "get_eval_params",
    "state_dict",
    "load_state_dict",
]
