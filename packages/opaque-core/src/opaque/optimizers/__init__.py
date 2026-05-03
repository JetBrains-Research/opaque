"""Functional optimizers for Opaque.

Mechanism-agnostic ``torchopt``-compatible factories.  Optional
DP-aware behavior is selected at ``update()`` time via two kwargs:

- ``noise_stddev`` — activates the DP-AdamW-BC φ-EMA correction on
  ``v̂``.  Per-step override of the constructor default.
- ``noisy_squared_grads`` — substitutes a JME paired-stream second
  moment in place of ``g²`` (post-processing argument; no extra
  privacy work needed in the optimizer itself).

Optimizers in this module:

- :func:`adamw` — universal Adam / AdamW; accepts both DP kwargs.
  Knobs: ``decoupled_weight_decay``, ``update_rms_clip`` (StableAdamW),
  ``noise_stddev`` (DP-AdamW-BC default).
- :func:`lion` — Lion (sign-of-momentum); no DP-aware mode (no v).
- :func:`ademamix` — AdEMAMix (two first moments, single v); accepts
  both DP kwargs.
- :func:`adafactor` — Adafactor with factored second moment.  Phase A
  ships vanilla + WD only; DP-aware modes are deferred (see module
  docstring for the per-axis bias derivation that's pending).
- :func:`schedule_free` — wrapper around any of the above (or
  ``torchopt.sgd``) implementing Defazio's schedule-free averaging.

Each factory returns a ``torchopt.base.GradientTransformation`` and is
state-isolated; multiple optimizers can coexist in the same process
without RNG / global state collisions.
"""

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
    # Factories
    "adamw",
    "lion",
    "ademamix",
    "adafactor",
    "schedule_free",
    # State types (for type annotations / introspection)
    "AdamState",
    "LionState",
    "AdEMAMixState",
    "AdafactorState",
    "ScheduleFreeState",
    # Helpers
    "get_eval_params",
]
