"""Functional optimizers for Opaque.

Single import path for every functional optimizer used in the Opaque
DP training pipeline — Opaque-built factories with a common wrapper-aware
update surface and DP-aware modes selected by the update value type.

Opaque-built:

- :func:`adamw` — universal Adam / AdamW; consumes ``NoisedPytree``
  and ``SecondMomentNoiseOutput`` metadata.
  Knobs: ``decoupled_weight_decay``, ``update_rms_clip`` (StableAdamW),
  ``noise_bias_correction``.
- :func:`adam` — original Adam/L2 variant of ``adamw`` with the same
  wrapper-aware update surface.
- :func:`radam` — Rectified Adam.  φ-EMA bias correction on the
  second moment when ``ρ_t > 5``; the early ``ρ_t ≤ 5`` SGD-of-momentum
  branch is naturally DP-robust (no v).
- :func:`sgd` — SGD wrapper that accepts ``NoisedPytree`` updates and ignores
  noise metadata because the update is unbiased under additive DP noise.
- :func:`lion` — Lion (sign-of-momentum); no DP-aware mode (no v).
- :func:`ademamix` — AdEMAMix (two first moments, single v); consumes
  the same DP metadata wrappers as AdamW.
- :func:`adafactor` — Adafactor with factored second moment.  Optional
  DP noise-variance bias correction subtracts a φ-EMA from each factor
  (``noise_bias_correction``).  No private second-moment substitution
  path — the privatised ``g²`` stream doesn't preserve the factorisation
  cleanly (see module docstring).
- :func:`rmsprop` — RMSprop with optional DP-aware φ-EMA correction
  on the second moment.
- :func:`adagrad` — Adagrad with optional DP-aware cumulative noise
  variance subtraction.  ``NoisedPytree`` updates with
  ``noise_bias_correction=True`` activate the ``v_acc - Φ_acc``
  correction; whether that helps in practice depends on the workload.
- :func:`schedule_free` — generic wrapper around any base
  ``GradientTransformation`` (Opaque-built or torchopt-imported)
  implementing Defazio's schedule-free averaging.  Read the published
  ``x`` weights for saving / evaluation directly from the state's
  ``state.x`` field (see :class:`opaque.optimizers.types.ScheduleFreeState`).

DP-aware behavior is selected at ``update()`` time by passing metadata
wrappers:

- ``NoisedPytree`` — carries the realized per-step noise σ with the
  privatized update; this activates whatever noise-aware path the
  optimizer has (φ-EMA on ``v̂`` for Adam-family / RMSprop / RAdam,
  cumulative Φ subtraction for Adagrad, planned sign gating for Lion,
  …).
- ``SecondMomentNoiseOutput`` — carries private first- and second-moment
  streams together and substitutes the private squared-gradient stream
  in place of ``g²`` (post-processing inside the optimizer).

``adamax`` is intentionally *not* exposed: its L∞
``u_t = max(β₂ u_{t-1}, |g_t|)`` rule rectifies the gradient before
the EMA, so the half-normal noise mean ``σ √(2/π)`` is permanently
absorbed and the standard variance-EMA BC trick does not apply.  No
principled fix has been published.

Each factory returns a ``torchopt.base.GradientTransformation`` and is
state-isolated; multiple optimizers can coexist in the same process
without RNG / global state collisions.

``state_dict`` / ``load_state_dict`` flatten / rebuild any chain state
for ``torch.save`` / ``torch.load``.

Per-optimizer state dataclasses (``AdamState``, ``LionState``,
``AdEMAMixState``, ``AdafactorState``, ``RMSpropState``,
``AdagradState``, ``RAdamState``, ``AdadeltaState``,
``ScheduleFreeState``) live in :mod:`opaque.optimizers.types`.
"""

from opaque.optimizers._adadelta import adadelta
from opaque.optimizers._adafactor import adafactor
from opaque.optimizers._adagrad import adagrad
from opaque.optimizers._adam import adam, adamw
from opaque.optimizers._ademamix import ademamix
from opaque.optimizers._lion import lion
from opaque.optimizers._radam import radam
from opaque.optimizers._rmsprop import rmsprop
from opaque.optimizers._schedule_free import schedule_free
from opaque.optimizers._serialization import load_state_dict, state_dict
from opaque.optimizers._sgd import sgd

__all__ = [
    "adam",
    "adamw",
    "sgd",
    "lion",
    "ademamix",
    "adafactor",
    "rmsprop",
    "adagrad",
    "adadelta",
    "radam",
    "schedule_free",
    "state_dict",
    "load_state_dict",
]
