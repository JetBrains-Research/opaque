"""DP optimizers built on TorchOpt.

Provides DP-aware optimizer variants that follow TorchOpt's
``GradientTransformation`` protocol::

    state = opt.init(params)
    updates, state = opt.update(grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
"""

from opaque.optimizers.adamw_bc import AdamWBCState, adamw_bc
from opaque.optimizers.adamw_jme import AdamWJMEState, adamw_jme

__all__ = [
    "adamw_bc",
    "AdamWBCState",
    "adamw_jme",
    "AdamWJMEState",
]
