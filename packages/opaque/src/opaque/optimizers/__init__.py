"""DP optimizers built on TorchOpt.

Provides DP-aware optimizer variants that follow TorchOpt's
``GradientTransformation`` protocol::

    state = opt.init(params)
    updates, state = opt.update(grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
"""

from opaque.optimizers.jme_adam import JmeAdamState, jme_adam

__all__ = [
    "jme_adam",
    "JmeAdamState",
]
