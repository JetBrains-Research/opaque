"""Opaque's differential privacy accounting interface.

This module re-exports the opaque-accounting package, which provides
compositional privacy accounting via Privacy Loss Distributions (PLD).

The underlying opaque-accounting package is independent and can be used
standalone with other libraries.

Example::

    from opaque.accounting import poisson, gaussian

    step = poisson(gaussian(1.1), sample_rate=0.01)
    training = step * 1000
    epsilon = training.epsilon_at(1e-5)

For calibration (finding parameters for target privacy):

    from opaque.accounting import calibration as cal

    budget = cal.epsilon_budget(3.0, delta=1e-5)
    result = cal.calibrate(
        budget,
        lambda nm: poisson(gaussian(nm), sample_rate=0.01) * 1000,
        param_min=0.1, param_max=5.0,
    )
"""

from opaque_accounting import *  # noqa: F401, F403
