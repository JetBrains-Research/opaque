r"""MSE-optimal per-group noise allocation for per-group clipping.

When using per-group clipping, the simplest approach is isotropic noise:

.. math::

    \sigma = \text{nm} \cdot \Delta_2
           = \text{nm} \cdot \frac{\lVert C \rVert_2}{n}

This uses the same noise standard deviation for every parameter and accounts
correctly with ``gaussian(nm)``.

:func:`per_group_noise_stddev` provides an **alternative** that reduces total
noise MSE by allocating less noise to groups with smaller clipping norms.
It returns a :class:`~opaque.utils.per_group.PerGroup` of per-group standard
deviations satisfying the Mahalanobis privacy constraint:

.. math::

    \sigma_i = \text{nm} \cdot \sqrt{C_i \cdot \textstyle\sum_j C_j} \;/\; n

Privacy accounting is identical to the isotropic case — just
``gaussian(nm)`` — because the allocation satisfies the Mahalanobis
constraint with equality.
"""

from __future__ import annotations

import math

from opaque.clipping.types import ClipState
from opaque.clipping.per_group import PerGroup


def per_group_noise_stddev(clip_state: ClipState, noise_multiplier: float) -> PerGroup:
    r"""Compute MSE-optimal per-group noise standard deviations.

    Given a clip state with per-group clipping norms :math:`C_1, \dots, C_K`
    and ``normalize_by`` :math:`= n`, returns per-group noise standard
    deviations:

    .. math::

        \sigma_i = \text{nm} \cdot
            \frac{\sqrt{C_i \cdot \sum_j C_j}}{n}

    This allocation minimizes the total noise MSE
    :math:`\sum_i d_i \sigma_i^2` (for equal group dimensions) among all
    allocations satisfying the Mahalanobis privacy constraint
    :math:`\sum_i (C_i/n)^2 / \sigma_i^2 \le 1/\text{nm}^2`.

    Privacy accounting is ``gaussian(nm)`` — identical to isotropic noise,
    with no composition penalty regardless of the number of groups.

    Args:
        clip_state: Clipping state with per-group ``clipping_norm``
            (:class:`~opaque.utils.per_group.PerGroup`).
        noise_multiplier: The noise multiplier used for privacy
            accounting via ``gaussian(nm)``.

    Returns:
        :class:`~opaque.utils.per_group.PerGroup` with per-group noise
        standard deviations.

    Raises:
        TypeError: If ``clip_state.clipping_norm`` is not ``PerGroup``.

    Example::

        from opaque.clipping import clipped_grad
        from opaque.dpsgd.noise.gaussian import gaussian_noise
        from opaque.dpsgd.noise.per_group_noise import per_group_noise_stddev

        grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=pg, ...)
        stddev = per_group_noise_stddev(clip_state, nm)
        noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(42))

        # Accounting: just gaussian(nm), same as isotropic.
    """
    cn = clip_state.clipping_norm
    if not isinstance(cn, PerGroup):
        raise TypeError(
            f"per_group_noise_stddev requires per-group clipping_norm, "
            f"got {type(cn).__name__}. Use nm * clip_state.sensitivity instead."
        )
    n = clip_state.normalize_by
    sum_c = sum(cn.values.values())
    return PerGroup(
        cn.groups,
        {k: noise_multiplier * math.sqrt(c * sum_c) / n for k, c in cn.values.items()},
    )


__all__ = ["per_group_noise_stddev"]
