r"""MSE-optimal per-group noise allocation for per-group clipping.

When using per-group clipping, the simplest approach is isotropic noise:

.. math::

        \sigma = \text{nm} \cdot \Delta_2
            = \text{nm} \cdot \lVert B \rVert_2

This uses the same noise standard deviation for every parameter and accounts
correctly with ``gaussian(nm)``.

:func:`per_group_noise_stddev` provides an **alternative** that reduces total
noise MSE by allocating less noise to groups with smaller clipping norms.
It returns a :class:`~opaque.utils.per_group.PerGroup` of per-group standard
deviations satisfying the Mahalanobis privacy constraint:

.. math::

    \sigma_i = \text{nm} \cdot \sqrt{B_i \cdot \textstyle\sum_j B_j}

Privacy accounting is identical to the isotropic case — just
``gaussian(nm)`` — because the allocation satisfies the Mahalanobis
constraint with equality.
"""

from __future__ import annotations

import math

from opaque.types import PerGroup


def per_group_noise_stddev(max_norm: PerGroup, noise_multiplier: float) -> PerGroup:
    r"""Compute MSE-optimal per-group noise standard deviations.

    Given per-group contribution bounds :math:`B_1, \dots, B_K`, returns
    per-group noise standard deviations:

    .. math::

        \sigma_i = \text{nm} \cdot
            \sqrt{B_i \cdot \sum_j B_j}

    This allocation minimizes the total noise MSE
    :math:`\sum_i d_i \sigma_i^2` (for equal group dimensions) among all
    allocations satisfying the Mahalanobis privacy constraint
    :math:`\sum_i (C_i/n)^2 / \sigma_i^2 \le 1/\text{nm}^2`.

    Privacy accounting is ``gaussian(nm)`` — identical to isotropic noise,
    with no composition penalty regardless of the number of groups.

    Args:
        max_norm: Per-group contribution bounds, typically
            ``clipped_grads.max_norm`` from per-group clipping.
        noise_multiplier: The noise multiplier used for privacy
            accounting via ``gaussian(nm)``.

    Returns:
        :class:`~opaque.utils.per_group.PerGroup` with per-group noise
        standard deviations.

    Raises:
        TypeError: If ``max_norm`` is not ``PerGroup``.

    Example::

        from opaque.clipping import clipped_grad
        from opaque.dpsgd.noise import per_group_noise_stddev

        grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=pg, ...)
        grads, clip_state = grad_fn(params, batch, state=clip_state)
        stddev = per_group_noise_stddev(grads.max_norm, nm)
        # stddev is a PerGroup allocation for mechanisms that accept
        # per-group standard deviations directly.

        # Accounting: just gaussian(nm), same as isotropic.
    """
    if not isinstance(max_norm, PerGroup):
        raise TypeError(
            "per_group_noise_stddev requires a PerGroup max_norm, "
            f"got {type(max_norm).__name__}."
        )
    for group_name, value in max_norm.values.items():
        if value < 0:
            raise ValueError(
                "per-group bounds must be non-negative, "
                f"got {value} for group '{group_name}'."
            )
    sum_c = sum(max_norm.values.values())
    return PerGroup(
        max_norm.groups,
        {
            k: noise_multiplier * math.sqrt(c * sum_c)
            for k, c in max_norm.values.items()
        },
    )


__all__ = ["per_group_noise_stddev"]
