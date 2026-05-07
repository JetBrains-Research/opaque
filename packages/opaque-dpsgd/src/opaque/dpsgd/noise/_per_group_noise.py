r"""MSE-optimal per-group noise allocation for per-group clipping.

When using per-group clipping, the simplest approach is isotropic noise:

.. math::

        \sigma = \text{nm} \cdot \Delta_2
            = \text{nm} \cdot \lVert B \rVert_2

This uses the same noise standard deviation for every parameter and accounts
correctly with ``gaussian(nm)``.

:func:`per_group_noise_stddev` provides an **alternative** that reduces total
noise MSE by allocating less noise to groups with smaller clipping norms.
It returns a :class:`~opaque.types.PerGroup` of per-group standard
deviations satisfying the Mahalanobis privacy constraint:

.. math::

    \sigma_i = \text{nm} \cdot \sqrt{B_i \cdot \textstyle\sum_j B_j}

Privacy accounting is identical to the isotropic case — just
``gaussian(nm)`` — because the allocation satisfies the Mahalanobis
constraint with equality.

Equivalent to ``ClippedPytree.noise_stddev_for(noise_multiplier=nm,
allocation='optimal')``; kept as a free function for callers that already
hold a bare :class:`PerGroup` (for example one restored with
:mod:`opaque.serialization`) rather than a clipped pytree.
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
        :class:`~opaque.types.PerGroup` with per-group noise standard
        deviations.

    Raises:
        TypeError: If ``max_norm`` is not ``PerGroup``.
        ValueError: If ``noise_multiplier`` is negative or any group bound
            is negative.

    Example::

        from opaque.clipping import clipped_grad
        from opaque.dpsgd.noise import per_group_noise_stddev

        grad_fn, clip_state = clipped_grad(loss_fn, clipping_norm=pg, ...)
        grads, clip_state = grad_fn(params, batch, state=clip_state)
        stddev = per_group_noise_stddev(grads.max_norm, nm)
        # Equivalent: stddev = grads.noise_stddev_for(noise_multiplier=nm)
    """
    if not isinstance(max_norm, PerGroup):
        raise TypeError(
            "per_group_noise_stddev requires a PerGroup max_norm, "
            f"got {type(max_norm).__name__}."
        )
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
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


def per_group_paired_noise_stddevs(
    first_max_norm: PerGroup,
    squared_max_norm: PerGroup,
    noise_multiplier: float,
) -> tuple[PerGroup, PerGroup]:
    r"""MSE-optimal joint Gaussian allocation for paired per-group release.

    Extends :func:`per_group_noise_stddev` to the joint first +
    second-moment paired release.  Given per-group sensitivities
    :math:`\Delta^{(1)}_g` (first-moment) and :math:`\Delta^{(2)}_g`
    (second-moment) for each group :math:`g`, returns a pair of
    :class:`~opaque.types.PerGroup` standard deviations satisfying::

        S = Σ_h (Δ¹_h + Δ²_h)
        σ¹_g = nm · sqrt(Δ¹_g · S)
        σ²_g = nm · sqrt(Δ²_g · S)

    This satisfies the Mahalanobis privacy constraint with equality
    over the joint 2K-stream release::

        Σ_g [(Δ¹_g/σ¹_g)² + (Δ²_g/σ²_g)²] = 1/nm²

    so privacy accounting is identical to :func:`gaussian` with the same
    ``noise_multiplier`` — no composition penalty across groups or
    streams.  Compared to the scalar overhead-based allocation in
    :func:`opaque.dpsgd.noise._second_moment.second_moment_stddevs`,
    this is data-driven (no ``ρ`` parameter) and always MSE-optimal
    for equal per-coordinate dimensions within each group.

    Args:
        first_max_norm: Per-group first-stream sensitivities (typically
            ``C_g / batch_size`` where ``C_g`` is the per-group clipping
            norm).
        squared_max_norm: Per-group second-stream sensitivities
            (typically ``C_g² / batch_size``).  Must share group
            membership with ``first_max_norm``.
        noise_multiplier: Privacy parameter; the same value used in
            ``gaussian(nm)`` accounting.

    Returns:
        ``(σ_first, σ_second)`` — two :class:`~opaque.types.PerGroup`
        objects with the same group keys as the inputs.

    Raises:
        TypeError: if either argument is not :class:`PerGroup`.
        ValueError: if the group mappings differ, the group sets
            differ, ``noise_multiplier`` is negative, or any
            sensitivity is negative.
    """
    if not isinstance(first_max_norm, PerGroup):
        raise TypeError(
            "per_group_paired_noise_stddevs requires a PerGroup "
            f"first_max_norm, got {type(first_max_norm).__name__}."
        )
    if not isinstance(squared_max_norm, PerGroup):
        raise TypeError(
            "per_group_paired_noise_stddevs requires a PerGroup "
            f"squared_max_norm, got {type(squared_max_norm).__name__}."
        )
    if first_max_norm.groups != squared_max_norm.groups:
        raise ValueError(
            "first_max_norm and squared_max_norm must share the same groups mapping."
        )
    if set(first_max_norm.values) != set(squared_max_norm.values):
        raise ValueError(
            "first_max_norm and squared_max_norm must have identical "
            f"group sets; got {sorted(first_max_norm.values)} vs "
            f"{sorted(squared_max_norm.values)}."
        )
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    for name, value in first_max_norm.values.items():
        if value < 0:
            raise ValueError(
                "first-stream per-group bounds must be non-negative, "
                f"got {value} for group '{name}'."
            )
    for name, value in squared_max_norm.values.items():
        if value < 0:
            raise ValueError(
                "second-stream per-group bounds must be non-negative, "
                f"got {value} for group '{name}'."
            )
    s = sum(first_max_norm.values.values()) + sum(squared_max_norm.values.values())
    sigma_first = PerGroup(
        first_max_norm.groups,
        {
            k: noise_multiplier * math.sqrt(v * s)
            for k, v in first_max_norm.values.items()
        },
    )
    sigma_second = PerGroup(
        squared_max_norm.groups,
        {
            k: noise_multiplier * math.sqrt(v * s)
            for k, v in squared_max_norm.values.items()
        },
    )
    return sigma_first, sigma_second


__all__ = ["per_group_noise_stddev", "per_group_paired_noise_stddevs"]
