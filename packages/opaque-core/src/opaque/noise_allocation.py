r"""MSE-optimal noise allocation helpers shared across DP-SGD and DP-FTRL.

These are pure mathematical utilities: given clipping-derived sensitivities
(scalar or :class:`~opaque.types.PerGroup`) they return the noise standard
deviations that satisfy the joint Mahalanobis Gaussian privacy budget with
equality, so the joint release has the same PLD as a single sensitivity-1
Gaussian at the chosen ``noise_multiplier``.

Two helpers live here:

- :func:`per_group_noise_stddev` — single-stream MSE-optimal allocation.
- :func:`paired_noise_stddevs` — paired first + element-wise-squared
  release; polymorphic in each stream (``float`` or ``PerGroup``).

Both are consumed from :mod:`opaque.dpsgd.noise` and
:mod:`opaque.dpftrl.noise`; they live in :mod:`opaque-core` to avoid an
``opaque-dpftrl → opaque-dpsgd`` package dependency.
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


def _validate_paired_sensitivity(value: float | PerGroup, *, label: str) -> None:
    if isinstance(value, PerGroup):
        for name, v in value.values.items():
            if v < 0:
                raise ValueError(
                    f"{label} per-group bounds must be non-negative, "
                    f"got {v} for group '{name}'."
                )
    else:
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{label} must be float or PerGroup, got {type(value).__name__}."
            )
        if value < 0:
            raise ValueError(f"{label} must be non-negative, got {value}.")


def _sum_sensitivity(value: float | PerGroup) -> float:
    if isinstance(value, PerGroup):
        return float(sum(value.values.values()))
    return float(value)


def _scale_sensitivity(
    value: float | PerGroup, sum_total: float, noise_multiplier: float
) -> float | PerGroup:
    """``σ = nm · sqrt(Δ · S)`` for a single stream (scalar or per-group)."""
    if isinstance(value, PerGroup):
        return PerGroup(
            value.groups,
            {
                k: noise_multiplier * math.sqrt(v * sum_total)
                for k, v in value.values.items()
            },
        )
    return noise_multiplier * math.sqrt(float(value) * sum_total)


def paired_noise_stddevs(
    noise_multiplier: float,
    *,
    first: float | PerGroup,
    second: float | PerGroup,
) -> tuple[float | PerGroup, float | PerGroup]:
    r"""MSE-optimal joint Gaussian allocation for the paired release.

    Given per-record sensitivities :math:`\Delta^{(1)}_g` (first-moment)
    and :math:`\Delta^{(2)}_g` (second-moment) for each group :math:`g`,
    returns standard deviations satisfying::

        S = Σ_h (Δ¹_h + Δ²_h)
        σ¹_g = nm · sqrt(Δ¹_g · S)
        σ²_g = nm · sqrt(Δ²_g · S)

    The 2K-stream Mahalanobis budget evaluates to ``1/nm²`` with equality::

        Σ_g [(Δ¹_g/σ¹_g)² + (Δ²_g/σ²_g)²] = 1/nm²

    so the joint mechanism has the **same PLD as a single sensitivity-1
    Gaussian release at noise multiplier ``nm``**.

    Note that ``noise_multiplier`` here is the **effective Gaussian noise
    multiplier of the joint release**, which is *not always* the same as
    the calibrated parameter of the underlying first-moment mechanism:

    - **DP-SGD with identity strategy** (``‖C‖ = 1``): pass the
      calibrated ``gaussian(nm)`` parameter directly.  The joint paired
      release has the same PLD as ``gaussian(nm)``.
    - **DP-FTRL with strategy ``‖C₁‖``**: ``MfGaussian(nm, ‖C₁‖)`` has
      PLD ``gaussian_pld(nm / ‖C₁‖)``.  The dispatcher therefore passes
      ``nm / ‖C₁‖`` here so the joint Mahalanobis budget evaluates to
      ``(‖C₁‖ / nm)²`` and the joint PLD matches.

    The function is polymorphic in each stream independently:

    - ``first`` and ``second`` ``float`` → returns ``(float, float)``.  This
      is the scalar collapse (``K=1``) of the per-group form.
    - ``first`` and ``second`` :class:`~opaque.types.PerGroup` with matching
      groups → returns ``(PerGroup, PerGroup)``.
    - mixed kinds (one ``float`` and one ``PerGroup``) → ``TypeError``.

    Args:
        noise_multiplier: Effective Gaussian noise multiplier of the
            joint paired release.  See note above on translating from the
            underlying mechanism's calibrated parameter (no-op for
            DP-SGD identity strategy; divide by ``‖C₁‖`` for DP-FTRL).
        first: First-stream per-record sensitivity ``Δ¹``.  For DP-SGD
            averaged clipping that is ``C / n`` (or ``PerGroup`` with
            ``C_g / n``).  For DP-FTRL it is the strategy-amplified
            ``ζ · ‖C₁‖``.
        second: Second-stream per-record sensitivity ``Δ²``.  Typically
            obtained as ``first * first`` element-wise; for DP-FTRL it is
            ``ζ² · ‖C₂‖``.

    Returns:
        ``(σ_first, σ_second)`` matching the input kind on each stream.

    Raises:
        TypeError: if the two streams have different kinds (one scalar
            and one ``PerGroup``).
        ValueError: if ``noise_multiplier`` is negative, the two
            ``PerGroup`` arguments have different group mappings or sets,
            or any sensitivity is negative.
    """
    if isinstance(first, PerGroup) != isinstance(second, PerGroup):
        raise TypeError(
            "paired_noise_stddevs requires both streams to have the same "
            f"kind; got first={type(first).__name__}, "
            f"second={type(second).__name__}."
        )
    if noise_multiplier < 0:
        raise ValueError(
            f"noise_multiplier must be non-negative, got {noise_multiplier}"
        )
    _validate_paired_sensitivity(first, label="first")
    _validate_paired_sensitivity(second, label="second")
    if isinstance(first, PerGroup):
        assert isinstance(second, PerGroup)
        if first.groups != second.groups:
            raise ValueError(
                "paired_noise_stddevs requires identical group mappings on "
                "both streams; got "
                f"{len(first.groups)} vs {len(second.groups)} parameter "
                "assignments."
            )
        if set(first.values) != set(second.values):
            raise ValueError(
                "paired_noise_stddevs requires identical group sets on both "
                f"streams; got {sorted(first.values)} vs "
                f"{sorted(second.values)}."
            )
    sum_total = _sum_sensitivity(first) + _sum_sensitivity(second)
    sigma_first = _scale_sensitivity(first, sum_total, noise_multiplier)
    sigma_second = _scale_sensitivity(second, sum_total, noise_multiplier)
    return sigma_first, sigma_second


__all__ = ["per_group_noise_stddev", "paired_noise_stddevs"]
