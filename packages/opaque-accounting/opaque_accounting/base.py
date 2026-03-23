"""DpProcess: stateless composable DP process algebra.

Mechanism constructors store parameters as frozen-dataclass fields.
The PLD is computed on demand via the ``pld()`` method, which is automatically
cached via ``@lru_cache`` on each implementation.  Use
:func:`~opaque.accounting.composition.cached` to increase cache size or as
a merge barrier.

Composition Optimizations
~~~~~~~~~~~~~~~~~~~~~~~~~

At construction time, the composition operators detect patterns that can
be reduced to cheaper operations using structural equality (``==``):

**Identity elision** (in ``__or__``):

- ``Identity | a``  →  ``a``
- ``a | Identity``  →  ``a``

**Flatten nested repeats** (in ``__mul__``):

- ``(step * n) * m``  →  ``Repeated(step, n * m)``
  Avoids nested ``self_compose`` calls (2 FFTs instead of 4).

**Merge same-leaf compose** (in ``__or__``):

- ``a | a``          →  ``a * 2``       (self_compose, 2 FFTs not 4)
- ``a * n | a``      →  ``a * (n + 1)``
- ``a | a * n``      →  ``a * (n + 1)``
- ``a * n | a * m``  →  ``a * (n + m)`` (2 FFTs not 6)

**Right-spine merge** (in ``__or__``):

- ``(X | a * n) | a * m``  →  ``X | a * (n + m)``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import TypeAlias

from . import opaque_accounting as _native

Pld: TypeAlias = _native.Pld

__all__ = ["DpProcess", "Pld"]

# Global registry of DpProcess subclasses for polymorphic deserialization.
# Automatically populated when each subclass is defined via __init_subclass__.
_PROCESS_REGISTRY: dict[str, type[DpProcess]] = {}


class DpProcess(ABC):
    """A differential privacy process backed by a PLD.

    Abstract base class for all mechanisms.  Subclasses implement
    :meth:`pld` to compute the Privacy Loss Distribution on demand.
    Results are automatically cached via ``@lru_cache`` on each
    implementation.

    Supports:

    - **Privacy metrics**: epsilon_at(), delta_at(), advantage(), beta_at(), risk_at()
    - **Composition**: ``a | b`` (heterogeneous), ``a * k`` (homogeneous)

    Example::

        import opaque.accounting as acc

        step = acc.poisson(acc.gaussian(1.1), 0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register subclass in the global DpProcess registry."""
        super().__init_subclass__(**kwargs)
        _PROCESS_REGISTRY[cls.__name__] = cls

    @abstractmethod
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        """Compute the Privacy Loss Distribution.

        Results are automatically cached via ``@lru_cache`` on each
        implementation.  Use :func:`~opaque.accounting.composition.cached`
        to increase cache size or as a merge barrier.

        Args:
            discretization: Grid spacing for PLD (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
        """
        ...

    def state_dict(self) -> dict[str, object]:
        """Serialize this process into a plain dict.

        The returned structure should be JSON-serializable and must not
        include cached or computed values (e.g., PLDs).

        Default implementation extracts dataclass fields and recursively
        converts nested DpProcess instances to state dicts.
        """

        def _serialize_value(value: object) -> object:
            """Recursively serialize DpProcess fields to state dicts."""
            if isinstance(value, DpProcess):
                return value.state_dict()
            elif isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return type(value)(_serialize_value(item) for item in value)
            else:
                return value

        # Start with type tag for readability
        data = {"type": self.__class__.__name__}

        # Extract all dataclass fields manually
        for field in fields(self):
            value = getattr(self, field.name)
            data[field.name] = _serialize_value(value)

        return data

    @classmethod
    def from_state_dict(cls, data: dict[str, object]) -> DpProcess:
        """Deserialize a DpProcess from a state dict.

        Recursively reconstructs the process tree using the auto-populated
        _PROCESS_REGISTRY. Each subclass registers itself via __init_subclass__.

        Args:
            data: Dictionary produced by :meth:`state_dict`.

        Returns:
            Reconstructed DpProcess (any subclass).

        Raises:
            ValueError: If type tag is unknown.
        """
        # Extract type tag and make mutable copy
        data = dict(data)
        t = data.pop("type")

        # Look up constructor
        process_cls = _PROCESS_REGISTRY.get(t)
        if process_cls is None:
            raise ValueError(f"Unknown process type: {t}")

        # Recursively deserialize nested processes
        for key in ["inner", "left", "right"]:
            if key in data and isinstance(data[key], dict):
                data[key] = cls.from_state_dict(data[key])

        # Instantiate using dataclass constructor
        return process_cls(**data)

    # -- Privacy metrics (compute PLD each time) -----------------------------

    def epsilon_at(
        self,
        delta: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> float:
        """Smallest ε achieving (ε, δ)-DP.

        Args:
            delta: Failure probability.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        ).epsilon_at(delta)

    def delta_at(
        self,
        epsilon: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> float:
        """Smallest δ achieving (ε, δ)-DP.

        Args:
            epsilon: Privacy budget.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        ).delta_at(epsilon)

    def advantage(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> float:
        """Total-variation advantage (f-DP).

        Args:
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        ).advantage()

    def beta_at(
        self,
        alpha: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> float:
        """Type-II error at given Type-I error α.

        Args:
            alpha: Type-I error rate.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        ).beta_at(alpha)

    def risk_at(
        self,
        prior: float,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> float:
        """Bayes risk under optimal adversary.

        Args:
            prior: Prior probability.
            discretization: Grid spacing (query-time override).
            log_x_mass_truncation_bound: Log tail mass cutoff in x-space (query-time override).
            pessimistic_estimate: Whether to use pessimistic rounding (query-time override).
            max_grid_size: Maximum grid size before coarsening (query-time override).
        """
        return self.pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        ).risk_at(prior)

    # -- CGF path (explicit opt-in) -------------------------------------------

    def cgf(self) -> Pld:
        """Return a CGF-backed PLD (no grid, O(1) composition).

        The CGF (Cumulant Generating Function) path uses Lugannani-Rice
        saddle-point approximation for delta_at/epsilon_at/advantage.
        For beta_at/risk_at, the CGF auto-materializes to PMF on demand.

        Not all mechanisms support CGF. Raises :exc:`NotImplementedError`
        for mechanisms without an analytical CGF (e.g., rectified_gaussian).

        Example::

            proc = acc.poisson(acc.gaussian(1.1), 0.01) * 1000
            eps = proc.cgf().epsilon_at(1e-5)  # fast, no grid
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not have a CGF implementation"
        )

    # -- Composition operators -----------------------------------------------

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        """Return ``(leaf, count)`` for merge optimization.

        Plain processes return ``(self, 1)``.
        Overridden by :class:`Repeated` to return ``(inner, count)``.
        """
        return (self, 1)

    def __mul__(self, count: int) -> DpProcess:
        """Repeat this process *count* times: ``step * 1000``.

        Flattens nested repeats: ``(step * n) * m`` → ``step * (n * m)``.

        Args:
            count: Number of repetitions. Must be positive.
                Use ``count=0`` is not allowed (use :func:`identity` instead).

        Raises:
            ValueError: If count < 1.
        """
        if count < 1:
            raise ValueError(
                f"Repeat count must be >= 1, got {count}. "
                "Use identity() for zero privacy loss."
            )

        from opaque_accounting.composition import Repeated

        leaf, existing = self._leaf_and_count()
        return Repeated(leaf, existing * count)

    def __rmul__(self, count: int) -> DpProcess:
        """Support ``1000 * step`` syntax."""
        return self.__mul__(count)

    def __or__(self, other: DpProcess) -> DpProcess:
        """Compose with another process: ``a | b``.

        Applies identity elision, direct merge, and right-spine merge
        using structural equality (``==``).
        """
        from opaque_accounting.composition import Composed, Repeated
        from opaque_accounting.mechanisms import Identity

        # Identity elision
        if isinstance(self, Identity):
            return other
        if isinstance(other, Identity):
            return self

        # Direct merge: both sides share the same leaf
        left_leaf, left_count = self._leaf_and_count()
        right_leaf, right_count = other._leaf_and_count()
        if left_leaf == right_leaf:
            return Repeated(left_leaf, left_count + right_count)

        # Right-spine merge: (X | a*n) | a*m  →  X | a*(n+m)
        if isinstance(self, Composed):
            r_leaf, r_count = self.right._leaf_and_count()
            if r_leaf == right_leaf:
                merged = Repeated(r_leaf, r_count + right_count)
                return Composed(self.left, merged)

        return Composed(self, other)
