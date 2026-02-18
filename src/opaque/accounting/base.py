"""DpProcess: stateless composable DP process algebra.

Mechanism constructors store parameters as frozen-dataclass fields.
The PLD is computed on demand via the ``pld()`` method — each call
recomputes from scratch.  Use :func:`~opaque.accounting.composition.cached`
to memoize.

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

import opaque_accounting as _native

Pld = _native.Pld
PldConfig = _native.PldConfig


class DpProcess(ABC):
    """A differential privacy process backed by a PLD.

    Abstract base class for all mechanisms.  Subclasses implement
    :meth:`pld` to compute the Privacy Loss Distribution on demand.
    Each call recomputes; use :func:`~opaque.accounting.composition.cached`
    for memoization.

    Supports:

    - **Privacy metrics**: epsilon_at(), delta_at(), advantage(), beta_at(), risk_at()
    - **Composition**: ``a | b`` (heterogeneous), ``a * k`` (homogeneous)

    Example::

        import opaque.accounting as acc

        step = acc.poisson(acc.gaussian(1.1), 0.01)
        training = step * 1000
        eps = training.epsilon_at(1e-5)
    """

    @abstractmethod
    def pld(self) -> Pld:
        """Compute the Privacy Loss Distribution.

        Each call recomputes from scratch.  Use
        :func:`~opaque.accounting.composition.cached` to memoize.
        """
        ...

    # -- Privacy metrics (compute PLD each time) -----------------------------

    def epsilon_at(self, delta: float) -> float:
        """Smallest ε achieving (ε, δ)-DP."""
        return self.pld().epsilon_at(delta)

    def delta_at(self, epsilon: float) -> float:
        """Smallest δ achieving (ε, δ)-DP."""
        return self.pld().delta_at(epsilon)

    def advantage(self) -> float:
        """Total-variation advantage (f-DP)."""
        return self.pld().advantage()

    def beta_at(self, alpha: float) -> float:
        """Type-II error at given Type-I error α."""
        return self.pld().beta_at(alpha)

    def risk_at(self, prior: float) -> float:
        """Bayes risk under optimal adversary."""
        return self.pld().risk_at(prior)

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
        """
        from opaque.accounting.nodes import Repeated

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
        from opaque.accounting.nodes import Composed, Identity, Repeated

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
