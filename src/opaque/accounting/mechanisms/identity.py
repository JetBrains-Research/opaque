"""Identity mechanism — zero privacy loss, composition identity element."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import opaque_accounting as _native

from opaque.accounting.base import (
    DiscretizationConfig,
    DpProcess,
    Pld,
)
from opaque.accounting.discretization import (
    resolve_pld_config,
    serialize_config,
)


@dataclass(frozen=True, slots=True)
class Identity(DpProcess):
    """Identity mechanism — zero privacy loss.

    Identity element of composition:
    ``Identity() | a`` → ``a`` and ``a | Identity()`` → ``a``.
    """

    config: DiscretizationConfig | None = field(default=None, repr=False)

    def pld(self) -> Pld:
        return _native.identity_pld(config=self.config)

    def state_dict(self) -> dict[str, object]:
        d = asdict(self)
        d["type"] = "Identity"
        d["config"] = serialize_config(self.config)
        return d


def identity(
    *,
    discretization: None | float | DiscretizationConfig = None,
) -> DpProcess:
    """Identity mechanism (zero privacy loss).

    Useful as a placeholder or identity element in composition.

    Args:
        discretization: PLD precision config (keyword-only).

    Returns:
        An :class:`Identity` process (ε=0 for any δ).

    Example::

        # Identity has ε=0 for any δ
        proc = acc.identity()
        eps = proc.epsilon_at(1e-5)  # ~0
    """
    config = resolve_pld_config(discretization)
    return Identity(config=config)
