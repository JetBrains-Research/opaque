"""Server-side per-client clipping aggregate, shipped cloudpickled to IFED.

The returned closure runs in IFED's executor venv, which has ``torch`` and
``ifed-sdk`` but **no opaque**. It must reference nothing from ``opaque.*``;
the default IFED state classes it uses travel by value when the server plan is
built.
"""

# NO `from __future__ import annotations` here: the executor resolves the
# pickled aggregate's type hints at runtime (ifed_server.phases re-types agent
# states from them), so the annotations must be concrete objects cloudpickle
# carries with the closure — not strings evaluated against globals the
# executor doesn't have.
from collections.abc import Callable

from ifed._defaults import AgentState, RoundResult


def make_clipping_aggregate(clipping_norm: float) -> Callable:
    """Build the pickled ``aggregate``: clip each CLIENT's contribution, sum.

    Clipping each client's gradient pytree to L2 norm ``clipping_norm`` before
    summation bounds the sum's sensitivity to any one client by
    ``clipping_norm`` — the contract a central-DP noise mechanism needs, with
    the *client* as the privacy unit.

    Args:
        clipping_norm: Per-client L2 clipping threshold ``C``.

    Returns:
        ``aggregate(states: list[AgentState]) -> RoundResult`` — cloudpickled
        by value into the server plan.
    """
    if clipping_norm <= 0:
        raise ValueError(f"clipping_norm must be > 0, got {clipping_norm}")

    def aggregate(states: list[AgentState]) -> RoundResult:
        import torch

        if not states:
            raise ValueError("aggregate received no agent states")
        total: dict[str, torch.Tensor] = {}
        for state in states:
            if state.params is None:
                raise ValueError("agent state has no params")
            flat = torch.cat([g.flatten() for g in state.params.values()])
            scale = min(1.0, clipping_norm / (float(flat.norm()) + 1e-12))
            for name, grad in state.params.items():
                clipped = grad * scale
                total[name] = total[name] + clipped if name in total else clipped
        return RoundResult(grads=total, count=len(states))

    return aggregate
