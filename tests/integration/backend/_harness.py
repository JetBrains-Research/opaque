"""Engine clipping harness shared by first-party provider tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad
from opaque.backend import use_backend

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.integration.backend._providers import ProviderCase

    from opaque.api.engine.clipping.types import AutoClippedGradAux, ClippedGradAux
    from opaque.types import ClippedPytree, ClipState, SecondMomentClippingOutput


@dataclass(frozen=True)
class ClippingRun:
    """Output and explicitly threaded state from an engine clipping factory."""

    grads: ClippedPytree | SecondMomentClippingOutput
    initial_state: ClipState
    state: ClipState
    aux: ClippedGradAux | AutoClippedGradAux | None


def run_clipping(
    provider: ProviderCase,
    loss_fn: Callable[..., Any],
    *args: Any,
    kind: Literal["fixed", "auto"],
    bound: Any,
    gamma: float = 0.05,
    argnums: int | tuple[int, ...] = 0,
    batch_argnums: int | tuple[int, ...] = (1, 2),
    has_aux: bool = False,
    return_aux: bool = False,
    normalize_by: float = 1.0,
    microbatch_size: int | None = None,
    second_moment: bool = False,
    dtype: Any | None = None,
    compute_dtype: Any | None = None,
) -> ClippingRun:
    """Construct and execute an engine fixed or AUTO-S clipping transform."""
    common = {
        "argnums": argnums,
        "has_aux": has_aux,
        "batch_argnums": batch_argnums,
        "return_aux": return_aux,
        "normalize_by": normalize_by,
        "microbatch_size": microbatch_size,
        "second_moment": second_moment,
        "dtype": dtype,
    }
    with use_backend(provider.backend):
        if kind == "fixed":
            transform, initial_state = clipped_grad(
                loss_fn,
                clipping_norm=bound,
                compute_dtype=compute_dtype,
                **common,
            )
        else:
            if compute_dtype is not None:
                raise ValueError("AUTO-S does not expose an explicit compute_dtype")
            transform, initial_state = auto_clipped_grad(
                loss_fn,
                R=bound,
                gamma=gamma,
                **common,
            )

        result, state = transform(*args, state=initial_state)

    if return_aux:
        grads, aux = result
    else:
        grads, aux = result, None

    return ClippingRun(
        grads=grads,
        initial_state=initial_state,
        state=state,
        aux=aux,
    )


__all__ = ["ClippingRun", "run_clipping"]
