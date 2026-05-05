"""Wrapper-aware SGD optimizer factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import torchopt
    from torchopt.base import GradientTransformation
except ImportError as exc:
    raise ImportError(
        "torchopt is required for opaque.optimizers. "
        "Install it with: pip install 'torchopt>=0.7.3'"
    ) from exc

from opaque.bounded import BoundedPytree, NoisyPytree
from opaque.core.noise import SecondMomentNoiseOutput


_LR = float | Callable[[Any], Any]


def _unwrap_update_value(updates: Any) -> Any:
    if isinstance(updates, NoisyPytree):
        return updates.pytree
    if isinstance(updates, BoundedPytree):
        raise TypeError(
            "optimizer.update() received BoundedPytree updates that have not "
            "passed through a noise mechanism. Pass NoisyPytree outputs from "
            "a DP mechanism, or unwrap `.pytree` explicitly for non-private use."
        )
    if isinstance(updates, SecondMomentNoiseOutput):
        return _unwrap_update_value(updates.noisy_grads)
    return updates


def sgd(
    lr: _LR,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    *,
    moment_requires_grad: bool = False,
    maximize: bool = False,
) -> GradientTransformation:
    """Create SGD with Opaque's wrapper-aware update API.

    SGD's update is unbiased under additive zero-mean DP noise, so it does not
    consume ``NoisyPytree.noise_stddev``. The wrapper accepts ``NoisyPytree``
    anyway and forwards the privatized pytree to TorchOpt's SGD primitive.
    """
    base = torchopt.sgd(
        lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        moment_requires_grad=moment_requires_grad,
        maximize=maximize,
    )

    def init_fn(params: Any) -> Any:
        return base.init(params)

    def update_fn(
        updates: Any,
        state: Any,
        *,
        params: Any = None,
        inplace: bool = True,
    ) -> tuple[Any, Any]:
        return base.update(
            _unwrap_update_value(updates),
            state,
            params=params,
            inplace=inplace,
        )

    return GradientTransformation(init_fn, update_fn)


__all__ = ["sgd"]
