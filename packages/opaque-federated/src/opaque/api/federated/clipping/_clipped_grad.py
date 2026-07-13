"""Federated per-client clipped gradients — the twin of central ``clipped_grad``."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

import ifed
from ifed_client import RoundInput

from opaque.api.engine.clipping.types import FixedClipState
from opaque.api.engine.types import ClippedPytree, clipped

from opaque.api.federated.clipping._callbacks import make_clipping_aggregate


def _dataset_of(loss_fn: Callable) -> type:
    """Read the ``ifed.Dataset`` the loss binds to from its annotations."""
    for param in inspect.signature(loss_fn).parameters.values():
        annotation = param.annotation
        if isinstance(annotation, type) and issubclass(annotation, ifed.Dataset):
            return annotation
    raise TypeError(
        "cannot infer the dataset from the loss signature; pass data=<ifed.Dataset "
        "subclass> (jit-compatible losses annotate their data arg as a dict, so "
        "the dataset binding cannot ride the annotation)"
    )


def clipped_grad(
    loss_fn: Callable,
    client: Any,
    *,
    clipping_norm: float,
    params: dict,
    data: type | None = None,
    normalize_by: float | None = None,
    local_batch_size: int = 32,
    population: str = "/hive",
    requirements: dict | None = None,
    dataset_policies: dict | None = None,
    extra_requirements: Sequence[str] = (),
    round_input_timeout: int = 3600,
    round_timeout: float = 3600.0,
) -> tuple[Callable, FixedClipState]:
    """Federated per-CLIENT clipped gradient — the twin of
    ``opaque.dpsgd.clipping.clipped_grad`` with the batch axis being a client
    cohort and IFED executing each round.

    The factory is **eager**: it compiles the loss into an IFED plan
    (``ifed.FunctionalModel`` → TorchScript agent artifact + cloudpickled
    server callbacks, with :func:`make_clipping_aggregate` as the per-client
    clipping ``aggregate``) and uploads it once. The returned ``grad_fn``
    lazily opens ONE interactive IFED task on its first call — sized by the
    cohort's loader (``rounds``, ``cardinality = cohort.size``,
    ``separation = cohort.separation``) — and then drives **one federated
    iteration per application**::

        grad_fn, clip_state = clipped_grad(loss, client, clipping_norm=1.0,
                                           params=params, data=Iris)
        for cohort in loader:                                    # DataLoader
            grads, clip_state = grad_fn(params, cohort, state=clip_state)

    ``grads`` is a ``ClippedPytree`` with ``max_norm = clipping_norm /
    normalize_by`` (default ``normalize_by = cohort.size``), so the
    noise → optimizer → accountant chain downstream is byte-identical to
    central DP-SGD.

    Args:
        loss_fn: Functional loss ``loss_fn(params, data) -> scalar`` with
            explicit params, written in the TorchScript subset (see
            ``ifed.FunctionalModel``).
        client: An ``ifed_client.Client`` session (server or simulation).
        clipping_norm: Per-client L2 clipping threshold ``C``.
        params: Parameter template (names/shapes/dtypes) for compilation;
            actual values travel per round through ``grad_fn``.
        data: The ``ifed.Dataset`` subclass the loss consumes. Defaults to
            reading an ``ifed.Dataset`` annotation off the loss signature.
        normalize_by: Gradient normalization constant; defaults to the cohort
            size, yielding averaged gradients with sensitivity ``C / k``.
        local_batch_size: Agent-side minibatch size for local computation.
        population: Population URI; must match the loader's population.
        requirements: Agent requirements dict (cpuCount, memoryMb, gpu, ...).
        dataset_policies: Per-dataset dataAvailability overrides.
        extra_requirements: Extra pip requirements for the executor venv.
        round_input_timeout: Seconds the *platform* waits for each round input.
        round_timeout: Seconds *this client* waits for each round's result.

    Returns:
        ``(grad_fn, clip_state)`` where ``grad_fn(params, cohort, *, state)``
        returns ``(ClippedPytree, state)`` and ``clip_state`` is the same
        fieldless :class:`FixedClipState` marker central ``clipped_grad``
        threads.
    """
    dataset = data if data is not None else _dataset_of(loss_fn)
    model = ifed.FunctionalModel(
        loss_fn, params=params, data=dataset, batch_size=local_batch_size
    )
    plan = client.compile(
        model,
        population=population,
        aggregate=make_clipping_aggregate(clipping_norm),
        requirements=requirements,
        dataset_policies=dataset_policies,
        extra_requirements=extra_requirements,
    )

    run = None
    origin: object | None = None
    cohort_size = 0
    cohort_separation = 0
    expected_round = 0
    nb = 0.0

    def grad_fn(
        params: dict, cohort: ifed.Cohort, *, state: FixedClipState
    ) -> tuple[ClippedPytree, FixedClipState]:
        nonlocal run, origin, cohort_size, cohort_separation, expected_round, nb
        if cohort.origin is None or cohort.rounds is None or cohort.population is None:
            raise ValueError(
                "cohorts must come from opaque.federated.DataLoader — a raw "
                "Cohort carries no rounds/population/origin to open the run with"
            )
        if run is None:
            if cohort.population.uri != population:
                raise ValueError(
                    f"loader population {cohort.population.uri!r} does not match "
                    f"the population this plan was compiled for ({population!r})"
                )
            nb = float(normalize_by) if normalize_by is not None else float(cohort.size)
            run = plan.open(
                rounds=cohort.rounds,
                cardinality=cohort.size,
                separation=cohort.separation,
                population=cohort.population.uri,
                round_input_timeout=round_input_timeout,
                iteration_timeout=round_timeout,
            )
            grad_fn.run = run
            origin = cohort.origin
            cohort_size = cohort.size
            cohort_separation = cohort.separation
        else:
            if cohort.origin is not origin:
                raise ValueError(
                    "cohort comes from a different DataLoader than round 0's"
                )
            if cohort.size != cohort_size or cohort.separation != cohort_separation:
                raise ValueError(
                    "cohort size/separation changed mid-run; IFED fixes both for "
                    "a task's lifetime"
                )
        if cohort.round != expected_round:
            raise ValueError(
                f"out-of-order cohort: expected round {expected_round}, got {cohort.round}"
            )
        expected_round += 1

        out = run.next(RoundInput(params=params))
        if out.count != cohort.size:
            raise RuntimeError(
                f"round {cohort.round} aggregated {out.count} contributions, "
                f"expected exactly {cohort.size}"
            )
        grads = {name: grad / nb for name, grad in out.grads.items()}
        return clipped(grads, max_norm=clipping_norm / nb), state

    grad_fn.run = None
    return grad_fn, FixedClipState()
