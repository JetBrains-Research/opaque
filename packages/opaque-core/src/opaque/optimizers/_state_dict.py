"""Optimizer state serialisation.

Public surface: :func:`state_dict` and :func:`load_state_dict`.  Walks
the chain-state tuple, encoding every tensor leaf and Python primitive
(``int``, ``float``, ``bool``, ``str``, ``None``) into a flat
``{path: value}`` dict.  Anything else (notably ``optree.PyTreeSpec``
on ``AdafactorState``) is **skipped** — those values can be re-derived
from the params at load time, so the saved dict only has to carry
the data that actually changes during training.

Path syntax::

    "<entry-index>.<field-name>"          # dataclass field of a chain entry
    "<entry-index>.<field-name>[<i>]"     # list / tuple element
    "<entry-index>.<field-name>.<key>"    # dict entry

For example, an ``AdamW`` chain ``(AdamState, EmptyState, EmptyState,
EmptyState)`` produces paths like ``"[0].mu.weight"``,
``"[0].nu.weight"``, ``"[0].phi"``, ``"[0].step"``.  Nested chains
(``schedule_free`` wrapping ``adamw``) extend naturally:
``"[0].inner[0].mu.weight"``.

The serialised dict is itself a plain ``dict[str, Any]``, ready for
``torch.save`` / ``torch.load``.  Tensors are detached and cloned at
save time so subsequent training mutations don't reach into the
saved snapshot.

``load_state_dict`` is template-driven: it takes a freshly-initialised
state of the right shape (``opt.init(params)`` on the params used at
save time) and rewrites its leaves from the dict.  Paths missing
from the dict keep the template value, which is forward-compatible
with optimizers that gain new state fields between releases.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch


# Python primitives we serialise verbatim.  Everything else that
# isn't a tensor / dataclass / collection is treated as opaque and
# skipped (re-derived from the template at load time).
_PRIMITIVES = (int, float, bool, str, type(None))


def state_dict(opt_state: Any) -> dict[str, Any]:
    """Serialise a chain optimizer state into a flat dict.

    Tensor leaves are detached and cloned.  Python primitives are
    saved verbatim.  Other objects (``optree.PyTreeSpec``, callables,
    …) are omitted from the dict — they're expected to be re-derived
    from the template at load time.

    Args:
        opt_state: The chain state returned by ``opt.init(params)``
            (and subsequently advanced by ``opt.update``).

    Returns:
        A ``dict[str, Any]`` mapping path strings to tensors and
        primitive values.  Suitable for ``torch.save``.

    Example::

        opt = adamw(lr=1e-3)
        state = opt.init(params)
        # ... train ...
        sd = state_dict(state)
        torch.save(sd, "opt.pt")
    """
    out: dict[str, Any] = {}
    _walk_for_save(opt_state, "", out)
    return out


def load_state_dict(template: Any, sd: dict[str, Any]) -> Any:
    """Rebuild a chain optimizer state from a serialised dict.

    Walks ``template`` (a freshly-initialised state of the same shape
    as the saved one) and replaces each leaf path that's present in
    ``sd``.  Tensor leaves get the saved value cast to the template's
    dtype/device; primitive leaves replace the template's primitive
    value verbatim.

    Args:
        template: A freshly-initialised state from
            ``opt.init(params)`` on the same params (or at least
            the same params *shape*) used when ``state_dict`` was
            called.
        sd: The dict returned by :func:`state_dict`.

    Returns:
        A new chain state with leaves restored from ``sd``.

    Example::

        opt = adamw(lr=1e-3)
        template = opt.init(params)
        sd = torch.load("opt.pt")
        state = load_state_dict(template, sd)
    """
    return _walk_for_load(template, sd, "")


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------


def _walk_for_save(state: Any, prefix: str, out: dict[str, Any]) -> None:
    if isinstance(state, torch.Tensor):
        out[prefix] = state.detach().clone()
        return
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        for f in dataclasses.fields(state):
            sub = f"{prefix}.{f.name}" if prefix else f.name
            _walk_for_save(getattr(state, f.name), sub, out)
        return
    if isinstance(state, tuple):
        for i, v in enumerate(state):
            _walk_for_save(v, f"{prefix}[{i}]", out)
        return
    if isinstance(state, list):
        for i, v in enumerate(state):
            _walk_for_save(v, f"{prefix}[{i}]", out)
        return
    if isinstance(state, dict):
        for k, v in state.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            _walk_for_save(v, sub, out)
        return
    if isinstance(state, _PRIMITIVES):
        out[prefix] = state
        return
    # Opaque value (PyTreeSpec, callable, …) — skip; load uses
    # the template's copy.


def _walk_for_load(template: Any, sd: dict[str, Any], prefix: str) -> Any:
    if isinstance(template, torch.Tensor):
        saved = sd.get(prefix)
        if saved is None:
            return template
        if not isinstance(saved, torch.Tensor):
            raise TypeError(
                f"state_dict[{prefix!r}] expected a torch.Tensor, "
                f"got {type(saved).__name__}"
            )
        return saved.to(dtype=template.dtype, device=template.device)
    if dataclasses.is_dataclass(template) and not isinstance(template, type):
        replacements = {}
        for f in dataclasses.fields(template):
            sub = f"{prefix}.{f.name}" if prefix else f.name
            replacements[f.name] = _walk_for_load(getattr(template, f.name), sd, sub)
        return dataclasses.replace(template, **replacements)
    if isinstance(template, tuple):
        return tuple(
            _walk_for_load(v, sd, f"{prefix}[{i}]") for i, v in enumerate(template)
        )
    if isinstance(template, list):
        return [
            _walk_for_load(v, sd, f"{prefix}[{i}]") for i, v in enumerate(template)
        ]
    if isinstance(template, dict):
        return {
            k: _walk_for_load(v, sd, f"{prefix}.{k}" if prefix else str(k))
            for k, v in template.items()
        }
    if isinstance(template, _PRIMITIVES):
        return sd.get(prefix, template)
    # Opaque — pass through.
    return template


__all__ = ["state_dict", "load_state_dict"]
