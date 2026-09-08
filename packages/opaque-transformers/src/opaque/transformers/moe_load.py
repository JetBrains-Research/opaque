"""MoE router-load release helper for hand-written functional loops.

Public façade over :mod:`opaque.api.transformers.moe_load`: the four seams a
DP training loop needs to release the batch router load as a differentially
private estimate (``attach_probe``, ``probe_bounds``, ``router_load_terms``,
``initial_state`` / ``filter_factors`` / ``update``) plus the shared
consumer pieces (``resolve_moe_geometry``, ``decide_trip``,
``check_resume_compatible``, ``summary``, ``telemetry_without_probe``).
``DPTrainer`` reaches the same mechanism through
``TrainingArguments.router_load_release``.
"""

from __future__ import annotations

from opaque.api.transformers.moe_load import (
    PROBE_NAME,
    RESUME_MATCH_FIELDS,
    RouterLoadState,
    attach_probe,
    check_resume_compatible,
    decide_trip,
    filter_factors,
    initial_state,
    is_router_module,
    load_bound,
    monitor_value,
    probe_bounds,
    resolve_moe_geometry,
    router_load_terms,
    summary,
    telemetry_without_probe,
    update,
)

__all__ = [
    "PROBE_NAME",
    "RESUME_MATCH_FIELDS",
    "RouterLoadState",
    "attach_probe",
    "check_resume_compatible",
    "decide_trip",
    "filter_factors",
    "initial_state",
    "is_router_module",
    "load_bound",
    "monitor_value",
    "probe_bounds",
    "resolve_moe_geometry",
    "router_load_terms",
    "summary",
    "telemetry_without_probe",
    "update",
]
