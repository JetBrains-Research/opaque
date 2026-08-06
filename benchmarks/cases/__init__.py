from __future__ import annotations

from benchmarks.cases.accounting import CASE as ACCOUNTING_CASE
from benchmarks.cases.clipping import CASE as CLIPPING_CASE
from benchmarks.cases.determinism import CASE as DETERMINISM_CASE
from benchmarks.cases.dpftrl import CASE as DPFTRL_CASE
from benchmarks.cases.fused_linear_cross_entropy import CASE as FUSED_CE_CASE
from benchmarks.cases.fused_swiglu import CASE as FUSED_SWIGLU_CASE
from benchmarks.cases.moe import CASE as MOE_CASE
from benchmarks.cases.optimizers import CASE as OPTIMIZER_CASE
from benchmarks.cases.training import CASE as TRAINING_CASE
from benchmarks.core import BenchmarkCase, BenchmarkError

_CASES = {
    case.case_id: case
    for case in (
        OPTIMIZER_CASE,
        ACCOUNTING_CASE,
        CLIPPING_CASE,
        DETERMINISM_CASE,
        DPFTRL_CASE,
        FUSED_CE_CASE,
        FUSED_SWIGLU_CASE,
        MOE_CASE,
        TRAINING_CASE,
    )
}


def list_cases() -> tuple[BenchmarkCase, ...]:
    return tuple(_CASES[case_id] for case_id in sorted(_CASES))


def get_case(case_id: str) -> BenchmarkCase:
    try:
        return _CASES[case_id]
    except KeyError as error:
        choices = ", ".join(sorted(_CASES))
        raise BenchmarkError(
            f"Unknown benchmark case {case_id!r}; choose one of: {choices}"
        ) from error


__all__ = ["get_case", "list_cases"]
