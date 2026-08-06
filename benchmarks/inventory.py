from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from benchmarks.claims import scan_repository
from benchmarks.core import BenchmarkError

if TYPE_CHECKING:
    from pathlib import Path

_CLASSIFICATIONS = {"benchmark", "behavioral_test", "derived", "example"}
_STATUSES = {
    "contradicted",
    "corrected",
    "derived",
    "example",
    "supported",
    "under_investigation",
    "withdrawn",
}


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    statement: str
    classification: str
    status: str
    decision: str
    case: str | None = None
    current_path: str | None = None
    current_match: str | None = None


def _optional_string(value: Any, field: str, claim_id: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BenchmarkError(f"Claim {claim_id!r} field {field!r} must be a string")
    return value


def load_claims(path: Path) -> tuple[ClaimRecord, ...]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise BenchmarkError(f"Cannot read claim inventory {path}: {error}") from error
    rows = document.get("claim")
    if not isinstance(rows, list):
        raise BenchmarkError("Claim inventory must contain [[claim]] records")
    claims: list[ClaimRecord] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BenchmarkError(f"claim[{index}] must be a table")
        claim_id = row.get("id")
        if not isinstance(claim_id, str) or not claim_id:
            raise BenchmarkError(f"claim[{index}].id must be a non-empty string")
        required = {}
        for field in ("statement", "classification", "status", "decision"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise BenchmarkError(f"Claim {claim_id!r} field {field!r} is required")
            required[field] = value
        claims.append(
            ClaimRecord(
                claim_id=claim_id,
                statement=required["statement"],
                classification=required["classification"],
                status=required["status"],
                decision=required["decision"],
                case=_optional_string(row.get("case"), "case", claim_id),
                current_path=_optional_string(
                    row.get("current_path"), "current_path", claim_id
                ),
                current_match=_optional_string(
                    row.get("current_match"), "current_match", claim_id
                ),
            )
        )
    return tuple(claims)


def validate_claim_inventory(
    root: Path,
    claims: tuple[ClaimRecord, ...],
    *,
    case_ids: set[str],
    result_case_ids: set[str],
) -> list[str]:
    errors: list[str] = []
    ids = [claim.claim_id for claim in claims]
    duplicates = sorted({claim_id for claim_id in ids if ids.count(claim_id) > 1})
    if duplicates:
        errors.append(f"duplicate claim ids: {duplicates}")

    for claim in claims:
        if claim.classification not in _CLASSIFICATIONS:
            errors.append(
                f"claim {claim.claim_id}: unknown classification "
                f"{claim.classification!r}"
            )
        if claim.status not in _STATUSES:
            errors.append(f"claim {claim.claim_id}: unknown status {claim.status!r}")
        if (claim.current_path is None) != (claim.current_match is None):
            errors.append(
                f"claim {claim.claim_id}: current_path and current_match must be paired"
            )
        if claim.classification == "benchmark":
            if claim.case is None:
                errors.append(
                    f"claim {claim.claim_id}: benchmark claim requires a case"
                )
            elif claim.case not in case_ids:
                errors.append(
                    f"claim {claim.claim_id}: unknown benchmark case {claim.case!r}"
                )
            elif claim.status == "supported" and claim.case not in result_case_ids:
                errors.append(
                    f"claim {claim.claim_id}: supported claim requires a committed "
                    f"result for {claim.case}"
                )

    findings = scan_repository(root)
    matched_claim_ids: set[str] = set()
    for finding in findings:
        relative = finding.path.relative_to(root).as_posix()
        matches = [
            claim
            for claim in claims
            if claim.current_path == relative
            and claim.current_match is not None
            and claim.current_match in finding.text
        ]
        if not matches:
            errors.append(
                f"unclassified numeric performance claim: "
                f"{relative}:{finding.line}: {finding.text}"
            )
        elif len(matches) > 1:
            errors.append(
                f"numeric claim {relative}:{finding.line} matches multiple records: "
                f"{[claim.claim_id for claim in matches]}"
            )
        else:
            matched_claim_ids.add(matches[0].claim_id)

    errors.extend(
        (f"claim {claim.claim_id}: current_match no longer matches a scanned claim")
        for claim in claims
        if claim.current_path is not None and claim.claim_id not in matched_claim_ids
    )
    return errors


__all__ = ["ClaimRecord", "load_claims", "validate_claim_inventory"]
