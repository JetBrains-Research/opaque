#!/usr/bin/env python3
"""Create GitHub issues for horizon-allocation follow-ups (audit-style).

Requires ``gh`` authenticated with permission to create issues on
``JetBrains-Research/opaque``. Run from the repository root::

    uv run python .github/scripts/create_horizon_followup_issues.py

Use ``--dry-run`` to print the payloads without calling the API.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


REPO = "JetBrains-Research/opaque"


@dataclass(frozen=True)
class IssueSpec:
    opq_id: str
    title: str
    body: str
    labels: tuple[str, ...]


def _issue(
    opq_id: str,
    short_title: str,
    *,
    paths: str,
    category: str,
    affects_epsilon: str,
    verdict: str,
    claim: str,
    fix_sketch: str,
    notes: str,
    labels: tuple[str, ...],
) -> IssueSpec:
    title = f"{opq_id} — {short_title}"
    body = f"""| Field | Value |
| --- | --- |
| OPQ ID | {opq_id} |
| Location | {paths} |
| Category | {category} |
| ε | {affects_epsilon} |
| Verdict | {verdict} |

## Claim

{claim}

## Fix sketch

{fix_sketch}

## Notes

{notes}
"""
    return IssueSpec(opq_id=opq_id, title=title, body=body, labels=labels)


ISSUES: tuple[IssueSpec, ...] = (
    _issue(
        "OPQ-187",
        "k-out-of-t Gaussian prefix PLD may be conservative beyond hypergeometric / block-cap strategy",
        paths=(
            "`packages/opaque-accounting/src/amplification/random_allocation.rs` "
            "(k-out-of-t prefix); "
            "`packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/amplification/_k_out_of_t.py`"
        ),
        category="privacy-correctness",
        affects_epsilon="yes",
        verdict="PLAUSIBLE",
        claim=(
            "Global k-out-of-t participation charges privacy through a full-horizon block "
            "bound plus a prefix path that uses a conservative hypergeometric cap where an "
            "exact or tighter per-prefix composition may exist. Training and calibration "
            "remain valid (conservative direction) but may over-report ε or over-noise "
            "relative to the true mechanism."
        ),
        fix_sketch=(
            "Derive and implement a tighter prefix PLD for k>1 (or document the gap with "
            "cross-validation against an external reference). Add regression tests that "
            "monotonicity and `per_step(K)` ≡ `pld_at(K)` hold with any tightened bound."
        ),
        notes=(
            "Follow-up from the horizon allocation refactor (PR #317). Initial ship uses "
            "safe-only conservative prefix accounting; this tracks remaining accounting research."
        ),
        labels=(
            "source: audit",
            "pkg: accounting",
            "pkg: dpsgd",
            "severity: medium",
            "impact: privacy",
            "impact: epsilon",
            "needs-validation",
        ),
    ),
    _issue(
        "OPQ-188",
        "Monte Carlo random-allocation accounting not implemented for DP-SGD Gaussian prefix",
        paths=(
            "`packages/opaque-accounting/src/amplification/random_allocation.rs`; "
            "`packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/amplification/_random_allocation.py`"
        ),
        category="privacy-correctness",
        affects_epsilon="yes",
        verdict="PLAUSIBLE",
        claim=(
            "Redrawn random allocation uses an analytic / geometric-convolution path with "
            "conservative discretization. There is no MC estimator with confidence correction "
            "for validation, reproducibility at scale, or sample-size sensitivity analysis "
            "(unlike balls-in-bins MC paths elsewhere in accounting)."
        ),
        fix_sketch=(
            "Add an optional MC reference path for random-allocation prefix PLDs, with "
            "documented confidence bounds and tests that the analytic path dominates the MC "
            "estimate at fixed (n_steps, num_bins, σ). Gate behind dev-only or "
            "`opaque-accounting[cross-validation]` if heavy."
        ),
        notes=(
            "Follow-up from horizon allocation §6 (out of scope for PR #317). Does not block "
            "the safe analytic ship path."
        ),
        labels=(
            "source: audit",
            "pkg: accounting",
            "pkg: dpsgd",
            "severity: low",
            "impact: privacy",
            "impact: epsilon",
            "needs-validation",
        ),
    ),
    _issue(
        "OPQ-189",
        "Optional Cadence / GPU feature-validation for horizon allocation trainer modes",
        paths=(
            "`examples/train_dpsgd_trainer.py`; `.cadence/configs/`; PR #317"
        ),
        category="test-gap",
        affects_epsilon="no",
        verdict="CONFIRMED",
        claim=(
            "Horizon allocation modes (`random_allocation`, `k_out_of_t`) are covered by CPU "
            "unit and trainer guarantee tests but lack the optional multi-hour Cadence + W&B "
            "validation used for other DP training surface changes."
        ),
        fix_sketch=(
            "Run the feature-validation agent protocol (or a minimal Cadence preset) comparing "
            "baseline Poisson vs `random_allocation` / `k_out_of_t` on mellum-kstack for loss "
            "and reported ε trajectories; file results in W&B and link from the PR or release notes."
        ),
        notes=(
            "Optional follow-up; not a merge gate for the library refactor. See "
            "`.claude/agents/feature-validation.md`."
        ),
        labels=(
            "enhancement",
            "triage",
            "pkg: transformers",
            "pkg: dpsgd",
            "area: examples",
            "impact: test-gap",
            "severity: low",
        ),
    ),
    _issue(
        "OPQ-190",
        "Close superseded open PRs after horizon allocation stack (#317) merges",
        paths="GitHub pull requests in the allocation / per-step trainer stack",
        category="process",
        affects_epsilon="no",
        verdict="CONFIRMED",
        claim=(
            "Older stacked trainer PRs (`cursor/dptrainer-allocation-per-step-d1f5`, etc.) "
            "are superseded by the two-PR stack **#314** (sampler schedule fixes) + **#317** "
            "(DpHorizonProcess / `per_step` / trainer), both from the "
            "`cursor/horizon-allocation-processes-d1f5` lineage. Leaving duplicate PRs "
            "open causes review and CI noise."
        ),
        fix_sketch=(
            "After #317 merges (with #314 merged first), close superseded PRs with a short "
            "comment pointing at #314/#317 and delete stale remote branches when safe."
        ),
        notes=(
            "Process hygiene after the allocation accounting unification. Do not close #314 or "
            "#317 — they are the intended merge sequence."
        ),
        labels=("triage", "area: ci"),
    ),
)


def create_issue(spec: IssueSpec, *, dry_run: bool) -> str | None:
    if dry_run:
        print(f"\n--- {spec.opq_id} ---\nTitle: {spec.title}\nLabels: {spec.labels}\n")
        print(spec.body[:400] + "...\n")
        return None
    cmd = [
        "gh",
        "issue",
        "create",
        "--repo",
        REPO,
        "--title",
        spec.title,
        "--body",
        spec.body,
    ]
    for label in spec.labels:
        cmd.extend(["--label", label])
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr or result.stdout, file=sys.stderr)
        return None
    url = (result.stdout or "").strip()
    print(f"{spec.opq_id}: {url}")
    return url


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print issue payloads without creating them.",
    )
    args = parser.parse_args()
    failures = 0
    for spec in ISSUES:
        if create_issue(spec, dry_run=args.dry_run) is None and not args.dry_run:
            failures += 1
    if failures:
        print(
            f"\n{failures} issue(s) failed to create. Ensure `gh auth login` has "
            "`issues: write` on the repository.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
