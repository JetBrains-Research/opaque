# Opaque development guidelines

Before planning, implementing, or reviewing a change:

1. Read `AGENTS.md` for repository layout, commands, and workflow.
2. Read `docs/development/architecture-contracts.md` and apply every normative
   active contract whose scope intersects the change.
3. For privacy-sensitive or mathematical work, read
   `docs/development/differential-privacy-review.md`. Trace the claim through
   sensitivity, clipping, noise, sampling, composition, accounting, and auditing
   as applicable.
4. Treat the advisory rules in the architecture document as design feedback, not automatic
   blockers.
5. Do not report missing infrastructure for contracts marked planned.

For implementation work, preserve the architecture contracts while making the
smallest complete change. Use the owning package's public façade in user-facing
code and its `opaque.api.*` implementation tree for implementation. Add or move
tests according to ARC-006 and run the smallest relevant existing validation.

For code review:

- Review the merge-base diff, not an exhaustive inventory of the repository.
- Post only actionable, high-confidence findings on changed lines.
- Treat mathematical correctness and consistency with the implemented privacy
  model as primary review areas, not optional specialist checks.
- For changes whose correctness depends on a theorem, algorithm, or empirical
  result, use web search and URL fetching to inspect the primary literature
  before concluding. Cite the source and precise assumption in the finding.
- Never fabricate a citation or present an unverified literature claim as fact.
  If the evidence is unavailable or ambiguous, state the uncertainty and request
  a derivation, reference, or numerical validation.
- Cite the relevant `ARC-*` or `ADV-*` ID for architecture findings.
- Distinguish normative violations from advisory design suggestions.
- Distinguish a broken privacy guarantee from workload-fidelity or utility
  differences that do not invalidate DP.
- Do not flag a new package, module, import, or export merely because it is new.
- Do not replace semantic reasoning with a hard-coded package/module list.

If a requested change intentionally revises an architecture contract, stop and
propose the policy change explicitly instead of silently working around it.
